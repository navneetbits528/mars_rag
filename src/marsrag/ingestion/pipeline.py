from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken

from marsrag.db import get_pool
from marsrag.ingestion.cleaner import clean_pages
from marsrag.ingestion.chunker import chunk_pages
from marsrag.ingestion.facts import batch_fact_chunks, extract_verified_facts
from marsrag.ingestion.glossary import extract_glossary, tag_chunk_acronyms
from marsrag.ingestion.hierarchy import build_parent_child_records
from marsrag.ingestion.loader import load_pdf_pages
from marsrag.ingestion.models import ChunkRecord, ParentRecord
from marsrag.ingestion.tables import extract_figure_caption_chunks, extract_table_chunks
from marsrag.llm import embed_documents, usage_tracker
from marsrag.settings import get_settings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enc(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text, model):
    return len(_enc(model).encode(text))


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest.json must contain a JSON array.")
    required = {"file", "title", "report_no", "year", "url", "status"}
    for item in data:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Manifest entry missing fields: {sorted(missing)}")
        if item["status"] not in {"current", "superseded"}:
            raise ValueError("Invalid manifest status.")
    return data


def _old_checksum(report_no):
    with get_pool().connection() as conn:
        row = conn.execute("SELECT file_sha256 FROM documents WHERE report_no=%s", (report_no,)).fetchone()
        return row["file_sha256"] if row else None


def _existing_ids(ids):
    if not ids:
        return set()
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT chunk_id FROM chunks WHERE chunk_id=ANY(%s)", (ids,)).fetchall()
        return {row["chunk_id"] for row in rows}


def _batches(texts, model, token_limit, item_limit, max_input_tokens):
    encoding = _enc(model)
    batch, tokens = [], 0
    for text in texts:
        count = len(encoding.encode(text))
        if count > max_input_tokens:
            raise ValueError(
                f"One embedding input has {count} tokens; configured per-input limit is "
                f"{max_input_tokens}. Split or reduce this table/chunk before ingestion."
            )
        if batch and (tokens + count > token_limit or len(batch) >= item_limit):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += count
    if batch:
        yield batch


def _prepare_entry(entry):
    path = Path(entry["file"])
    if not path.is_file():
        print(f"SKIP missing PDF: {path}")
        return None
    checksum = sha256_file(path)
    pages = clean_pages(load_pdf_pages(path)).pages
    base_chunks = chunk_pages(
        pages, report_no=entry["report_no"], title=entry["title"], year=entry["year"],
        url=entry["url"], status=entry["status"]
    )
    glossary = extract_glossary(pages)
    tag_chunk_acronyms(base_chunks, glossary)
    table_chunks = extract_table_chunks(
        path, pages, report_no=entry["report_no"], title=entry["title"], year=entry["year"],
        url=entry["url"], status=entry["status"]
    )
    figure_chunks = extract_figure_caption_chunks(
        pages, report_no=entry["report_no"], title=entry["title"], year=entry["year"],
        url=entry["url"], status=entry["status"]
    )
    all_parent_inputs = base_chunks + table_chunks + figure_chunks
    tag_chunk_acronyms(all_parent_inputs, glossary)
    parents, children = build_parent_child_records(all_parent_inputs)
    return {
        "entry": entry, "path": path, "checksum": checksum, "pages": pages,
        "parents": parents, "chunks": children, "glossary": glossary,
        "ocr": any(bool(page.metadata.get("ocr")) for page in pages),
        "table_count": len(table_chunks), "figure_caption_count": len(figure_chunks),
    }


def _prepare(entries):
    return [prepared for entry in entries if (prepared := _prepare_entry(entry)) is not None]


def _request_estimate(prepared, chunks_to_embed):
    settings = get_settings()
    embedding_batches = list(_batches(
        [chunk.embedding_text for chunk in chunks_to_embed], settings.embed_model,
        settings.embedding_batch_token_limit, settings.embedding_batch_item_limit,
        settings.embedding_max_input_tokens
    ))
    facts_requests = 0
    if settings.facts_extraction_enabled:
        facts_requests = sum(len(batch_fact_chunks(item["chunks"])) for item in prepared)
    return embedding_batches, facts_requests


def estimate_ingest(entries):
    settings = get_settings()
    prepared = _prepare(entries)
    all_chunks = [chunk for item in prepared for chunk in item["chunks"]]
    tokens = sum(count_tokens(chunk.embedding_text, settings.embed_model) for chunk in all_chunks)
    embedding_batches, facts_requests = _request_estimate(prepared, all_chunks)
    total_requests = len(embedding_batches) + facts_requests
    return {
        "reports": [{"report_no": item["entry"]["report_no"], "pages": len(item["pages"]),
                     "chunks": len(item["chunks"]), "tables": item["table_count"],
                     "figure_captions": item["figure_caption_count"], "ocr_pages": sum(
                         bool(page.metadata.get("ocr")) for page in item["pages"])} for item in prepared],
        "chunks": len(all_chunks), "embedding_tokens": tokens,
        "estimated_embedding_cost_usd": round(tokens * settings.embedding_price_per_1m_usd / 1_000_000, 8),
        "embedding_api_requests": len(embedding_batches), "fact_extraction_api_requests": facts_requests,
        "total_estimated_api_requests": total_requests, "request_budget": settings.openai_max_requests,
        "fits_request_budget": total_requests <= settings.openai_max_requests,
    }


def _store_prepared(item, vectors, facts):
    settings = get_settings()
    entry, path, checksum = item["entry"], item["path"], item["checksum"]
    parents: list[ParentRecord] = item["parents"]
    chunks: list[ChunkRecord] = item["chunks"]
    with get_pool().connection() as conn:
        with conn.transaction():
            conn.execute(
                """INSERT INTO documents
                (report_no,title,year,url,status,supersedes,file_path,file_sha256,ocr,corpus_version,ingested_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT(report_no) DO UPDATE SET title=EXCLUDED.title,year=EXCLUDED.year,
                url=EXCLUDED.url,status=EXCLUDED.status,supersedes=EXCLUDED.supersedes,
                file_path=EXCLUDED.file_path,file_sha256=EXCLUDED.file_sha256,ocr=EXCLUDED.ocr,
                corpus_version=EXCLUDED.corpus_version,ingested_at=now()""",
                (entry["report_no"], entry["title"], int(entry["year"]), entry["url"],
                 entry["status"], entry.get("supersedes"), str(path), checksum, item["ocr"],
                 settings.corpus_version),
            )
            conn.execute("DELETE FROM facts WHERE report_no=%s", (entry["report_no"],))
            conn.execute("DELETE FROM glossary WHERE report_no=%s", (entry["report_no"],))
            for glossary_entry in item["glossary"]:
                conn.execute(
                    """INSERT INTO glossary(acronym,expansion,report_no,page,quote)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT(acronym) DO UPDATE SET expansion=EXCLUDED.expansion,
                    report_no=EXCLUDED.report_no,page=EXCLUDED.page,quote=EXCLUDED.quote""",
                    (glossary_entry.acronym, glossary_entry.expansion, entry["report_no"],
                     glossary_entry.page, glossary_entry.quote),
                )
            parent_ids = [parent.parent_id for parent in parents]
            for parent in parents:
                conn.execute(
                    """INSERT INTO parents(parent_id,report_no,page_start,page_end,section_title,
                    section_path,content,metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(parent_id) DO UPDATE SET page_start=EXCLUDED.page_start,
                    page_end=EXCLUDED.page_end,section_title=EXCLUDED.section_title,
                    section_path=EXCLUDED.section_path,content=EXCLUDED.content,metadata=EXCLUDED.metadata""",
                    (parent.parent_id, parent.report_no, parent.page_start, parent.page_end,
                     parent.section_title, parent.section_path, parent.content,
                     json.dumps(parent.metadata)),
                )
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            for chunk in chunks:
                vector = vectors.get(chunk.embedding_text)
                if vector is None:
                    # Existing chunks can retain their vector; only their metadata/parent link changes.
                    existing = conn.execute("SELECT 1 FROM chunks WHERE chunk_id=%s", (chunk.chunk_id,)).fetchone()
                    if existing:
                        conn.execute(
                            """UPDATE chunks SET parent_id=%s,metadata=%s::jsonb,section_title=%s,
                            section_path=%s,chunk_type=%s WHERE chunk_id=%s""",
                            (chunk.parent_id, json.dumps(chunk.metadata), chunk.section_title,
                             chunk.section_path, chunk.chunk_type, chunk.chunk_id),
                        )
                        continue
                    raise RuntimeError(f"Missing embedding for new chunk {chunk.chunk_id}")
                vector_literal = "[" + ",".join(f"{float(value):.10f}" for value in vector) + "]"
                conn.execute(
                    """INSERT INTO chunks
                    (chunk_id,report_no,page,title,year,status,url,section_title,section_path,chunk_type,
                     parent_id,content,embedding,metadata,content_sha256)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s::jsonb,%s)
                    ON CONFLICT(chunk_id) DO UPDATE SET title=EXCLUDED.title,year=EXCLUDED.year,
                    status=EXCLUDED.status,url=EXCLUDED.url,section_title=EXCLUDED.section_title,
                    section_path=EXCLUDED.section_path,chunk_type=EXCLUDED.chunk_type,
                    parent_id=EXCLUDED.parent_id,content=EXCLUDED.content,embedding=EXCLUDED.embedding,
                    metadata=EXCLUDED.metadata,content_sha256=EXCLUDED.content_sha256""",
                    (chunk.chunk_id, chunk.report_no, chunk.page, chunk.title, chunk.year, chunk.status,
                     chunk.url, chunk.section_title, chunk.section_path, chunk.chunk_type, chunk.parent_id,
                     chunk.content, vector_literal, json.dumps(chunk.metadata), chunk.content_sha256),
                )
            if chunk_ids:
                conn.execute("DELETE FROM chunks WHERE report_no=%s AND NOT (chunk_id=ANY(%s))",
                             (entry["report_no"], chunk_ids))
            else:
                conn.execute("DELETE FROM chunks WHERE report_no=%s", (entry["report_no"],))
            if parent_ids:
                conn.execute("DELETE FROM parents WHERE report_no=%s AND NOT (parent_id=ANY(%s))",
                             (entry["report_no"], parent_ids))
            else:
                conn.execute("DELETE FROM parents WHERE report_no=%s", (entry["report_no"],))
            for fact in facts:
                conn.execute(
                    """
                    INSERT INTO facts(
                        fact_id,
                        parameter,
                        value,
                        unit,
                        context,
                        domain,
                        category,
                        report_no,
                        page,
                        source_chunk_id,
                        quote
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(fact_id) DO UPDATE SET
                        parameter = EXCLUDED.parameter,
                        value = EXCLUDED.value,
                        unit = EXCLUDED.unit,
                        context = EXCLUDED.context,
                        domain = EXCLUDED.domain,
                        category = EXCLUDED.category,
                        quote = EXCLUDED.quote
                    """,
                    (
                        fact["fact_id"],
                        fact["parameter"],
                        fact["value"],
                        fact.get("unit"),
                        fact.get("context"),
                        fact.get("domain", "General"),
                        fact.get("category", "Other"),
                        fact["report_no"],
                        fact["page"],
                        fact["source_chunk_id"],
                        fact["quote"],
                    ),
                )
            conn.execute(
                """INSERT INTO ingest_files(file_path,report_no,file_sha256,processed_at)
                VALUES (%s,%s,%s,now()) ON CONFLICT(file_path) DO UPDATE SET
                report_no=EXCLUDED.report_no,file_sha256=EXCLUDED.file_sha256,processed_at=now()""",
                (str(path), entry["report_no"], checksum),
            )


def ingest_entries(entries, force=False):
    settings = get_settings()
    prepared, unchanged = [], 0
    for entry in entries:
        path = Path(entry["file"])
        if not path.is_file():
            print(f"SKIP missing PDF: {path}")
            continue
        checksum = sha256_file(path)
        if not force and _old_checksum(entry["report_no"]) == checksum:
            unchanged += 1
            print(f"UNCHANGED {entry['report_no']}: checksum matches.")
            continue
        item = _prepare_entry(entry)
        if item:
            prepared.append(item)

    all_chunks = [chunk for item in prepared for chunk in item["chunks"]]
    existing = _existing_ids([chunk.chunk_id for chunk in all_chunks])
    missing = [chunk for chunk in all_chunks if chunk.chunk_id not in existing]
    batches, facts_requests = _request_estimate(prepared, missing)
    token_estimate = sum(count_tokens(text, settings.embed_model)
                         for batch in batches for text in batch)
    cost_estimate = token_estimate * settings.embedding_price_per_1m_usd / 1_000_000
    total_requests = len(batches) + facts_requests
    remaining = settings.openai_max_requests - usage_tracker.totals.api_requests
    print(f"Prepared {len(all_chunks)} child chunks; {len(missing)} need embeddings.")
    print(f"Embedding requests={len(batches)}; optional fact requests={facts_requests}; remaining budget={remaining}.")
    print(f"Estimated embedding tokens={token_estimate:,}; embedding cost=${cost_estimate:.6f}")
    if total_requests > remaining:
        raise RuntimeError("Phase 3 ingest exceeds remaining API request budget. Reduce PDFs or disable fact extraction.")

    vectors = {}
    for batch in batches:
        result = embed_documents(batch)
        for text, vector in zip(batch, result, strict=True):
            vectors.setdefault(text, vector)

    facts_by_report = {}
    if settings.facts_extraction_enabled:
        for item in prepared:
            facts_by_report[item["entry"]["report_no"]] = extract_verified_facts(item["chunks"])

    for item in prepared:
        report_no = item["entry"]["report_no"]
        _store_prepared(item, vectors, facts_by_report.get(report_no, []))

    return {
        "reports_processed": len(prepared), "unchanged_skipped": unchanged,
        "chunks_total": len(all_chunks), "chunks_embedded": len(missing),
        "parents_total": sum(len(item["parents"]) for item in prepared),
        "tables_total": sum(item["table_count"] for item in prepared),
        "figure_captions_total": sum(item["figure_caption_count"] for item in prepared),
        "ocr_reports": sum(bool(item["ocr"]) for item in prepared),
        "facts_total": sum(len(value) for value in facts_by_report.values()),
        "embedding_requests": len(batches), "embedding_tokens_estimated": token_estimate,
        "estimated_embedding_cost_usd": cost_estimate,
        "actual_embedding_tokens_estimated": usage_tracker.totals.embedding_tokens,
        "actual_cost_usd_estimated": usage_tracker.totals.estimated_cost_usd,
    }
