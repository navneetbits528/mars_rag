from __future__ import annotations

import hashlib
import re

import tiktoken

from marsrag.ingestion.chunker import normalize_text, stable_chunk_id
from marsrag.ingestion.models import ChunkRecord, ParentRecord
from marsrag.settings import get_settings


def _encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _split_tokens(text: str, max_tokens: int, encoding) -> list[str]:
    tokens = encoding.encode(text)
    if not tokens:
        return []
    pieces = []
    for start in range(0, len(tokens), max_tokens):
        part = encoding.decode(tokens[start:start + max_tokens]).strip()
        if part:
            pieces.append(part)
    return pieces


def build_parent_child_records(chunks: list[ChunkRecord]) -> tuple[list[ParentRecord], list[ChunkRecord]]:
    """Create deterministic section parents and <=configured-token child chunks."""
    settings = get_settings()
    encoding = _encoding(settings.embed_model)
    groups: dict[tuple[str, str], list[ChunkRecord]] = {}
    for chunk in chunks:
        groups.setdefault((chunk.report_no, chunk.section_path or "Unsectioned"), []).append(chunk)

    parents: list[ParentRecord] = []
    children: list[ChunkRecord] = []
    for (report_no, section_path), section_chunks in groups.items():
        section_chunks.sort(key=lambda item: (item.page, item.chunk_type, item.chunk_id))
        parent_content = "\n\n".join(item.content for item in section_chunks)
        normalized_parent = normalize_text(parent_content)
        parent_id = hashlib.sha256(
            f"{report_no}|{section_path}|{normalized_parent.casefold()}".encode("utf-8")
        ).hexdigest()
        parent = ParentRecord(
            parent_id=parent_id,
            report_no=report_no,
            page_start=min(item.page for item in section_chunks),
            page_end=max(item.page for item in section_chunks),
            section_title=section_chunks[0].section_title,
            section_path=section_path,
            content=parent_content,
            metadata={"child_count": len(section_chunks)},
        )
        parents.append(parent)

        for chunk in section_chunks:
            # Tables and captions remain individually addressable; long text is token-split.
            parts = [chunk.content] if chunk.chunk_type in {"table", "figure_caption"} else _split_tokens(
                chunk.content, settings.child_chunk_tokens, encoding
            )
            for part in parts:
                normalized = normalize_text(part)
                if not normalized:
                    continue
                child_id = stable_chunk_id(
                    report_no, chunk.page, f"{chunk.chunk_type}:{normalized}"
                )
                child = ChunkRecord(
                    chunk_id=child_id,
                    report_no=chunk.report_no,
                    page=chunk.page,
                    title=chunk.title,
                    year=chunk.year,
                    status=chunk.status,
                    url=chunk.url,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    chunk_type=chunk.chunk_type,
                    content=part,
                    embedding_text=f"Section: {chunk.section_title}\nType: {chunk.chunk_type}\n{part}",
                    content_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    metadata={**chunk.metadata, "parent_id": parent_id,
                              "ocr": bool(chunk.metadata.get("ocr"))},
                    parent_id=parent_id,
                )
                children.append(child)
    unique_parents = {item.parent_id: item for item in parents}
    unique_children = {item.chunk_id: item for item in children}
    return list(unique_parents.values()), list(unique_children.values())
