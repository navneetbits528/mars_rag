from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from marsrag.ingestion.models import ChunkRecord, PageText
from marsrag.ingestion.chunker import stable_chunk_id, normalize_text

_FIGURE_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.)\s+\d+(?:[.-]\d+)*\s*[:.]?\s*.+", re.I)


def _markdown_table(rows: list[list[str | None]]) -> str:
    cleaned = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    separator = ["---"] * width
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)


def extract_table_chunks(pdf_path: str | Path, pages: list[PageText], *, report_no: str,
                         title: str, year: int, url: str, status: str) -> list[ChunkRecord]:
    page_lookup = {p.page_number: p for p in pages}
    chunks: list[ChunkRecord] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, table in enumerate(page.extract_tables() or [], start=1):
                markdown = _markdown_table(table)
                if not markdown:
                    continue
                source_text = page_lookup.get(page_number).text if page_number in page_lookup else ""
                lines = [line.strip() for line in source_text.splitlines() if line.strip()]
                caption = next((line for line in reversed(lines)
                                if len(line) <= 240 and (line.lower().startswith("table ") or
                                   line.lower().startswith("exhibit "))), "")
                content = f"{caption}\n\n{markdown}" if caption else markdown
                normalized = normalize_text(content)
                chunk_id = stable_chunk_id(report_no, page_number, f"table:{normalized}")
                chunks.append(ChunkRecord(
                    chunk_id, report_no, page_number, title, int(year), status, url,
                    caption or "Table", caption or "Tables", "table", content,
                    f"Table: {caption}\n{content}" if caption else f"Table\n{content}",
                    __import__("hashlib").sha256(normalized.encode()).hexdigest(),
                    {"table_index": table_index, "ocr": bool(page_lookup.get(page_number) and
                     page_lookup[page_number].metadata.get("ocr"))},
                ))
    return chunks


def extract_figure_caption_chunks(pages: list[PageText], *, report_no: str, title: str,
                                  year: int, url: str, status: str) -> list[ChunkRecord]:
    import hashlib
    chunks: list[ChunkRecord] = []
    for page in pages:
        for line in page.text.splitlines():
            caption = line.strip()
            if not _FIGURE_CAPTION.match(caption):
                continue
            normalized = normalize_text(caption)
            chunks.append(ChunkRecord(
                stable_chunk_id(report_no, page.page_number, f"figure:{normalized}"),
                report_no, page.page_number, title, int(year), status, url,
                "Figures", "Figures", "figure_caption", caption,
                f"Figure caption: {caption}", hashlib.sha256(normalized.encode()).hexdigest(),
                {"ocr": bool(page.metadata.get("ocr"))},
            ))
    return chunks
