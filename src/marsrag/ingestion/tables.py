from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pdfplumber

from marsrag.ingestion.models import ChunkRecord, PageText
from marsrag.ingestion.chunker import stable_chunk_id, normalize_text

_FIGURE_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.)\s+\d+(?:[.-]\d+)*\s*[:.]?\s*.+", re.I)


def _clean_table(rows: list[list[str | None]]) -> list[list[str]]:
    cleaned = [[re.sub(r"\s+", " ", cell or "").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def _is_structured_table(rows: list[list[str | None]]) -> bool:
    """Reject paragraph-like false positives returned by PDF table heuristics."""
    cleaned = _clean_table(rows)
    if len(cleaned) < 2 or max(map(len, cleaned), default=0) < 2:
        return False
    width = max(map(len, cleaned))
    if width < 2:
        return False
    populated = [cell for row in cleaned for cell in row if cell]
    if len(populated) < 4:
        return False
    # A table extractor that returns a whole page as one or two giant cells is
    # almost always detecting page layout, not a tabular structure.
    if max((len(cell) for cell in populated), default=0) > 700:
        return False
    rows_with_multiple_cells = sum(sum(bool(cell) for cell in row) >= 2 for row in cleaned)
    if rows_with_multiple_cells < 2:
        return False
    return True


def _markdown_table(rows: list[list[str | None]]) -> str:
    cleaned = _clean_table(rows)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
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
                if not _is_structured_table(table):
                    continue
                markdown = _markdown_table(table)
                if not markdown:
                    continue
                source_page = page_lookup.get(page_number)
                source_text = source_page.text if source_page else ""
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
                    hashlib.sha256(normalized.encode()).hexdigest(),
                    {"table_index": table_index, "ocr": bool(source_page and
                     source_page.metadata.get("ocr"))},
                ))
    return chunks


def extract_figure_caption_chunks(pages: list[PageText], *, report_no: str, title: str,
                                  year: int, url: str, status: str) -> list[ChunkRecord]:
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
