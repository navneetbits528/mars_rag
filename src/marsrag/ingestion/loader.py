from pathlib import Path

import pymupdf

from marsrag.ingestion.models import PageText
from marsrag.ingestion.ocr import ocr_page
from marsrag.settings import get_settings


def _layout_aware_text(page) -> str:
    """Extract text in reading order, detecting clear two-column layouts.

    PyMuPDF's ``sort=True`` sorts by coordinates, but can interleave lines from
    adjacent columns. This helper only switches to column-major ordering when
    the page has a clear vertical gutter with text blocks on both sides.
    Otherwise it keeps PyMuPDF's normal sorted extraction.
    """
    blocks = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text = (block[4] or "").strip()
        if text:
            blocks.append((float(block[0]), float(block[1]), float(block[2]),
                           float(block[3]), text))

    if len(blocks) < 4:
        return (page.get_text("text", sort=True) or "").strip()

    page_width = float(page.rect.width)
    intervals = sorted((b[0], b[2]) for b in blocks if b[2] > b[0])
    merged = []
    for left, right in intervals:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)

    candidates = []
    for (_, left_end), (right_start, _) in zip(merged, merged[1:]):
        gap = right_start - left_end
        if gap >= max(18.0, page_width * 0.035):
            midpoint = (left_end + right_start) / 2
            left_count = sum(1 for b in blocks if (b[0] + b[2]) / 2 < midpoint)
            right_count = sum(1 for b in blocks if (b[0] + b[2]) / 2 > midpoint)
            if left_count >= 2 and right_count >= 2:
                balance = min(left_count, right_count) / max(left_count, right_count)
                candidates.append((gap * balance, left_end, right_start))

    if not candidates:
        return (page.get_text("text", sort=True) or "").strip()

    _, gutter_left, gutter_right = max(candidates)
    left_blocks, right_blocks, spanning = [], [], []
    for block in blocks:
        x0, y0, x1, y1, text = block
        if x0 < gutter_left and x1 > gutter_right:
            spanning.append(block)
        elif (x0 + x1) / 2 <= (gutter_left + gutter_right) / 2:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    # Full-width titles/captions generally precede the column content.
    spanning.sort(key=lambda b: (b[1], b[0]))
    left_blocks.sort(key=lambda b: (b[1], b[0]))
    right_blocks.sort(key=lambda b: (b[1], b[0]))
    ordered = spanning + left_blocks + right_blocks
    return "\n".join(block[4] for block in ordered).strip()


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract page text and OCR only pages that appear scanned/image-only."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    settings = get_settings()
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"Encrypted PDF requires a password: {path}")
        for index, page in enumerate(doc):
            text = _layout_aware_text(page)
            metadata = {"ocr": False, "layout_order": "columns_or_blocks"}
            if settings.ocr_enabled and len(text) < settings.ocr_min_text_chars:
                text = ocr_page(page, language=settings.ocr_language)
                metadata["ocr"] = bool(text)
                metadata["layout_order"] = "ocr" if text else "empty"
            pages.append(PageText(index + 1, text, metadata))
    return pages
