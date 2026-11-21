from pathlib import Path

import pymupdf

from marsrag.ingestion.models import PageText
from marsrag.ingestion.ocr import ocr_page
from marsrag.settings import get_settings


def _text_blocks(page):
    """Return non-empty text blocks as (x0, y0, x1, y1, text)."""
    result = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text = (block[4] or "").strip()
        if text:
            result.append((float(block[0]), float(block[1]), float(block[2]), float(block[3]), text))
    return result


def _column_groups(blocks, page_width, page_height):
    """Cluster paragraph blocks by their left edge; ignore page furniture."""
    candidates = [
        b for b in blocks
        if len(b[4]) >= 40
        and b[1] < page_height * 0.82
        and b[3] < page_height * 0.86
        and 0 < (b[2] - b[0]) < page_width * 0.48
    ]
    if len(candidates) < 4:
        return []

    # Group blocks whose left edges are close. This works for narrow and wide
    # columns and avoids treating paragraph line wrapping as separate columns.
    gap = max(45.0, page_width * 0.075)
    by_left = sorted(candidates, key=lambda b: b[0])
    groups = [[by_left[0]]]
    for block in by_left[1:]:
        current_mean = sum(x[0] for x in groups[-1]) / len(groups[-1])
        if block[0] - current_mean > gap:
            groups.append([block])
        else:
            groups[-1].append(block)

    # A genuine column should have at least two substantial paragraph blocks.
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return []
    return groups


def _layout_aware_text(page) -> str:
    """Read multi-column pages one column at a time, left to right."""
    blocks = _text_blocks(page)
    if len(blocks) < 4:
        return (page.get_text("text", sort=True) or "").strip()

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    groups = _column_groups(blocks, page_width, page_height)
    if len(groups) < 2:
        return (page.get_text("text", sort=True) or "").strip()

    columns = []
    for group in groups:
        left = sum(b[0] for b in group) / len(group)
        columns.append((left, []))

    furniture = []
    for block in blocks:
        x0, y0, x1, y1, text = block
        width = x1 - x0
        # Wide blocks (titles, banners, footer links) and bottom-page items
        # are kept outside the column bodies.
        if width >= page_width * 0.48 or y0 >= page_height * 0.82 or y1 >= page_height * 0.86:
            furniture.append(block)
            continue
        center_x = (x0 + x1) / 2
        # Assign by left edge, not center: column widths can differ greatly.
        idx = min(range(len(columns)), key=lambda i: abs(x0 - columns[i][0]))
        columns[idx][1].append(block)

    for _, column in columns:
        column.sort(key=lambda b: (b[1], b[0]))

    top = sorted((b for b in furniture if b[3] <= page_height * 0.18), key=lambda b: (b[1], b[0]))
    bottom = sorted((b for b in furniture if b not in top), key=lambda b: (b[1], b[0]))

    ordered = top[:]
    for _, column in sorted(columns, key=lambda item: item[0]):
        ordered.extend(column)
    ordered.extend(bottom)
    return "\n".join(b[4] for b in ordered).strip()


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
