from pathlib import Path

import fitz

from marsrag.ingestion.models import PageText
from marsrag.ingestion.ocr import ocr_page
from marsrag.settings import get_settings


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract page text and OCR only pages that appear to be scanned/image-only."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    settings = get_settings()
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"Encrypted PDF requires a password: {path}")
        for index, page in enumerate(doc):
            text = (page.get_text("text", sort=True) or "").strip()
            metadata = {"ocr": False}
            if settings.ocr_enabled and len(text) < settings.ocr_min_text_chars:
                text = ocr_page(page, language=settings.ocr_language)
                metadata["ocr"] = bool(text)
            pages.append(PageText(index + 1, text, metadata))
    return pages
