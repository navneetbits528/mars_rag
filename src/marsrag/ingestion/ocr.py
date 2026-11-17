from __future__ import annotations

import io

import fitz


def ocr_page(page: fitz.Page, language: str = "eng", dpi_scale: float = 2.0) -> str:
    """OCR one PDF page using the local Tesseract executable."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Install the Python OCR dependencies with `uv sync --extra dev`."
        ) from exc

    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale), alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    try:
        return pytesseract.image_to_string(image, lang=language).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR executable was not found. Install Tesseract OCR, add its "
            "installation directory to Windows PATH, and restart PowerShell."
        ) from exc
