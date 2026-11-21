from marsrag.ingestion.cleaner import clean_pages
from marsrag.ingestion.models import PageText


def test_cleaner_preserves_page_metadata():
    pages = [
        PageText(i, f"Shared report header\nUnique content for page {i}\nShared footer", {"ocr": i == 2})
        for i in range(1, 4)
    ]
    cleaned = clean_pages(pages).pages[1]
    assert cleaned.metadata == {"ocr": True}
    assert "Unique content for page 2" in cleaned.text
