from marsrag.ingestion.models import PageText
from marsrag.ingestion.cleaner import clean_pages


def test_repeated_header_removed():
    pages = [
        PageText(
            i,
            f"NASA MARS REPORT\nPage {i} contains meaningful content and details.\n{i}"
        )
        for i in range(1, 4)
    ]

    result = clean_pages(pages)

    assert "nasa mars report" in result.removed_repeated_lines