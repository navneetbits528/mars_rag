
from marsrag.ingestion.chunker import chunk_pages
from marsrag.ingestion.models import PageText


def test_roman_numeral_headings_start_new_sections():
    page = PageText(
        page_number=36,
        text=(
            "Nutrition discussion\n"
            + "Food and nutrition risks affect crew health. " * 30
            + "\nXI. REFERENCES\n"
            + "NASA research reference and bibliography details. " * 30
            + "\nXIII. LIST OF ACRONYMS\n"
            + "AFT Advanced Food Technology Project. " * 30
        ),
    )

    chunks = chunk_pages(
        [page],
        report_no="TEST-REPORT",
        title="Test Report",
        year=2024,
        url="https://example.com",
        status="current",
    )

    nutrition = [
        chunk for chunk in chunks
        if "Food and nutrition risks" in chunk.content
    ]
    references = [
        chunk for chunk in chunks
        if "NASA research reference" in chunk.content
    ]
    acronyms = [
        chunk for chunk in chunks
        if "Advanced Food Technology" in chunk.content
    ]

    assert nutrition
    assert references
    assert acronyms

    assert nutrition[0].section_path == "Introduction"
    assert references[0].section_path == "XI REFERENCES"
    assert acronyms[0].section_path == "XIII LIST OF ACRONYMS"
