from marsrag.ingestion.glossary import extract_glossary, tag_chunk_acronyms
from marsrag.ingestion.models import ChunkRecord, PageText


def _chunk():
    return ChunkRecord("id", "NASA-1", 1, "Report", 2025, "current", "https://example.com",
                       "Section", "1 > Section", "text", "ISRU supports Mars missions.",
                       "ISRU supports Mars missions.", "hash", {})


def test_extract_glossary_and_tag_chunk():
    pages = [PageText(1, "In Situ Resource Utilization (ISRU) can produce propellant.")]
    glossary = extract_glossary(pages)
    assert any(item.acronym == "ISRU" and item.expansion == "In Situ Resource Utilization"
               for item in glossary)
    chunk = _chunk()
    tag_chunk_acronyms([chunk], glossary)
    assert chunk.metadata["acronym_expansions"]["ISRU"] == "In Situ Resource Utilization"
    assert "ISRU = In Situ Resource Utilization" in chunk.embedding_text
