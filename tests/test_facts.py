from marsrag.ingestion.facts import ExtractedFact, FactExtractionResult, _record_result
from marsrag.ingestion.models import ChunkRecord


def test_fact_is_kept_only_when_quote_is_verbatim():
    chunk = ChunkRecord("chunk-1", "NASA-1", 8, "Report", 2025, "current", "https://example.com",
                        "Crew", "Crew", "text", "The nominal crew size is four astronauts.",
                        "The nominal crew size is four astronauts.", "hash", {})
    result = FactExtractionResult(facts=[
        ExtractedFact(parameter="crew_size", value="4", unit="people",
                      context="Nominal mission crew", quote="The nominal crew size is four astronauts."),
        ExtractedFact(parameter="transit_duration", value="180", unit="days",
                      context=None, quote="The transit takes about 180 days.")
    ])
    verified = []
    _record_result(result, [chunk], verified)
    assert len(verified) == 1
    assert verified[0]["source_chunk_id"] == "chunk-1"
    assert verified[0]["quote"] == "The nominal crew size is four astronauts."
