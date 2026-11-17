from marsrag.ingestion.hierarchy import build_parent_child_records
from marsrag.ingestion.models import ChunkRecord


def test_parent_child_records_are_linked_and_deterministic():
    chunk = ChunkRecord("old", "NASA-1", 2, "Report", 2025, "current", "https://example.com",
                        "Radiation", "4 > Crew Health > Radiation", "text",
                        "Mars radiation evidence. " * 100,
                        "Section: Radiation\n" + "Mars radiation evidence. " * 100,
                        "hash", {})
    parents1, children1 = build_parent_child_records([chunk])
    parents2, children2 = build_parent_child_records([chunk])
    assert len(parents1) == 1
    assert children1
    assert all(child.parent_id == parents1[0].parent_id for child in children1)
    assert [child.chunk_id for child in children1] == [child.chunk_id for child in children2]
