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


def test_child_ids_differ_when_section_path_changes():
    from marsrag.ingestion.hierarchy import build_parent_child_records

    def make_chunk(section_path):
        return ChunkRecord(
            "old", "NASA-1", 2, "Report", 2025, "current",
            "https://example.com", "Nutrition", section_path, "text",
            "Mars nutrition evidence. " * 100,
            "Section: Nutrition\nMars nutrition evidence. " * 100,
            "hash", {},
        )

    _, children_a = build_parent_child_records([
        make_chunk("Nutrition")
    ])
    _, children_b = build_parent_child_records([
        make_chunk("XI REFERENCES")
    ])

    assert children_a[0].chunk_id != children_b[0].chunk_id