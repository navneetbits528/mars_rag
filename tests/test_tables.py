from marsrag.ingestion.tables import _is_structured_table, _markdown_table


def test_table_to_markdown():
    result = _markdown_table([["Parameter", "Value"], ["Crew", "4"]])
    assert "| Parameter | Value |" in result
    assert "| --- | --- |" in result
    assert "| Crew | 4 |" in result


def test_real_table_shape_is_accepted():
    assert _is_structured_table([
        ["Parameter", "Value", "Unit"],
        ["Crew", "4", "people"],
        ["Duration", "180", "days"],
    ])


def test_page_like_paragraph_cells_are_rejected():
    assert not _is_structured_table([
        ["This is a long page paragraph that was incorrectly split by the PDF table detector. " * 12, ""],
        ["Another paragraph-like block with no real tabular structure. " * 12, ""],
    ])


def test_single_row_is_not_a_table():
    assert not _is_structured_table([["A", "B", "C"]])
