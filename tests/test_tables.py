from marsrag.ingestion.tables import _markdown_table


def test_table_to_markdown():
    result = _markdown_table([["Parameter", "Value"], ["Crew", "4"]])
    assert "| Parameter | Value |" in result
    assert "| --- | --- |" in result
    assert "| Crew | 4 |" in result
