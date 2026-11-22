from unittest.mock import patch

import pytest

from marsrag.retrieval import keyword


def test_search_keyword_returns_results():
    expected = [{"chunk_id": "chunk-1", "report_no": "NASA-HRP-EVIDENCE"}]
    with patch.object(keyword, "fetch_all", return_value=[]) as mock_fetch:
        results = keyword.search_keyword(
            "nutritional risks spaceflight", top_k=3
        )

    assert results == []
    mock_fetch.assert_called_once()
    sql, params = mock_fetch.call_args.args
    assert "content_tsv" in sql
    assert "websearch_to_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert "report_no = %s" in sql
    assert "status = 'current'" in sql
    assert params[1] == "NASA-HRP-EVIDENCE"
    assert params[-1] == 3


def test_keyword_passes_optional_metadata_filters():
    with patch.object(keyword, "fetch_all", return_value=[]) as mock_fetch:
        keyword.search_keyword(
            "nutrition", section_path="V FOOD SAFETY", chunk_type="table"
        )
    params = mock_fetch.call_args.args[1]
    assert params[2:6] == ("V FOOD SAFETY", "V FOOD SAFETY", "table", "table")


def test_search_keyword_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        keyword.search_keyword("  ")


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_search_keyword_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        keyword.search_keyword("nutrition", top_k=top_k)


def test_search_keyword_rejects_empty_report_no():
    with pytest.raises(ValueError, match="report_no"):
        keyword.search_keyword("nutrition", report_no=" ")


@pytest.mark.parametrize("kwargs", [{"section_path": " "}, {"chunk_type": " "}])
def test_keyword_rejects_blank_filters(kwargs):
    with pytest.raises(ValueError, match="cannot be empty"):
        keyword.search_keyword("nutrition", **kwargs)
