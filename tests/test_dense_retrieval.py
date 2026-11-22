from unittest.mock import patch

import pytest

from marsrag.retrieval import dense


def test_search_dense_returns_results():
    expected = [{"chunk_id": "chunk-1", "report_no": "NASA-HRP-EVIDENCE"}]
    with (
        patch.object(dense, "embed_query", return_value=[0.1, 0.2, 0.3]),
        patch.object(dense, "fetch_all", return_value=expected) as mock_fetch,
    ):
        results = dense.search_dense("nutrition requirements", top_k=3)

    assert results == expected
    sql, params = mock_fetch.call_args.args
    assert "embedding <=> %s::vector" in sql
    assert "report_no = %s" in sql
    assert "status = 'current'" in sql
    assert params[1] == "NASA-HRP-EVIDENCE"
    assert params[-1] == 3


def test_dense_passes_optional_metadata_filters():
    with (
        patch.object(dense, "embed_query", return_value=[0.1]),
        patch.object(dense, "fetch_all", return_value=[]) as mock_fetch,
    ):
        dense.search_dense(
            "nutrition", section_path="V FOOD SAFETY", chunk_type="table"
        )
    params = mock_fetch.call_args.args[1]
    assert params[2:6] == ("V FOOD SAFETY", "V FOOD SAFETY", "table", "table")


def test_search_dense_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        dense.search_dense("  ")


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_search_dense_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        dense.search_dense("nutrition", top_k=top_k)


def test_search_dense_rejects_empty_report_no():
    with pytest.raises(ValueError, match="report_no"):
        dense.search_dense("nutrition", report_no=" ")


@pytest.mark.parametrize("kwargs", [{"section_path": " "}, {"chunk_type": " "}])
def test_dense_rejects_blank_filters(kwargs):
    with pytest.raises(ValueError, match="cannot be empty"):
        dense.search_dense("nutrition", **kwargs)
