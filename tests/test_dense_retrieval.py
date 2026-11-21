
from unittest.mock import patch

import pytest

from marsrag.retrieval import dense


def test_search_dense_returns_results():
    expected = [
        {
            "chunk_id": "chunk-1",
            "report_no": "NASA-HRP-EVIDENCE",
            "page": 10,
            "content": "Nutrition requirements for astronauts.",
            "similarity": 0.92,
        }
    ]

    with (
        patch.object(dense, "embed_query", return_value=[0.1, 0.2, 0.3]),
        patch.object(dense, "fetch_all", return_value=expected) as mock_fetch,
    ):
        results = dense.search_dense("nutrition requirements", top_k=3)

    assert results == expected
    mock_fetch.assert_called_once()

    sql = mock_fetch.call_args.args[0]
    params = mock_fetch.call_args.args[1]

    assert "embedding <=> %s::vector" in sql
    assert "report_no = %s" in sql
    assert params[1] == "NASA-HRP-EVIDENCE"
    assert params[3] == 3


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
