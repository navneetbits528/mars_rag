
from unittest.mock import patch

import pytest

from marsrag.retrieval import keyword


def test_search_keyword_returns_results():
    expected = [
        {
            "chunk_id": "chunk-1",
            "report_no": "NASA-HRP-EVIDENCE",
            "page": 16,
            "content": "Nutritional risks during spaceflight.",
            "keyword_score": 0.25,
        }
    ]

    with patch.object(
        keyword, "fetch_all", return_value=expected
    ) as mock_fetch:
        results = keyword.search_keyword(
            "nutritional risks spaceflight",
            top_k=3,
        )

    assert results == expected
    mock_fetch.assert_called_once()

    sql = mock_fetch.call_args.args[0]
    params = mock_fetch.call_args.args[1]

    assert "content_tsv" in sql
    assert "websearch_to_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert "report_no = %s" in sql
    assert params[1] == "NASA-HRP-EVIDENCE"
    assert params[3] == 3


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
