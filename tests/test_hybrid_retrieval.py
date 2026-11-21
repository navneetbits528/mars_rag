
from unittest.mock import patch

import pytest

from marsrag.retrieval import hybrid


def test_hybrid_combines_and_ranks_results():
    dense_results = [
        {"chunk_id": "A", "content": "Dense first", "similarity": 0.9},
        {"chunk_id": "B", "content": "Dense second", "similarity": 0.8},
    ]

    keyword_results = [
        {"chunk_id": "B", "content": "Keyword first", "keyword_score": 0.5},
        {"chunk_id": "C", "content": "Keyword second", "keyword_score": 0.3},
    ]

    with (
        patch.object(hybrid, "search_dense", return_value=dense_results),
        patch.object(hybrid, "search_keyword", return_value=keyword_results),
    ):
        results = hybrid.search_hybrid("nutrition", top_k=3)

    assert len(results) == 3

    # B appears near the top in both lists, so it should rank first.
    assert results[0]["chunk_id"] == "B"
    assert results[0]["dense_rank"] == 2
    assert results[0]["keyword_rank"] == 1

    # A and C appear in only one list.
    assert {r["chunk_id"] for r in results} == {"A", "B", "C"}

    expected_score = (1 / 62) + (1 / 61)
    assert results[0]["rrf_score"] == pytest.approx(expected_score)


def test_hybrid_respects_top_k():
    with (
        patch.object(hybrid, "search_dense", return_value=[
            {"chunk_id": "A", "content": "A", "similarity": 0.9},
            {"chunk_id": "B", "content": "B", "similarity": 0.8},
        ]),
        patch.object(hybrid, "search_keyword", return_value=[]),
    ):
        results = hybrid.search_hybrid("nutrition", top_k=1)

    assert len(results) == 1


def test_hybrid_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        hybrid.search_hybrid(" ")


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_hybrid_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        hybrid.search_hybrid("nutrition", top_k=top_k)


def test_hybrid_rejects_invalid_rrf_k():
    with pytest.raises(ValueError, match="rrf_k"):
        hybrid.search_hybrid("nutrition", rrf_k=0)


def test_hybrid_rejects_empty_report_no():
    with pytest.raises(ValueError, match="report_no"):
        hybrid.search_hybrid("nutrition", report_no=" ")
