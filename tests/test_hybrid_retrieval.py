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
        patch.object(hybrid, "expand_parent_context", side_effect=lambda x: x),
    ):
        results = hybrid.search_hybrid("nutrition", top_k=3)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "B"
    assert results[0]["dense_rank"] == 2
    assert results[0]["keyword_rank"] == 1
    assert {r["chunk_id"] for r in results} == {"A", "B", "C"}
    assert results[0]["rrf_score"] == pytest.approx((1 / 62) + (1 / 61))


def test_hybrid_passes_filters_to_both_retrievers():
    with (
        patch.object(hybrid, "search_dense", return_value=[]) as dense_search,
        patch.object(hybrid, "search_keyword", return_value=[]) as keyword_search,
        patch.object(hybrid, "expand_parent_context", side_effect=lambda x: x),
    ):
        hybrid.search_hybrid(
            "nutrition",
            section_path="V FOOD SAFETY",
            chunk_type="table",
        )
    for mocked_search in (dense_search, keyword_search):
        assert mocked_search.call_args.kwargs["section_path"] == "V FOOD SAFETY"
        assert mocked_search.call_args.kwargs["chunk_type"] == "table"


def test_hybrid_respects_top_k():
    with (
        patch.object(hybrid, "search_dense", return_value=[
            {"chunk_id": "A", "content": "A", "similarity": 0.9},
            {"chunk_id": "B", "content": "B", "similarity": 0.8},
        ]),
        patch.object(hybrid, "search_keyword", return_value=[]),
        patch.object(hybrid, "expand_parent_context", side_effect=lambda x: x),
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
        
    
@pytest.mark.parametrize("kwargs", [{"section_path": " "}, {"chunk_type": " "}])
def test_hybrid_rejects_blank_filters(kwargs):
    with pytest.raises(ValueError, match="cannot be empty"):
        hybrid.search_hybrid("nutrition", **kwargs)
