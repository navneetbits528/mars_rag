
from unittest.mock import patch

import pytest

from marsrag.retrieval.parent import expand_parent_context


def test_empty_results():
    assert expand_parent_context([]) == []


def test_invalid_max_parent_chars():
    with pytest.raises(ValueError):
        expand_parent_context([], max_parent_chars=0)


@patch("marsrag.retrieval.parent.fetch_all")
def test_parent_context_is_attached(mock_fetch):
    mock_fetch.return_value = [{
        "parent_id": "p1",
        "section_title": "Nutrition",
        "page_start": 6,
        "page_end": 10,
        "content": "Nutrition context",
    }]

    results = [{
        "chunk_id": "c1",
        "parent_id": "p1",
        "content": "Nutrition",
        "rrf_score": 0.03,
        "dense_rank": 1,
        "keyword_rank": 2,
    }]

    expanded = expand_parent_context(results)

    assert expanded[0]["parent_context"] == "Nutrition context"
    assert expanded[0]["parent_section_title"] == "Nutrition"
    assert expanded[0]["rrf_score"] == 0.03
    assert expanded[0]["dense_rank"] == 1
    assert expanded[0]["keyword_rank"] == 2


@patch("marsrag.retrieval.parent.fetch_all")
def test_duplicate_parent_ids_are_fetched_once(mock_fetch):
    mock_fetch.return_value = [{
        "parent_id": "p1",
        "section_title": "Nutrition",
        "page_start": 6,
        "page_end": 10,
        "content": "Nutrition context",
    }]

    results = [
        {"chunk_id": "c1", "parent_id": "p1", "content": "Nutrition"},
        {"chunk_id": "c2", "parent_id": "p1", "content": "Nutrition"},
    ]

    expanded = expand_parent_context(results)

    assert len(expanded) == 2
    assert mock_fetch.call_count == 1
    assert all(
        item["parent_context"] == "Nutrition context"
        for item in expanded
    )


@patch("marsrag.retrieval.parent.fetch_all")
def test_missing_parent_is_handled(mock_fetch):
    mock_fetch.return_value = []

    results = [{
        "chunk_id": "c1",
        "parent_id": "missing",
        "content": "Some content",
    }]

    expanded = expand_parent_context(results)

    assert expanded[0]["parent_context"] is None


@patch("marsrag.retrieval.parent.fetch_all")
def test_large_parent_context_is_bounded(mock_fetch):
    content = "A" * 5000

    mock_fetch.return_value = [{
        "parent_id": "p1",
        "section_title": "Large section",
        "page_start": 6,
        "page_end": 46,
        "content": content,
    }]

    results = [{
        "chunk_id": "c1",
        "parent_id": "p1",
        "content": "not found",
    }]

    expanded = expand_parent_context(
        results,
        max_parent_chars=3000,
    )

    assert len(expanded[0]["parent_context"]) == 3000
