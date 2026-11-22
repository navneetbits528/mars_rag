from unittest.mock import patch

import pytest

from marsrag.generation import AnswerResponse
from marsrag.semantic_cache import (
    build_cache_key,
    delete_expired_cache_entries,
    find_cached_answer,
    store_verified_answer,
)


def _response(**overrides):
    values = {
        "answer": "The report identifies the relevant evidence.",
        "abstained": False,
        "verified": True,
        "citations": [],
    }
    values.update(overrides)
    return AnswerResponse(**values)


def test_cache_key_is_deterministic_and_scope_sensitive():
    base = {
        "report_no": "NASA-HRP-EVIDENCE",
        "section_path": None,
        "chunk_type": None,
        "top_k": 5,
        "use_reranker": False,
        "model_tier": "main",
        "prompt_version": "v1",
        "corpus_version": 1,
    }
    first = build_cache_key(**base)
    second = build_cache_key(**base)
    changed = build_cache_key(**{**base, "report_no": "NASA-M2M-ARCH"})

    assert first == second
    assert first != changed


def test_find_cached_answer_returns_verified_similar_result():
    cached = _response()
    with (
        patch("marsrag.semantic_cache.embed_query", return_value=[0.1, 0.2]),
        patch(
            "marsrag.semantic_cache.fetch_one",
            return_value={"answer": cached.model_dump(mode="json"), "similarity": 0.97},
        ) as fetch,
    ):
        result = find_cached_answer("What evidence is identified?", "scope-key")

    assert result == cached
    assert fetch.called


def test_find_cached_answer_rejects_low_similarity():
    with (
        patch("marsrag.semantic_cache.embed_query", return_value=[0.1, 0.2]),
        patch(
            "marsrag.semantic_cache.fetch_one",
            return_value={"answer": _response().model_dump(mode="json"), "similarity": 0.80},
        ),
    ):
        result = find_cached_answer("A different question", "scope-key")

    assert result is None


@pytest.mark.parametrize(
    "response",
    [
        _response(abstained=True, verified=False),
        _response(abstained=False, verified=False),
    ],
)
def test_store_refuses_unverified_or_abstained_answers(response):
    with patch("marsrag.semantic_cache.embed_query") as embed:
        stored = store_verified_answer("Question", "scope-key", response, ttl_seconds=60)

    assert stored is False
    embed.assert_not_called()


def test_store_persists_verified_answer():
    response = _response()
    with (
        patch("marsrag.semantic_cache.embed_query", return_value=[0.1, 0.2]),
        patch("marsrag.semantic_cache.execute", return_value=1) as execute,
    ):
        stored = store_verified_answer("Question", "scope-key", response, ttl_seconds=60)

    assert stored is True
    assert execute.called
    assert execute.call_args.args[1][1] == "Question"


def test_cache_rejects_invalid_threshold_and_ttl():
    with pytest.raises(ValueError):
        find_cached_answer("Question", "scope-key", similarity_threshold=0)
    with pytest.raises(ValueError):
        store_verified_answer("Question", "scope-key", _response(), ttl_seconds=0)


def test_expired_cache_cleanup_is_bounded():
    with patch("marsrag.semantic_cache.execute", return_value=3) as execute:
        deleted = delete_expired_cache_entries(limit=50)

    assert deleted == 3
    assert execute.call_args.args[1] == (50,)
