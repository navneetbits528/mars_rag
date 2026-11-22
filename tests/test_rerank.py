from unittest.mock import patch

import pytest

from marsrag.retrieval import rerank


def _candidates():
    return [
        {"chunk_id": "A", "content": "First", "rrf_score": 0.03},
        {"chunk_id": "B", "content": "Second", "rrf_score": 0.02},
        {"chunk_id": "C", "content": "Third", "rrf_score": 0.01},
    ]


def _structured_model():
    class Model:
        def with_structured_output(self, schema):
            assert schema is rerank.RerankResponse
            return self
    return Model()


def test_rerank_results_orders_by_judge_score_and_preserves_metadata():
    response = {
        "rankings": [
            {"chunk_id": "A", "score": 2, "reason": "partial"},
            {"chunk_id": "B", "score": 5, "reason": "direct"},
            {"chunk_id": "C", "score": 3, "reason": "related"},
        ]
    }
    with (
        patch.object(rerank, "build_chat_model", return_value=_structured_model()),
        patch.object(rerank, "invoke_chat", return_value=response) as invoke,
    ):
        results = rerank.rerank_results("What is the answer?", _candidates())

    assert [item["chunk_id"] for item in results] == ["B", "C", "A"]
    assert results[0]["rrf_score"] == 0.02
    assert results[0]["rerank_rank"] == 1
    assert results[0]["rerank_reason"] == "direct"
    invoke.assert_called_once()


def test_rerank_accepts_partial_scores_and_preserves_unscored_order():
    response = {
        "rankings": [
            {"chunk_id": "C", "score": 4, "reason": "relevant"},
            {"chunk_id": "A", "score": 2, "reason": "less relevant"},
        ]
    }
    with (
        patch.object(rerank, "build_chat_model", return_value=_structured_model()),
        patch.object(rerank, "invoke_chat", return_value=response),
    ):
        results = rerank.rerank_results("question", _candidates())

    assert [item["chunk_id"] for item in results] == ["C", "A", "B"]
    assert results[0]["rerank_rank"] == 1
    assert results[1]["rerank_rank"] == 2
    assert "rerank_score" not in results[2]


def test_rerank_limits_candidates_and_keeps_unscored_tail():
    response = {
        "rankings": [
            {"chunk_id": "A", "score": 4, "reason": "ok"},
            {"chunk_id": "B", "score": 3, "reason": "ok"},
        ]
    }
    with (
        patch.object(rerank, "build_chat_model", return_value=_structured_model()),
        patch.object(rerank, "invoke_chat", return_value=response),
    ):
        results = rerank.rerank_results("question", _candidates(), max_candidates=2)

    assert [item["chunk_id"] for item in results] == ["A", "B", "C"]
    assert "rerank_score" not in results[-1]


def test_rerank_falls_back_to_original_order_for_invalid_structured_output():
    original = _candidates()
    with (
        patch.object(rerank, "build_chat_model", return_value=_structured_model()),
        patch.object(rerank, "invoke_chat", return_value={"unexpected": []}),
    ):
        results = rerank.rerank_results("question", original)

    assert results == original


def test_rerank_rejects_unknown_or_duplicate_ids():
    for rankings in (
        [{"chunk_id": "A", "score": 4}, {"chunk_id": "X", "score": 3}],
        [{"chunk_id": "A", "score": 4}, {"chunk_id": "A", "score": 3}],
    ):
        with pytest.raises(ValueError):
            rerank._validate_rankings({"rankings": rankings}, {"A", "B"})


def test_rerank_rejects_empty_rankings():
    with pytest.raises(ValueError, match="any valid"):
        rerank._validate_rankings({"rankings": []}, {"A", "B"})


def test_rerank_skips_model_for_empty_candidates():
    with patch.object(rerank, "invoke_chat") as invoke:
        assert rerank.rerank_results("question", []) == []
    invoke.assert_not_called()


def test_rerank_rejects_invalid_max_candidates():
    with pytest.raises(ValueError, match="max_candidates"):
        rerank.rerank_results("question", _candidates(), max_candidates=0)
