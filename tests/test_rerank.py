from types import SimpleNamespace
from unittest.mock import patch

from marsrag.retrieval import rerank


def _candidates():
    return [
        {"chunk_id": "A", "content": "First", "rrf_score": 0.03},
        {"chunk_id": "B", "content": "Second", "rrf_score": 0.02},
        {"chunk_id": "C", "content": "Third", "rrf_score": 0.01},
    ]


def test_rerank_results_orders_by_judge_score_and_preserves_metadata():
    response = SimpleNamespace(
        content='{"rankings": ['
        '{"chunk_id": "A", "score": 2, "reason": "partial"},'
        '{"chunk_id": "B", "score": 5, "reason": "direct"},'
        '{"chunk_id": "C", "score": 3, "reason": "related"}]}'
    )
    with (
        patch.object(rerank, "build_chat_model", return_value=object()),
        patch.object(rerank, "invoke_chat", return_value=response) as invoke,
    ):
        results = rerank.rerank_results("What is the answer?", _candidates())

    assert [item["chunk_id"] for item in results] == ["B", "C", "A"]
    assert results[0]["rrf_score"] == 0.02
    assert results[0]["rerank_rank"] == 1
    assert results[0]["rerank_reason"] == "direct"
    invoke.assert_called_once()


def test_rerank_limits_candidates_and_keeps_unscored_tail():
    response = SimpleNamespace(
        content='{"rankings":[{"chunk_id":"A","score":4,"reason":"ok"},'
        '{"chunk_id":"B","score":3,"reason":"ok"}]}'
    )
    with (
        patch.object(rerank, "build_chat_model", return_value=object()),
        patch.object(rerank, "invoke_chat", return_value=response),
    ):
        results = rerank.rerank_results("question", _candidates(), max_candidates=2)

    assert [item["chunk_id"] for item in results] == ["A", "B", "C"]
    assert "rerank_score" not in results[-1]


def test_rerank_falls_back_to_original_order_for_invalid_json():
    original = _candidates()
    with (
        patch.object(rerank, "build_chat_model", return_value=object()),
        patch.object(rerank, "invoke_chat", return_value=SimpleNamespace(content="not json")),
    ):
        results = rerank.rerank_results("question", original)

    assert results == original


def test_rerank_skips_model_for_empty_candidates():
    with patch.object(rerank, "invoke_chat") as invoke:
        assert rerank.rerank_results("question", []) == []
    invoke.assert_not_called()


def test_rerank_rejects_invalid_max_candidates():
    import pytest

    with pytest.raises(ValueError, match="max_candidates"):
        rerank.rerank_results("question", _candidates(), max_candidates=0)
