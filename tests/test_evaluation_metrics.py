import pytest

from marsrag.evaluation.dataset import EvalDataset
from marsrag.evaluation.metrics import (
    evaluate_retrieval,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_uses_fraction_of_relevant_chunks():
    assert recall_at_k(["a", "b", "c"], ["b", "d"], 2) == 0.5


def test_recall_at_k_returns_zero_when_no_relevant_chunks_found():
    assert recall_at_k(["a", "b"], ["x"], 2) == 0.0


def test_recall_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="at least 1"):
        recall_at_k(["a"], ["a"], 0)


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5


def test_reciprocal_rank_returns_zero_when_no_match():
    assert reciprocal_rank(["a", "b"], ["x"]) == 0.0


def test_mean_reciprocal_rank_handles_empty_input():
    assert mean_reciprocal_rank([]) == 0.0
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == 0.5


def test_evaluate_retrieval_aggregates_answerable_examples_only():
    dataset = EvalDataset.model_validate({
        "dataset_name": "test",
        "corpus_report_no": "NASA-HRP-EVIDENCE",
        "examples": [
            {
                "id": "q1",
                "question": "What is the first evidence question?",
                "report_no": "NASA-HRP-EVIDENCE",
                "category": "factual",
                "answerable": True,
                "expected_answer": "Answer one",
                "relevant_chunk_ids": ["a", "b"],
                "expected_pages": [2],
            },
            {
                "id": "q2",
                "question": "What is the second evidence question?",
                "report_no": "NASA-HRP-EVIDENCE",
                "category": "factual",
                "answerable": True,
                "expected_answer": "Answer two",
                "relevant_chunk_ids": ["c"],
                "expected_pages": [3],
            },
            {
                "id": "q3",
                "question": "Is an unsupported item discussed?",
                "report_no": "NASA-HRP-EVIDENCE",
                "category": "unanswerable",
                "answerable": False,
                "expected_answer": None,
                "relevant_chunk_ids": [],
                "expected_pages": [],
            },
        ],
    })
    calls = []

    def fake_retrieve(question, report_no):
        calls.append((question, report_no))
        if question.startswith("What is the first"):
            return [{"chunk_id": "x"}, {"chunk_id": "a"}]
        return [{"chunk_id": "c"}]

    result = evaluate_retrieval(dataset, fake_retrieve, k_values=(1, 2))

    assert len(calls) == 2
    assert result["total_examples"] == 3
    assert result["answerable_examples"] == 2
    assert result["evaluated_examples"] == 2
    assert result["metrics"]["recall@1"] == 0.5
    assert result["metrics"]["recall@2"] == 1.0
    assert result["metrics"]["mrr"] == 0.75
    assert [row["id"] for row in result["per_example"]] == ["q1", "q2"]


def test_evaluate_retrieval_rejects_invalid_k_values():
    dataset = EvalDataset(
        dataset_name="empty",
        corpus_report_no="NASA-HRP-EVIDENCE",
        examples=[],
    )
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_retrieval(dataset, lambda question, report: [], k_values=(0,))
