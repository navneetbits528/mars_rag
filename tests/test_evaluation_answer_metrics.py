import pytest

from marsrag.evaluation.dataset import EvalDataset
from marsrag.evaluation.metrics import (
    evaluate_answers,
    exact_match_score,
    normalize_answer,
    token_f1_score,
)


def _answer_dataset():
    return EvalDataset.model_validate({
        "dataset_name": "answer-metrics-test",
        "corpus_report_no": "NASA-HRP-EVIDENCE",
        "examples": [
            {
                "id": "a1",
                "question": "What is the factual answer?",
                "report_no": "NASA-HRP-EVIDENCE",
                "category": "factual",
                "answerable": True,
                "expected_answer": "Mars missions require reliable food systems.",
                "relevant_chunk_ids": ["chunk-1"],
                "expected_pages": [4],
            },
            {
                "id": "u1",
                "question": "Does the report discuss teleportation?",
                "report_no": "NASA-HRP-EVIDENCE",
                "category": "unanswerable",
                "answerable": False,
                "expected_answer": None,
                "relevant_chunk_ids": [],
                "expected_pages": [],
            },
        ],
    })


def test_normalize_answer_removes_punctuation_and_extra_spaces():
    assert normalize_answer("Mars, missions!  require food.") == "mars missions require food"


def test_exact_match_uses_normalized_text():
    assert exact_match_score("Mars missions require food!", "Mars missions require food.") == 1.0
    assert exact_match_score("Mars missions", "Food systems") == 0.0


def test_token_f1_scores_partial_overlap():
    assert token_f1_score("Mars missions require food", "Mars missions need food") == pytest.approx(0.75)


def test_token_f1_handles_empty_values():
    assert token_f1_score("", "") == 1.0
    assert token_f1_score("some answer", "") == 0.0


def test_evaluate_answers_reports_correctness_citations_faithfulness_and_abstention():
    dataset = _answer_dataset()

    def fake_answer(question, report_no):
        if "factual answer" in question:
            return {
                "answer": "Mars missions require reliable food systems!",
                "abstained": False,
                "verified": True,
                "citations": [{"chunk_id": "chunk-1"}],
                "retrieved_chunk_ids": ["chunk-1", "chunk-2"],
            }
        return {
            "answer": "I don't have enough evidence.",
            "abstained": True,
            "verified": False,
            "citations": [],
            "retrieved_chunk_ids": [],
        }

    result = evaluate_answers(dataset, fake_answer)

    assert result["metrics"]["exact_match"] == 1.0
    assert result["metrics"]["token_f1"] == 1.0
    assert result["metrics"]["citation_validity"] == 0.5
    assert result["metrics"]["faithfulness"] == 0.5
    assert result["metrics"]["abstention_accuracy"] == 1.0
    assert result["answerable_examples"] == 1
    assert result["unanswerable_examples"] == 1


def test_evaluate_answers_marks_citation_outside_retrieved_context_invalid():
    dataset = _answer_dataset()

    def fake_answer(question, report_no):
        return {
            "answer": "An answer",
            "abstained": False,
            "verified": True,
            "citations": [{"chunk_id": "not-retrieved"}],
            "retrieved_chunk_ids": ["chunk-1"],
        }

    result = evaluate_answers(dataset, fake_answer)
    assert result["per_example"][0]["citation_valid"] is False
    assert result["per_example"][0]["faithful"] is False


def test_evaluate_answers_handles_empty_dataset():
    dataset = EvalDataset(
        dataset_name="empty",
        corpus_report_no="NASA-HRP-EVIDENCE",
        examples=[],
    )
    result = evaluate_answers(dataset, lambda question, report: {})
    assert result["total_examples"] == 0
    assert result["metrics"]["exact_match"] == 0.0
