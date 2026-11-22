import pytest

from marsrag.evaluation.dataset import EvalDataset
from marsrag.evaluation.ablation import _merge_ranked_lists, run_ablation


def make_dataset():
    return EvalDataset.model_validate({
        "dataset_name": "ablation-test",
        "corpus_report_no": "NASA-HRP-EVIDENCE",
        "examples": [
            {
                "id": "q1",
                "question": "What is the first evidence question?",
                "report_no": "NASA-HRP-EVIDENCE",
                "category": "factual",
                "answerable": True,
                "expected_answer": "Answer one",
                "relevant_chunk_ids": ["a"],
                "expected_pages": [2],
            },
            {
                "id": "q2",
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


def test_merge_ranked_lists_uses_rrf_and_deduplicates_ids():
    merged = _merge_ranked_lists(
        [
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
            [{"chunk_id": "b"}, {"chunk_id": "c"}],
        ],
        top_k=3,
    )
    assert [item["chunk_id"] for item in merged] == ["b", "a", "c"]
    assert len({item["chunk_id"] for item in merged}) == 3


def test_run_ablation_evaluates_each_method_on_same_dataset():
    calls = {"first": 0, "second": 0}

    def first(question, report_no):
        calls["first"] += 1
        return [{"chunk_id": "a"}]

    def second(question, report_no):
        calls["second"] += 1
        return [{"chunk_id": "x"}]

    result = run_ablation(
        make_dataset(),
        {"first": first, "second": second},
        top_k=5,
    )

    assert result["method_count"] == 2
    assert calls == {"first": 1, "second": 1}
    assert result["methods"]["first"]["metrics"]["recall@1"] == 1.0
    assert result["methods"]["second"]["metrics"]["recall@1"] == 0.0
    assert result["methods"]["first"]["latency"]["total_seconds"] >= 0


def test_run_ablation_requires_at_least_one_method():
    with pytest.raises(ValueError, match="At least one"):
        run_ablation(make_dataset(), methods={})


def test_merge_ranked_lists_respects_top_k():
    merged = _merge_ranked_lists(
        [[{"chunk_id": "a"}, {"chunk_id": "b"}]],
        top_k=1,
    )
    assert [item["chunk_id"] for item in merged] == ["a"]
