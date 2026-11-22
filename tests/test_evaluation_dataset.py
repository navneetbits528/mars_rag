import json

import pytest
from pydantic import ValidationError

from marsrag.evaluation.dataset import EvalDataset, EvalExample, load_dataset


def _example(**overrides):
    values = {
        "id": "hrp-001",
        "question": "What evidence is reported about the topic?",
        "report_no": "NASA-HRP-EVIDENCE",
        "category": "factual",
        "answerable": True,
        "expected_answer": "A source-grounded reference answer.",
        "relevant_chunk_ids": ["chunk-001"],
        "expected_pages": [4],
    }
    values.update(overrides)
    return values


def test_valid_answerable_example():
    example = EvalExample.model_validate(_example())
    assert example.relevant_chunk_ids == ["chunk-001"]


def test_valid_unanswerable_example():
    example = EvalExample.model_validate({
        **_example(),
        "id": "hrp-002",
        "category": "unanswerable",
        "answerable": False,
        "expected_answer": None,
        "relevant_chunk_ids": [],
        "expected_pages": [],
    })
    assert example.answerable is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_answer": None},
        {"relevant_chunk_ids": []},
        {"category": "unanswerable"},
        {"answerable": False},
    ],
)
def test_invalid_answerable_ground_truth(overrides):
    with pytest.raises(ValidationError):
        EvalExample.model_validate(_example(**overrides))


def test_unanswerable_cannot_have_reference_answer_or_evidence():
    with pytest.raises(ValidationError):
        EvalExample.model_validate({
            **_example(),
            "category": "unanswerable",
            "answerable": False,
        })


def test_dataset_rejects_duplicate_example_ids():
    example = _example()
    with pytest.raises(ValidationError, match="unique"):
        EvalDataset(
            dataset_name="test",
            corpus_report_no="NASA-HRP-EVIDENCE",
            examples=[example, example],
        )


def test_dataset_rejects_examples_from_another_report():
    with pytest.raises(ValidationError, match="mismatched"):
        EvalDataset(
            dataset_name="test",
            corpus_report_no="NASA-HRP-EVIDENCE",
            examples=[_example(report_no="NASA-M2M-ARCH")],
        )


def test_dataset_template_loads_with_no_examples(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_name": "test",
        "corpus_report_no": "NASA-HRP-EVIDENCE",
        "examples": [],
    }), encoding="utf-8")
    dataset = load_dataset(path)
    assert dataset.examples == []


def test_chunk_candidate_export_uses_current_report_only(monkeypatch, tmp_path):
    from marsrag.evaluation import dataset as module

    captured = {}

    def fake_fetch_all(query, params):
        captured["query"] = query
        captured["params"] = params
        return [{
            "chunk_id": "chunk-001",
            "report_no": "NASA-HRP-EVIDENCE",
            "page": 4,
            "title": "HRP Evidence",
            "section_title": "Overview",
            "section_path": "1",
            "chunk_type": "text",
            "content": "Evidence text",
        }]

    monkeypatch.setattr(module, "fetch_all", fake_fetch_all)
    output = tmp_path / "candidates.jsonl"
    count = module.export_chunk_candidates(
        report_no="NASA-HRP-EVIDENCE",
        output_path=output,
    )

    assert count == 1
    assert captured["params"] == ("NASA-HRP-EVIDENCE", 1000)
    assert "status = 'current'" in captured["query"]
    assert json.loads(output.read_text(encoding="utf-8"))["chunk_id"] == "chunk-001"
