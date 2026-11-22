import csv
import json

import pytest

from marsrag.evaluation.report import (
    build_consolidated_report,
    write_consolidated_report,
)


def sample_inputs():
    return {
        "ablation": {
            "dataset_name": "sample",
            "corpus_report_no": "NASA-HRP-EVIDENCE",
            "methods": {
                "hybrid": {
                    "metrics": {"recall@5": 0.8, "mrr": 0.7},
                    "latency": {"average_seconds_per_question": 0.2},
                },
                "hybrid_reranker": {
                    "metrics": {"recall@5": 0.9, "mrr": 0.8},
                    "latency": {"average_seconds_per_question": 2.0},
                },
            },
        },
        "cache_thresholds": {
            "dataset_name": "cache-sample",
            "pair_count": 2,
            "thresholds": {
                "0.93": {
                    "metrics": {"precision": 0.9, "recall": 0.5},
                    "counts": {"true_positive": 2, "false_positive": 0},
                    "average_decision_latency_ms": 0.01,
                }
            },
        },
        "cache_benchmark": {
            "embedding_generation_ms": 120.0,
            "initial_cache_miss_ms": 20.0,
            "cache_hit_lookup": {"count": 30, "mean_ms": 5.0, "median_ms": 4.0},
            "repeat_cache_hit_lookup": {"count": 30, "mean_ms": 5.1},
        },
        "eval_dataset": {
            "dataset_name": "sample",
            "corpus_report_no": "NASA-HRP-EVIDENCE",
            "examples": [
                {"answerable": True, "category": "factual"},
                {"answerable": False, "category": "unanswerable"},
            ],
        },
        "cache_dataset": {
            "dataset_name": "cache-sample",
            "pairs": [
                {"equivalent": True},
                {"equivalent": False},
            ],
        },
    }


def test_build_report_profiles_datasets_and_derives_findings():
    report = build_consolidated_report(**sample_inputs())
    assert report["dataset"]["retrieval"]["total_examples"] == 2
    assert report["dataset"]["retrieval"]["answerable_examples"] == 1
    assert report["dataset"]["semantic_cache"]["equivalent_pairs"] == 1
    assert report["findings"]["retrieval_best_recall_at_5"]["configuration"] == "hybrid_reranker"
    assert report["findings"]["fastest_retrieval"]["configuration"] == "hybrid"
    assert report["answer_quality"]["status"] == "not_evaluated"


def test_report_includes_optional_answer_quality():
    inputs = sample_inputs()
    inputs["answer_quality"] = {"metrics": {"exact_match": 0.6, "token_f1": 0.8}}
    report = build_consolidated_report(**inputs)
    assert report["answer_quality"]["status"] == "provided"


def test_report_rejects_dataset_mismatch():
    inputs = sample_inputs()
    inputs["cache_dataset"]["dataset_name"] = "different"
    with pytest.raises(ValueError, match="names do not match"):
        build_consolidated_report(**inputs)


def test_write_report_creates_json_and_csv(tmp_path):
    report = build_consolidated_report(**sample_inputs())
    json_path = tmp_path / "nested" / "report.json"
    csv_path = tmp_path / "nested" / "report.csv"
    write_consolidated_report(report, json_output=json_path, csv_output=csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "recall@5" for row in rows)
    assert any(row["metric"] == "cache_hit_lookup_mean_ms" for row in rows)
    assert any(row["section"] == "findings" for row in rows)
