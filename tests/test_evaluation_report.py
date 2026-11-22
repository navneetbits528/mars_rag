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
                }
            },
        },
        "cache_thresholds": {
            "thresholds": {
                "0.93": {
                    "metrics": {"precision": 0.9, "recall": 0.5},
                    "counts": {"true_positive": 2, "false_positive": 0},
                    "average_decision_latency_ms": 0.01,
                }
            }
        },
        "cache_benchmark": {
            "embedding_generation_ms": 120.0,
            "initial_cache_miss_ms": 20.0,
            "cache_hit_lookup": {
                "count": 30, "mean_ms": 5.0, "median_ms": 4.0,
            },
        },
    }


def test_build_report_combines_all_evaluation_sections():
    report = build_consolidated_report(**sample_inputs())
    assert report["retrieval_ablation"]["methods"]["hybrid"]["metrics"]["mrr"] == 0.7
    assert report["cache_threshold_evaluation"]["thresholds"]["0.93"]["metrics"]["precision"] == 0.9
    assert report["live_cache_benchmark"]["cache_hit_lookup"]["mean_ms"] == 5.0
    assert report["answer_quality"] is None


def test_report_includes_optional_answer_quality():
    inputs = sample_inputs()
    inputs["answer_quality"] = {"metrics": {"exact_match": 0.6, "token_f1": 0.8}}
    report = build_consolidated_report(**inputs)
    assert report["answer_quality"]["metrics"]["exact_match"] == 0.6


def test_report_rejects_missing_sections():
    inputs = sample_inputs()
    inputs["ablation"] = {"methods": {}}
    with pytest.raises(ValueError):
        build_consolidated_report(**inputs)


def test_write_report_creates_json_and_csv(tmp_path):
    report = build_consolidated_report(**sample_inputs())
    json_path = tmp_path / "nested" / "report.json"
    csv_path = tmp_path / "nested" / "report.csv"
    write_consolidated_report(
        report, json_output=json_path, csv_output=csv_path
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "recall@5" for row in rows)
    assert any(row["metric"] == "cache_hit_mean_ms" for row in rows)
    assert any(row["section"] == "cache_threshold" for row in rows)
