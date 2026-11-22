from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, *, required: bool = True) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Evaluation input not found: {source}")
        return None
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return payload


def build_consolidated_report(
    *,
    ablation: dict[str, Any],
    cache_thresholds: dict[str, Any],
    cache_benchmark: dict[str, Any],
    answer_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine existing evaluation outputs without rerunning model/API calls."""
    methods = ablation.get("methods")
    thresholds = cache_thresholds.get("thresholds")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Ablation report must contain a non-empty methods object.")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("Cache report must contain a non-empty thresholds object.")

    benchmark_metrics = cache_benchmark.get("cache_hit_lookup")
    if not isinstance(benchmark_metrics, dict):
        raise ValueError("Cache benchmark is missing cache_hit_lookup metrics.")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": ablation.get("dataset_name"),
        "corpus_report_no": ablation.get("corpus_report_no"),
        "retrieval_ablation": ablation,
        "cache_threshold_evaluation": cache_thresholds,
        "live_cache_benchmark": cache_benchmark,
        "answer_quality": answer_quality,
        "limitations": [
            "Retrieval metrics are measured on the labelled retrieval dataset and may not generalize beyond that corpus.",
            "Cache threshold metrics depend on the human-labelled question-pair dataset.",
            "Cache lookup latency excludes embedding generation; embedding time is reported separately.",
            "Answer faithfulness reuses the application's verification result and is not an independent human faithfulness judgement.",
            "Exact match and token F1 are lexical metrics and may under-credit semantically correct paraphrases.",
        ],
    }


def _csv_rows(report: dict[str, Any]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []

    for method, result in report["retrieval_ablation"]["methods"].items():
        for metric, value in result.get("metrics", {}).items():
            rows.append({"section": "retrieval_ablation", "configuration": method,
                         "metric": metric, "value": value, "unit": "score"})
        latency = result.get("latency", {})
        for metric, value in latency.items():
            rows.append({"section": "retrieval_ablation", "configuration": method,
                         "metric": metric, "value": value, "unit": "seconds"})

    for threshold, result in report["cache_threshold_evaluation"]["thresholds"].items():
        for metric, value in result.get("metrics", {}).items():
            rows.append({"section": "cache_threshold", "configuration": threshold,
                         "metric": metric, "value": value, "unit": "score"})
        for metric, value in result.get("counts", {}).items():
            rows.append({"section": "cache_threshold", "configuration": threshold,
                         "metric": metric, "value": value, "unit": "count"})
        rows.append({"section": "cache_threshold", "configuration": threshold,
                     "metric": "average_decision_latency_ms",
                     "value": result.get("average_decision_latency_ms", 0.0),
                     "unit": "milliseconds"})

    benchmark = report["live_cache_benchmark"]
    for metric in ("embedding_generation_ms", "initial_cache_miss_ms"):
        if metric in benchmark:
            rows.append({"section": "live_cache_benchmark", "configuration": "postgres",
                         "metric": metric, "value": benchmark[metric],
                         "unit": "milliseconds"})
    for metric, value in benchmark.get("cache_hit_lookup", {}).items():
        rows.append({"section": "live_cache_benchmark", "configuration": "postgres",
                     "metric": f"cache_hit_{metric}", "value": value,
                     "unit": "milliseconds" if metric.endswith("_ms") else "count"})

    answer_quality = report.get("answer_quality")
    if answer_quality:
        for metric, value in answer_quality.get("metrics", {}).items():
            rows.append({"section": "answer_quality", "configuration": "qa_pipeline",
                         "metric": metric, "value": value, "unit": "score"})

    return rows


def write_consolidated_report(
    report: dict[str, Any],
    *,
    json_output: str | Path,
    csv_output: str | Path,
) -> None:
    json_path = Path(json_output)
    csv_path = Path(csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = _csv_rows(report)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "configuration", "metric", "value", "unit"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine existing Mars RAG evaluation outputs into JSON and CSV."
    )
    parser.add_argument("--ablation", default="eval/ablation_results.json")
    parser.add_argument("--cache-thresholds", default="eval/cache_results.json")
    parser.add_argument("--cache-benchmark", default="eval/cache_benchmark_results.json")
    parser.add_argument("--answer-quality", default=None,
                        help="Optional JSON output from an answer-quality evaluation run.")
    parser.add_argument("--json-output", default="eval/final_evaluation_report.json")
    parser.add_argument("--csv-output", default="eval/final_evaluation_report.csv")
    args = parser.parse_args()

    report = build_consolidated_report(
        ablation=_load_json(args.ablation),
        cache_thresholds=_load_json(args.cache_thresholds),
        cache_benchmark=_load_json(args.cache_benchmark),
        answer_quality=_load_json(args.answer_quality, required=False)
        if args.answer_quality else None,
    )
    write_consolidated_report(
        report, json_output=args.json_output, csv_output=args.csv_output
    )
    print(f"Retrieval configurations: {len(report['retrieval_ablation']['methods'])}")
    print(f"Cache thresholds: {len(report['cache_threshold_evaluation']['thresholds'])}")
    print(f"Answer-quality metrics included: {report['answer_quality'] is not None}")
    print(f"Saved JSON report to {args.json_output}")
    print(f"Saved CSV report to {args.csv_output}")


if __name__ == "__main__":
    main()
