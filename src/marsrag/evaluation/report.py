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


def _dataset_profile(dataset: dict[str, Any] | None, *, kind: str) -> dict[str, Any] | None:
    if dataset is None:
        return None
    if kind == "retrieval":
        examples = dataset.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("Evaluation dataset must contain a non-empty examples list.")
        return {
            "dataset_name": dataset.get("dataset_name"),
            "corpus_report_no": dataset.get("corpus_report_no"),
            "total_examples": len(examples),
            "answerable_examples": sum(bool(item.get("answerable")) for item in examples),
            "unanswerable_examples": sum(not bool(item.get("answerable")) for item in examples),
            "categories": _count_values(examples, "category"),
        }
    pairs = dataset.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Cache dataset must contain a non-empty pairs list.")
    return {
        "dataset_name": dataset.get("dataset_name"),
        "total_pairs": len(pairs),
        "equivalent_pairs": sum(bool(item.get("equivalent")) for item in pairs),
        "non_equivalent_pairs": sum(not bool(item.get("equivalent")) for item in pairs),
    }


def _count_values(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_findings(ablation: dict[str, Any], cache: dict[str, Any],
                    benchmark: dict[str, Any]) -> dict[str, Any]:
    methods = ablation["methods"]
    best_recall = max(methods, key=lambda name: methods[name]["metrics"].get("recall@5", 0.0))
    best_mrr = max(methods, key=lambda name: methods[name]["metrics"].get("mrr", 0.0))
    fastest = min(
        methods,
        key=lambda name: methods[name].get("latency", {}).get(
            "average_seconds_per_question", float("inf")
        ),
    )
    threshold_results = cache["thresholds"]
    # Prefer the highest observed precision among thresholds with at least one predicted hit.
    usable_thresholds = [
        (float(threshold), result)
        for threshold, result in threshold_results.items()
        if result.get("counts", {}).get("true_positive", 0)
         + result.get("counts", {}).get("false_positive", 0) > 0
    ]
    highest_precision_threshold = (
        max(usable_thresholds, key=lambda item: item[1]["metrics"].get("precision", 0.0))[0]
        if usable_thresholds else None
    )
    return {
        "retrieval_best_recall_at_5": {
            "configuration": best_recall,
            "value": methods[best_recall]["metrics"].get("recall@5"),
            "average_latency_seconds_per_question": methods[best_recall].get("latency", {}).get("average_seconds_per_question"),
        },
        "retrieval_best_mrr": {
            "configuration": best_mrr,
            "value": methods[best_mrr]["metrics"].get("mrr"),
            "average_latency_seconds_per_question": methods[best_mrr].get("latency", {}).get("average_seconds_per_question"),
        },
        "fastest_retrieval": {
            "configuration": fastest,
            "average_latency_seconds_per_question": methods[fastest].get("latency", {}).get("average_seconds_per_question"),
            "recall_at_5": methods[fastest]["metrics"].get("recall@5"),
        },
        "cache_threshold_with_highest_observed_precision": highest_precision_threshold,
        "live_cache": {
            "embedding_generation_ms": benchmark.get("embedding_generation_ms"),
            "initial_cache_miss_ms": benchmark.get("initial_cache_miss_ms"),
            "cache_hit_mean_ms": benchmark.get("cache_hit_lookup", {}).get("mean_ms"),
            "cache_hit_median_ms": benchmark.get("cache_hit_lookup", {}).get("median_ms"),
            "repeated_hit_mean_ms": benchmark.get("repeat_cache_hit_lookup", {}).get("mean_ms"),
        },
    }


def build_consolidated_report(
    *,
    ablation: dict[str, Any],
    cache_thresholds: dict[str, Any],
    cache_benchmark: dict[str, Any],
    eval_dataset: dict[str, Any] | None = None,
    cache_dataset: dict[str, Any] | None = None,
    answer_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the actual evaluation outputs without rerunning model/API calls."""
    methods = ablation.get("methods")
    thresholds = cache_thresholds.get("thresholds")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Ablation report must contain a non-empty methods object.")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("Cache report must contain a non-empty thresholds object.")
    if not isinstance(cache_benchmark.get("cache_hit_lookup"), dict):
        raise ValueError("Cache benchmark is missing cache_hit_lookup metrics.")

    if eval_dataset and ablation.get("dataset_name") != eval_dataset.get("dataset_name"):
        raise ValueError("Ablation report and retrieval dataset names do not match.")
    if eval_dataset and ablation.get("corpus_report_no") != eval_dataset.get("corpus_report_no"):
        raise ValueError("Ablation report and retrieval dataset corpus IDs do not match.")
    if cache_dataset and cache_thresholds.get("dataset_name") != cache_dataset.get("dataset_name"):
        raise ValueError("Cache threshold report and cache dataset names do not match.")
    if cache_dataset and cache_thresholds.get("pair_count") != len(cache_dataset.get("pairs", [])):
        raise ValueError("Cache threshold report pair count does not match the cache dataset.")

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "retrieval": _dataset_profile(eval_dataset, kind="retrieval"),
            "semantic_cache": _dataset_profile(cache_dataset, kind="cache"),
        },
        "retrieval_ablation": ablation,
        "cache_threshold_evaluation": cache_thresholds,
        "live_cache_benchmark": cache_benchmark,
        "answer_quality": {
            "status": "not_evaluated",
            "reason": (
                "No answer-quality result JSON was supplied. The existing dataset and "
                "retrieval results do not contain generated QA answers, so answer quality "
                "cannot be calculated from them."
            ),
        } if answer_quality is None else {
            "status": "provided",
            "results": answer_quality,
        },
        "findings": _build_findings(ablation, cache_thresholds, cache_benchmark),
        "limitations": [
            "Retrieval results cover 23 answerable questions in one NASA-HRP-EVIDENCE corpus; generalization is not established.",
            "The semantic-cache threshold evaluation uses 30 labelled pairs (9 equivalent and 21 non-equivalent); its false-positive estimates are uncertain at this sample size.",
            "The cache benchmark reports database lookup with a precomputed embedding. Embedding generation is separate; neither number represents full end-to-end QA latency.",
            "The benchmark's repeat_cache_hit_lookup is a second set of successful hits, not a cache-miss measurement.",
            "Answer quality was not evaluated because no generated-answer evaluation output was supplied.",
        ],
    }


def _csv_rows(report: dict[str, Any]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []

    def add(section: str, configuration: str, metric: str, value: Any, unit: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({"section": section, "configuration": configuration,
                         "metric": metric, "value": value, "unit": unit})

    for method, result in report["retrieval_ablation"]["methods"].items():
        for metric, value in result.get("metrics", {}).items():
            add("retrieval_ablation", method, metric, value, "score")
        for metric, value in result.get("latency", {}).items():
            add("retrieval_ablation", method, metric, value, "seconds")

    for threshold, result in report["cache_threshold_evaluation"]["thresholds"].items():
        for metric, value in result.get("metrics", {}).items():
            add("cache_threshold", threshold, metric, value, "score")
        for metric, value in result.get("counts", {}).items():
            add("cache_threshold", threshold, metric, value, "count")
        add("cache_threshold", threshold, "average_decision_latency_ms",
            result.get("average_decision_latency_ms"), "milliseconds")

    benchmark = report["live_cache_benchmark"]
    for metric in ("embedding_generation_ms", "initial_cache_miss_ms"):
        add("live_cache_benchmark", "postgres", metric, benchmark.get(metric), "milliseconds")
    for result_name in ("cache_hit_lookup", "repeat_cache_hit_lookup"):
        for metric, value in benchmark.get(result_name, {}).items():
            add("live_cache_benchmark", "postgres", f"{result_name}_{metric}", value,
                "milliseconds" if metric.endswith("_ms") else "count")

    findings = report.get("findings", {})
    for finding, details in findings.items():
        if not isinstance(details, dict):
            add("findings", "summary", finding, details, "value")
            continue
        for metric, value in details.items():
            add("findings", str(details.get("configuration", "summary")),
                f"{finding}_{metric}", value,
                "seconds" if "seconds" in metric else "score" if metric in {"value", "recall_at_5"} else "milliseconds")
    return rows


def write_consolidated_report(
    report: dict[str, Any], *, json_output: str | Path, csv_output: str | Path
) -> None:
    json_path, csv_path = Path(json_output), Path(csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "section", "configuration", "metric", "value", "unit"
        ])
        writer.writeheader()
        writer.writerows(_csv_rows(report))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a consolidated Mars RAG evaluation report from saved JSON results."
    )
    parser.add_argument("--ablation", default="eval/ablation_results.json")
    parser.add_argument("--cache-thresholds", default="eval/cache_results.json")
    parser.add_argument("--cache-benchmark", default="eval/cache_benchmark_results.json")
    parser.add_argument("--eval-dataset", default="eval/dataset.json")
    parser.add_argument("--cache-dataset", default="eval/cache_dataset.json")
    parser.add_argument("--answer-quality", default=None)
    parser.add_argument("--json-output", default="eval/final_evaluation_report.json")
    parser.add_argument("--csv-output", default="eval/final_evaluation_report.csv")
    args = parser.parse_args()

    report = build_consolidated_report(
        ablation=_load_json(args.ablation),
        cache_thresholds=_load_json(args.cache_thresholds),
        cache_benchmark=_load_json(args.cache_benchmark),
        eval_dataset=_load_json(args.eval_dataset),
        cache_dataset=_load_json(args.cache_dataset),
        answer_quality=_load_json(args.answer_quality, required=False) if args.answer_quality else None,
    )
    write_consolidated_report(report, json_output=args.json_output, csv_output=args.csv_output)
    print(f"Retrieval questions: {report['dataset']['retrieval']['total_examples']}")
    print(f"Cache question pairs: {report['dataset']['semantic_cache']['total_pairs']}")
    print(f"Best Recall@5: {report['findings']['retrieval_best_recall_at_5']['configuration']}")
    print(f"Best MRR: {report['findings']['retrieval_best_mrr']['configuration']}")
    print(f"Answer quality: {report['answer_quality']['status']}")
    print(f"Saved JSON report to {args.json_output}")
    print(f"Saved CSV report to {args.csv_output}")


if __name__ == "__main__":
    main()
