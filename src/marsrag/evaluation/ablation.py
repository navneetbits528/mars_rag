from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from marsrag.evaluation.dataset import EvalDataset, load_dataset
from marsrag.evaluation.metrics import evaluate_retrieval
from marsrag.query_understanding import analyze_query
from marsrag.retrieval.dense import search_dense
from marsrag.retrieval.hybrid import search_hybrid
from marsrag.retrieval.keyword import search_keyword


RetrievalFn = Callable[[str, str], list[dict[str, Any]]]


def _merge_ranked_lists(
    result_sets: list[list[dict[str, Any]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge multiple ranked query result sets with reciprocal-rank fusion."""
    merged: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**item, "query_rrf_score": 0.0}
            merged[chunk_id]["query_rrf_score"] += 1.0 / (rrf_k + rank)
    return sorted(
        merged.values(),
        key=lambda item: item["query_rrf_score"],
        reverse=True,
    )[:top_k]


def build_ablation_methods(*, top_k: int = 5) -> dict[str, RetrievalFn]:
    """Create comparable retrieval configurations for the evaluation corpus."""
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    def dense(question: str, report_no: str) -> list[dict[str, Any]]:
        return search_dense(question, top_k=top_k, report_no=report_no)

    def keyword(question: str, report_no: str) -> list[dict[str, Any]]:
        return search_keyword(question, top_k=top_k, report_no=report_no)

    def hybrid(question: str, report_no: str) -> list[dict[str, Any]]:
        return search_hybrid(
            question, top_k=top_k, report_no=report_no, use_reranker=False
        )

    def hybrid_reranker(question: str, report_no: str) -> list[dict[str, Any]]:
        return search_hybrid(
            question, top_k=top_k, report_no=report_no, use_reranker=True
        )

    def hybrid_query_expansion(
        question: str, report_no: str
    ) -> list[dict[str, Any]]:
        plan = analyze_query(question)
        queries = list(dict.fromkeys(
            query.strip()
            for query in [plan.standalone_query, *plan.search_queries]
            if query and query.strip()
        ))[:4]
        result_sets = [
            search_hybrid(
                query,
                top_k=top_k,
                report_no=report_no,
                use_reranker=False,
            )
            for query in queries
        ]
        return _merge_ranked_lists(result_sets, top_k=top_k)

    def hybrid_reranker_expansion(
        question: str, report_no: str
    ) -> list[dict[str, Any]]:
        plan = analyze_query(question)
        queries = list(dict.fromkeys(
            query.strip()
            for query in [plan.standalone_query, *plan.search_queries]
            if query and query.strip()
        ))[:4]
        result_sets = [
            search_hybrid(
                query,
                top_k=top_k,
                report_no=report_no,
                use_reranker=True,
            )
            for query in queries
        ]
        return _merge_ranked_lists(result_sets, top_k=top_k)

    return {
        "dense": dense,
        "keyword": keyword,
        "hybrid": hybrid,
        "hybrid_reranker": hybrid_reranker,
        "hybrid_query_expansion": hybrid_query_expansion,
        "hybrid_reranker_query_expansion": hybrid_reranker_expansion,
    }


def run_ablation(
    dataset: EvalDataset,
    methods: dict[str, RetrievalFn] | None = None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run every retrieval configuration against the same labelled questions."""
    configured_methods = (
        build_ablation_methods(top_k=top_k) if methods is None else methods
    )
    if not configured_methods:
        raise ValueError("At least one ablation method is required.")

    results: dict[str, Any] = {}
    for name, retrieve_fn in configured_methods.items():
        started = time.perf_counter()
        evaluation = evaluate_retrieval(dataset, retrieve_fn)
        elapsed = time.perf_counter() - started
        evaluated = evaluation["evaluated_examples"]
        results[name] = {
            **evaluation,
            "latency": {
                "total_seconds": round(elapsed, 4),
                "average_seconds_per_question": round(elapsed / evaluated, 4)
                if evaluated else 0.0,
            },
        }

    return {
        "schema_version": 1,
        "dataset_name": dataset.dataset_name,
        "corpus_report_no": dataset.corpus_report_no,
        "top_k": top_k,
        "method_count": len(results),
        "methods": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Mars RAG retrieval configurations."
    )
    parser.add_argument("--input", default="eval/dataset.json")
    parser.add_argument("--output", default="eval/ablation_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    report = run_ablation(dataset, top_k=args.top_k)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: {report['dataset_name']}")
    print(f"Methods: {report['method_count']}")
    for name, result in report["methods"].items():
        metrics = result["metrics"]
        latency = result["latency"]["average_seconds_per_question"]
        print(
            f"{name:38} "
            f"Recall@5={metrics['recall@5']:.3f} "
            f"MRR={metrics['mrr']:.3f} "
            f"avg_latency={latency:.3f}s"
        )
    print(f"Saved ablation results to {destination}")


if __name__ == "__main__":
    main()
