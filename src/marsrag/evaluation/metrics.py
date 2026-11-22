from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from statistics import mean
from typing import Any

from marsrag.evaluation.dataset import EvalDataset, EvalExample

DEFAULT_K_VALUES = (1, 3, 5, 10)


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of known relevant chunks found in the first k results."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return 0.0
    retrieved = set(retrieved_chunk_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Iterable[str],
) -> float:
    """Reciprocal rank of the first relevant result; zero when none is found."""
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rank_values: Iterable[float]) -> float:
    """Mean reciprocal rank, returning zero for an empty collection."""
    values = list(rank_values)
    return mean(values) if values else 0.0


def evaluate_retrieval(
    dataset: EvalDataset,
    retrieve_fn: Callable[[str, str], Sequence[dict[str, Any]]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate retrieval against human-labelled relevant chunk IDs.

    retrieve_fn receives (question, report_no) and must return ranked result
    dictionaries containing a chunk_id field. Unanswerable examples are
    excluded because they have no relevant evidence labels.
    """
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers.")

    answerable_examples = [example for example in dataset.examples if example.answerable]
    per_example: list[dict[str, Any]] = []

    for example in answerable_examples:
        results = retrieve_fn(example.question, example.report_no)
        retrieved_ids = [str(item["chunk_id"]) for item in results if item.get("chunk_id")]
        row: dict[str, Any] = {
            "id": example.id,
            "retrieved_chunk_ids": retrieved_ids,
            "reciprocal_rank": reciprocal_rank(retrieved_ids, example.relevant_chunk_ids),
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(
                retrieved_ids, example.relevant_chunk_ids, k
            )
        per_example.append(row)

    metrics: dict[str, float] = {
        f"recall@{k}": mean(row[f"recall@{k}"] for row in per_example)
        if per_example else 0.0
        for k in k_values
    }
    metrics["mrr"] = mean_reciprocal_rank(
        row["reciprocal_rank"] for row in per_example
    )

    return {
        "dataset_name": dataset.dataset_name,
        "corpus_report_no": dataset.corpus_report_no,
        "total_examples": len(dataset.examples),
        "answerable_examples": len(answerable_examples),
        "evaluated_examples": len(per_example),
        "metrics": metrics,
        "per_example": per_example,
    }
