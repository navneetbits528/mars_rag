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


import re


def normalize_answer(value: str) -> str:
    """Normalize answer text for deterministic lexical comparison."""
    value = value.casefold()
    value = re.sub(r"[^a-z0-9\\s]", " ", value)
    return " ".join(value.split())


def exact_match_score(prediction: str, expected: str) -> float:
    """Return 1 for normalized exact match, otherwise 0."""
    return float(normalize_answer(prediction) == normalize_answer(expected))


def token_f1_score(prediction: str, expected: str) -> float:
    """Token-overlap F1; useful alongside exact match for concise factual answers."""
    predicted_tokens = normalize_answer(prediction).split()
    expected_tokens = normalize_answer(expected).split()
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)

    predicted_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1

    overlap = sum(
        min(count, expected_counts.get(token, 0))
        for token, count in predicted_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answers(
    dataset: EvalDataset,
    answer_fn: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate answer quality using a deterministic callback.

    answer_fn receives (question, report_no) and returns a mapping with:
    answer, abstained, verified, citations, and retrieved_chunk_ids.
    Faithfulness is operationalized as the system's verified flag, which
    represents successful citation and claim-support verification.
    """
    per_example: list[dict[str, Any]] = []

    for example in dataset.examples:
        result = answer_fn(example.question, example.report_no)
        abstained = bool(result.get("abstained", False))
        verified = bool(result.get("verified", False))
        answer = str(result.get("answer") or "")
        citations = result.get("citations") or []
        retrieved_ids = {
            str(chunk_id) for chunk_id in result.get("retrieved_chunk_ids", [])
        }
        citation_ids = [
            str(citation.get("chunk_id"))
            for citation in citations
            if isinstance(citation, dict) and citation.get("chunk_id")
        ]
        citation_validity = (
            bool(citation_ids)
            and bool(retrieved_ids)
            and all(chunk_id in retrieved_ids for chunk_id in citation_ids)
        )

        row: dict[str, Any] = {
            "id": example.id,
            "expected_answerable": example.answerable,
            "predicted_abstained": abstained,
            "abstention_correct": abstained == (not example.answerable),
            "citation_valid": citation_validity,
            "faithful": verified and citation_validity,
        }
        if example.answerable:
            row["exact_match"] = exact_match_score(answer, example.expected_answer or "")
            row["token_f1"] = token_f1_score(answer, example.expected_answer or "")
        else:
            row["exact_match"] = None
            row["token_f1"] = None
        per_example.append(row)

    answerable = [row for row in per_example if row["expected_answerable"]]
    unanswerable = [row for row in per_example if not row["expected_answerable"]]

    def average(rows: Sequence[dict[str, Any]], key: str) -> float:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return mean(values) if values else 0.0

    return {
        "dataset_name": dataset.dataset_name,
        "corpus_report_no": dataset.corpus_report_no,
        "total_examples": len(per_example),
        "answerable_examples": len(answerable),
        "unanswerable_examples": len(unanswerable),
        "metrics": {
            "exact_match": average(answerable, "exact_match"),
            "token_f1": average(answerable, "token_f1"),
            "citation_validity": average(per_example, "citation_valid"),
            "faithfulness": average(per_example, "faithful"),
            "abstention_accuracy": average(per_example, "abstention_correct"),
        },
        "per_example": per_example,
    }
