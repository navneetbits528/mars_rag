from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Callable

from pydantic import BaseModel, Field, model_validator

DEFAULT_THRESHOLDS = (0.90, 0.93, 0.95, 0.97)


class CachePair(BaseModel):
    """A human-labelled pair for evaluating semantic cache reuse."""

    id: str = Field(min_length=1, max_length=100)
    cached_question: str = Field(min_length=5, max_length=2000)
    candidate_question: str = Field(min_length=5, max_length=2000)
    equivalent: bool


class CacheEvalDataset(BaseModel):
    """Question-pair benchmark with human-labelled semantic equivalence."""

    schema_version: int = 1
    dataset_name: str = Field(min_length=1)
    description: str = ""
    pairs: list[CachePair] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> "CacheEvalDataset":
        ids = [pair.id for pair in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("Cache evaluation pair IDs must be unique.")
        if not any(pair.equivalent for pair in self.pairs):
            raise ValueError("Dataset must include at least one equivalent pair.")
        if not any(not pair.equivalent for pair in self.pairs):
            raise ValueError("Dataset must include at least one non-equivalent pair.")
        return self


def load_cache_dataset(path: str | Path) -> CacheEvalDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return CacheEvalDataset.model_validate(payload)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Embeddings must be non-empty and have matching dimensions.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding vectors must have non-zero magnitude.")
    return dot / (left_norm * right_norm)


def evaluate_cache_thresholds(
    dataset: CacheEvalDataset,
    similarity_fn: Callable[[str, str], float],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict:
    """Measure cache reuse classification against labelled question pairs.

    A predicted hit means the candidate question would reuse the cached answer.
    Similarity computation latency is measured separately from embedding/API time.
    """
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("Thresholds must be in (0, 1].")

    scored_pairs = []
    embedding_or_similarity_seconds = 0.0
    for pair in dataset.pairs:
        started = time.perf_counter()
        similarity = float(similarity_fn(pair.cached_question, pair.candidate_question))
        embedding_or_similarity_seconds += time.perf_counter() - started
        if not -1.000001 <= similarity <= 1.000001:
            raise ValueError("Similarity scores must be cosine values in [-1, 1].")
        scored_pairs.append({
            "id": pair.id,
            "similarity": similarity,
            "equivalent": pair.equivalent,
        })

    results = {}
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        per_pair = []
        decision_times = []
        for pair in scored_pairs:
            started = time.perf_counter()
            predicted_hit = pair["similarity"] >= threshold
            decision_times.append(time.perf_counter() - started)
            actual_hit = pair["equivalent"]
            tp += int(predicted_hit and actual_hit)
            fp += int(predicted_hit and not actual_hit)
            tn += int(not predicted_hit and not actual_hit)
            fn += int(not predicted_hit and actual_hit)
            per_pair.append({
                **pair,
                "predicted_hit": predicted_hit,
                "correct": predicted_hit == actual_hit,
            })

        total = len(scored_pairs)
        actual_positive = tp + fn
        predicted_positive = tp + fp
        results[str(threshold)] = {
            "counts": {"true_positive": tp, "false_positive": fp,
                       "true_negative": tn, "false_negative": fn},
            "metrics": {
                "hit_rate": predicted_positive / total if total else 0.0,
                "precision": tp / predicted_positive if predicted_positive else 0.0,
                "recall": tp / actual_positive if actual_positive else 0.0,
                "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                "accuracy": (tp + tn) / total if total else 0.0,
            },
            "average_decision_latency_ms": (
                mean(decision_times) * 1000 if decision_times else 0.0
            ),
            "per_pair": per_pair,
        }

    return {
        "schema_version": 1,
        "dataset_name": dataset.dataset_name,
        "pair_count": len(dataset.pairs),
        "embedding_or_similarity_seconds": round(embedding_or_similarity_seconds, 4),
        "thresholds": results,
        "limitations": [
            "This evaluates semantic-match decisions, not end-to-end PostgreSQL cache lookup latency.",
            "Pair labels must be reviewed by a person; similarity is not proof of answer equivalence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic cache similarity thresholds.")
    parser.add_argument("--input", default="eval/cache_dataset.json")
    parser.add_argument("--output", default="eval/cache_results.json")
    args = parser.parse_args()

    from marsrag.llm import embed_query

    dataset = load_cache_dataset(args.input)
    embeddings: dict[str, list[float]] = {}

    def similarity(left: str, right: str) -> float:
        for question in (left, right):
            if question not in embeddings:
                embeddings[question] = embed_query(question)
        return cosine_similarity(embeddings[left], embeddings[right])

    report = evaluate_cache_thresholds(dataset, similarity)
    report["unique_questions_embedded"] = len(embeddings)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    print(f"Dataset: {report['dataset_name']}; pairs={report['pair_count']}")
    for threshold, result in report["thresholds"].items():
        metrics = result["metrics"]
        print(
            f"threshold={threshold} hit_rate={metrics['hit_rate']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"false_positive_rate={metrics['false_positive_rate']:.3f}"
        )
    print(f"Saved cache evaluation to {destination}")


if __name__ == "__main__":
    main()
