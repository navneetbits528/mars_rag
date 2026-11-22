from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from marsrag.db import execute
from marsrag.generation import AnswerResponse
from marsrag.llm import embed_query
from marsrag.semantic_cache import find_cached_answer, store_verified_answer


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min_ms": 0.0, "median_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    milliseconds = [value * 1000 for value in values]
    return {
        "count": len(milliseconds),
        "min_ms": round(min(milliseconds), 3),
        "median_ms": round(statistics.median(milliseconds), 3),
        "mean_ms": round(statistics.mean(milliseconds), 3),
        "max_ms": round(max(milliseconds), 3),
    }


def run_cache_benchmark(*, repeats: int = 20, threshold: float = 0.93) -> dict:
    """Benchmark real PostgreSQL cache miss/hit lookups using a disposable cache key."""
    if repeats < 1 or repeats > 500:
        raise ValueError("repeats must be between 1 and 500.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")

    question = "Benchmark-only question: what evidence does this report identify?"
    cache_key = f"benchmark-{uuid.uuid4().hex}"
    embedding_started = time.perf_counter()
    embedding = embed_query(question)
    embedding_seconds = time.perf_counter() - embedding_started

    response = AnswerResponse(
        answer="Benchmark-only verified cache response.",
        abstained=False,
        verified=True,
        citations=[],
    )

    miss_times: list[float] = []
    hit_times: list[float] = []
    cleanup_succeeded = False

    try:
        started = time.perf_counter()
        initial = find_cached_answer(
            question, cache_key, similarity_threshold=threshold,
            query_embedding=embedding,
        )
        initial_miss_seconds = time.perf_counter() - started
        if initial is not None:
            raise RuntimeError("Unexpected cache entry found for a new benchmark key.")

        stored = store_verified_answer(
            question, cache_key, response, ttl_seconds=300,
            query_embedding=embedding,
        )
        if not stored:
            raise RuntimeError("Could not seed the temporary benchmark cache entry.")

        for _ in range(repeats):
            started = time.perf_counter()
            result = find_cached_answer(
                question, cache_key, similarity_threshold=threshold,
                query_embedding=embedding,
            )
            hit_times.append(time.perf_counter() - started)
            if result is None or result.answer != response.answer:
                raise RuntimeError("Temporary cache entry was not returned as a hit.")

        for _ in range(repeats):
            started = time.perf_counter()
            result = find_cached_answer(
                question, cache_key, similarity_threshold=threshold,
                query_embedding=embedding,
            )
            miss_times.append(time.perf_counter() - started)
            if result is None:
                raise RuntimeError("Repeated cache lookup unexpectedly missed.")

        return {
            "schema_version": 1,
            "benchmark": "postgres_semantic_cache_lookup",
            "repeats": repeats,
            "threshold": threshold,
            "embedding_generation_ms": round(embedding_seconds * 1000, 3),
            "initial_cache_miss_ms": round(initial_miss_seconds * 1000, 3),
            "cache_hit_lookup": _summary(hit_times),
            "repeat_cache_lookup": _summary(miss_times),
            "notes": [
                "Lookup timings use a precomputed embedding and therefore exclude embedding API latency.",
                "The initial miss and subsequent hits use a unique temporary cache key.",
                "A repeated lookup is still a database hit; the report does not simulate a miss after deleting the entry.",
                "This benchmark does not measure end-to-end QA latency or generation cost.",
            ],
        }
    finally:
        deleted = execute(
            "DELETE FROM semantic_cache WHERE cache_key = %s",
            (cache_key,),
        )
        cleanup_succeeded = deleted >= 0
        if not cleanup_succeeded:
            raise RuntimeError("Could not confirm cleanup of temporary benchmark cache entry.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PostgreSQL semantic cache lookup latency.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.93)
    parser.add_argument("--output", default="eval/cache_benchmark_results.json")
    args = parser.parse_args()

    report = run_cache_benchmark(repeats=args.repeats, threshold=args.threshold)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")

    print(f"Embedding generation: {report['embedding_generation_ms']:.3f} ms")
    print(f"Initial cache miss: {report['initial_cache_miss_ms']:.3f} ms")
    print(f"Cache hit mean: {report['cache_hit_lookup']['mean_ms']:.3f} ms")
    print(f"Cache hit median: {report['cache_hit_lookup']['median_ms']:.3f} ms")
    print(f"Saved benchmark to {destination}")


if __name__ == "__main__":
    main()
