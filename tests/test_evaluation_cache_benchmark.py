from unittest.mock import patch

import pytest

from marsrag.evaluation.cache_benchmark import run_cache_benchmark


def test_cache_benchmark_measures_hits_and_cleans_temporary_entry():
    from marsrag.generation import AnswerResponse

    answer = AnswerResponse(
        answer="Benchmark-only verified cache response.",
        abstained=False,
        verified=True,
        citations=[],
    )
    with (
        patch("marsrag.evaluation.cache_benchmark.embed_query", return_value=[0.1, 0.2]),
        patch("marsrag.evaluation.cache_benchmark.find_cached_answer", side_effect=[None, answer, answer, answer, answer]),
        patch("marsrag.evaluation.cache_benchmark.store_verified_answer", return_value=True),
        patch("marsrag.evaluation.cache_benchmark.execute", return_value=1) as execute,
    ):
        report = run_cache_benchmark(repeats=2)

    assert report["cache_hit_lookup"]["count"] == 2
    assert report["cache_hit_lookup"]["mean_ms"] >= 0
    execute.assert_called_once()
    assert execute.call_args.args[1][0].startswith("benchmark-")


def test_cache_benchmark_validates_repeat_count():
    with pytest.raises(ValueError):
        run_cache_benchmark(repeats=0)
