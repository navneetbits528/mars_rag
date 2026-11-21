
from __future__ import annotations

from typing import Any

from marsrag.retrieval.dense import DEFAULT_REPORT_NO, search_dense
from marsrag.retrieval.keyword import search_keyword
from marsrag.retrieval.parent import expand_parent_context

DEFAULT_RRF_K = 60
MAX_TOP_K = 50


def search_hybrid(
    query: str,
    top_k: int = 5,
    report_no: str = DEFAULT_REPORT_NO,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Combine dense and keyword retrieval using Reciprocal Rank Fusion."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    if not report_no or not report_no.strip():
        raise ValueError("report_no cannot be empty.")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    candidate_k = min(max(top_k * 3, 10), MAX_TOP_K)

    dense_results = search_dense(
        query=query,
        top_k=candidate_k,
        report_no=report_no,
    )

    keyword_results = search_keyword(
        query=query,
        top_k=candidate_k,
        report_no=report_no,
    )

    fused: dict[str, dict[str, Any]] = {}

    def add_results(
        results: list[dict[str, Any]],
        source: str,
    ) -> None:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "dense_rank": None,
                    "keyword_rank": None,
                    "dense_score": None,
                    "keyword_score": None,
                    "rrf_score": 0.0,
                }

            fused[chunk_id][f"{source}_rank"] = rank
            fused[chunk_id][f"{source}_score"] = (
                result.get("similarity")
                if source == "dense"
                else result.get("keyword_score")
            )
            fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    add_results(dense_results, "dense")
    add_results(keyword_results, "keyword")

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]

    return expand_parent_context(ranked_results)
