from __future__ import annotations

from typing import Any, Sequence

from marsrag.filter_resolution import resolve_filters
from marsrag.generation import AnswerResponse, generate_answer
from marsrag.query_understanding import analyze_query
from marsrag.retrieval.dense import DEFAULT_REPORT_NO
from marsrag.retrieval.hybrid import search_hybrid

QUERY_RRF_K = 60
MAX_RETRIEVAL_QUERIES = 5


def _merge_query_results(
    result_sets: list[list[dict[str, Any]]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fuse ranked results from multiple query variants using reciprocal rank."""
    merged: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, source in enumerate(results, start=1):
            chunk_id = source["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**source, "query_rrf_score": 0.0}
            merged[chunk_id]["query_rrf_score"] += 1.0 / (QUERY_RRF_K + rank)

    return sorted(
        merged.values(),
        key=lambda item: item["query_rrf_score"],
        reverse=True,
    )[:top_k]


def _build_retrieval_queries(plan: Any) -> list[str]:
    """Build a bounded retrieval plan, prioritizing decomposed multi-hop questions."""
    if plan.route == "multi_hop":
        candidates = [plan.standalone_query, *plan.sub_questions, *plan.search_queries]
    else:
        candidates = [plan.standalone_query, *plan.search_queries]
    return list(dict.fromkeys(
        query.strip() for query in candidates if query and query.strip()
    ))[:MAX_RETRIEVAL_QUERIES]


def answer_question(
    question: str,
    *,
    report_no: str | None = None,
    top_k: int = 5,
    section_path: str | None = None,
    chunk_type: str | None = None,
    use_reranker: bool = False,
    model_tier: str = "main",
    conversation_history: Sequence[dict[str, str]] | None = None,
) -> AnswerResponse:
    """Analyze, validate corpus filters, retrieve, and answer."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    original_question = question.strip()
    plan = analyze_query(original_question, conversation_history=conversation_history)
    if plan.route == "clarification" and plan.clarification_question:
        return AnswerResponse(
            answer=plan.clarification_question,
            abstained=True,
            verified=False,
            citations=[],
            reason="Clarification is required before searching the report.",
        )

    filters = resolve_filters(
        original_question,
        report_no=report_no,
        section_path=section_path,
        chunk_type=chunk_type,
        inferred_chunk_type=plan.chunk_type,
    )
    queries = _build_retrieval_queries(plan)
    result_sets = [
        search_hybrid(
            query,
            top_k=top_k,
            report_no=filters.report_no,
            section_path=filters.section_path,
            chunk_type=filters.chunk_type,
            use_reranker=use_reranker,
        )
        for query in queries
    ]
    sources = _merge_query_results(result_sets, top_k=top_k)
    return generate_answer(original_question, sources, model_tier=model_tier)
