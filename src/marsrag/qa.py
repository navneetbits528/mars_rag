from __future__ import annotations

from typing import Any, Sequence, TypedDict
import logging

from langgraph.graph import END, START, StateGraph

from marsrag.filter_resolution import ResolvedFilters, resolve_filters
from marsrag.generation import AnswerResponse, generate_answer, stream_draft_answer
from marsrag.llm import RequestBudgetExceeded, request_api_budget
from marsrag.settings import get_settings
from marsrag.llm import embed_query
from marsrag.semantic_cache import build_cache_key, find_cached_answer, store_verified_answer
from marsrag.query_understanding import QueryPlan, analyze_query
from marsrag.retrieval.hybrid import search_hybrid

QUERY_RRF_K = 60
MAX_RETRIEVAL_QUERIES = 5
logger = logging.getLogger(__name__)


class QAState(TypedDict, total=False):
    question: str
    report_no: str | None
    top_k: int
    section_path: str | None
    chunk_type: str | None
    use_reranker: bool
    model_tier: str
    conversation_history: Sequence[dict[str, str]] | None
    plan: QueryPlan
    filters: ResolvedFilters
    result_sets: list[list[dict[str, Any]]]
    sources: list[dict[str, Any]]
    response: AnswerResponse
    cache_key: str
    cache_embedding: list[float]
    cacheable: bool


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


def _analyze_node(state: QAState) -> dict[str, Any]:
    plan = analyze_query(
        state["question"],
        conversation_history=state.get("conversation_history"),
    )
    if plan.route == "clarification" and plan.clarification_question:
        return {
            "plan": plan,
            "response": AnswerResponse(
                answer=plan.clarification_question,
                abstained=True,
                verified=False,
                citations=[],
                reason="Clarification is required before searching the report.",
            ),
        }
    return {"plan": plan}


def _resolve_filters_node(state: QAState) -> dict[str, Any]:
    plan = state["plan"]
    filters = resolve_filters(
        state["question"],
        report_no=state.get("report_no"),
        section_path=state.get("section_path"),
        chunk_type=state.get("chunk_type"),
        inferred_chunk_type=plan.chunk_type,
    )
    settings = get_settings()
    cache_key = build_cache_key(
        report_no=filters.report_no,
        section_path=filters.section_path,
        chunk_type=filters.chunk_type,
        top_k=state["top_k"],
        use_reranker=state["use_reranker"],
        model_tier=state["model_tier"],
        prompt_version=settings.prompt_version,
        corpus_version=settings.corpus_version,
        query_route=plan.route,
    )
    return {
        "filters": filters,
        "cache_key": cache_key,
        "cacheable": not bool(state.get("conversation_history")),
    }


def _route_after_analysis(state: QAState) -> str:
    if "response" in state:
        return "finish"
    return "resolve_filters"


def _cache_lookup_node(state: QAState) -> dict[str, Any]:
    settings = get_settings()
    if not settings.semantic_cache_enabled or not state.get("cacheable", False):
        return {}
    query = state["plan"].standalone_query
    embedding = embed_query(query)
    try:
        cached = find_cached_answer(
            query,
            state["cache_key"],
            similarity_threshold=settings.semantic_cache_similarity_threshold,
            query_embedding=embedding,
        )
    except RequestBudgetExceeded:
        raise
    except Exception:
        logger.exception("semantic_cache_lookup_failed; continuing without cache")
        return {}
    if cached is not None:
        logger.info("semantic_cache_hit")
        return {"response": cached, "cache_embedding": embedding}
    logger.info("semantic_cache_miss")
    return {"cache_embedding": embedding}


def _route_after_cache(state: QAState) -> str:
    if "response" in state:
        return "finish"
    return "fast_retrieval" if state["plan"].route == "simple" else "slow_retrieval"


def _store_cache_node(state: QAState) -> dict[str, Any]:
    settings = get_settings()
    response = state["response"]
    if (
        settings.semantic_cache_enabled
        and state.get("cacheable", False)
        and response.verified
        and not response.abstained
    ):
        try:
            store_verified_answer(
                state["plan"].standalone_query,
                state["cache_key"],
                response,
                ttl_seconds=settings.semantic_cache_ttl_seconds,
                query_embedding=state.get("cache_embedding"),
            )
        except RequestBudgetExceeded:
            raise
        except Exception:
            logger.exception("semantic_cache_store_failed; returning generated answer")
    return {}


def _retrieve(state: QAState, queries: list[str]) -> dict[str, Any]:
    filters = state["filters"]
    result_sets = [
        search_hybrid(
            query,
            top_k=state["top_k"],
            report_no=filters.report_no,
            section_path=filters.section_path,
            chunk_type=filters.chunk_type,
            use_reranker=state["use_reranker"],
        )
        for query in queries
    ]
    return {"result_sets": result_sets}


def _fast_retrieval_node(state: QAState) -> dict[str, Any]:
    """Low-latency path: one focused retrieval for simple questions."""
    return _retrieve(state, [state["plan"].standalone_query])


def _slow_retrieval_node(state: QAState) -> dict[str, Any]:
    """Evidence-expansion path for multi-hop and comparison questions."""
    return _retrieve(state, _build_retrieval_queries(state["plan"]))


def _aggregate_node(state: QAState) -> dict[str, Any]:
    return {
        "sources": _merge_query_results(
            state.get("result_sets", []),
            top_k=state["top_k"],
        )
    }


def _generate_node(state: QAState) -> dict[str, Any]:
    return {
        "response": generate_answer(
            state["question"],
            state.get("sources", []),
            model_tier=state["model_tier"],
        )
    }


def stream_answer_question(
    question: str,
    *,
    report_no: str | None = None,
    top_k: int = 5,
    section_path: str | None = None,
    chunk_type: str | None = None,
    use_reranker: bool = False,
    model_tier: str = "main",
    conversation_history: Sequence[dict[str, str]] | None = None,
):
    """Yield provisional text tokens followed by the final verified QA response."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    state: QAState = {
        "question": question.strip(),
        "report_no": report_no,
        "top_k": top_k,
        "section_path": section_path,
        "chunk_type": chunk_type,
        "use_reranker": use_reranker,
        "model_tier": model_tier,
        "conversation_history": conversation_history,
    }
    try:
        with request_api_budget(get_settings().max_requests_per_qa):
            yield "status", {"stage": "analyzing"}
            state.update(_analyze_node(state))
            if "response" in state:
                yield "answer", state["response"].model_dump(mode="json")
                return

            state.update(_resolve_filters_node(state))
            yield "status", {"stage": "retrieving"}
            state.update(_cache_lookup_node(state))
            if "response" in state:
                yield "answer", state["response"].model_dump(mode="json")
                return

            if state["plan"].route == "simple":
                state.update(_fast_retrieval_node(state))
            else:
                state.update(_slow_retrieval_node(state))
            state.update(_aggregate_node(state))

            if state.get("sources"):
                yield "status", {"stage": "drafting", "verified": False}
                for token in stream_draft_answer(
                    state["question"],
                    state["sources"],
                    model_tier=state["model_tier"],
                ):
                    yield "token", {"text": token, "verified": False}

            yield "status", {"stage": "verifying"}
            state.update(_generate_node(state))
            _store_cache_node(state)
            yield "answer", state["response"].model_dump(mode="json")
    except RequestBudgetExceeded:
        response = AnswerResponse(
            answer="The configured AI API budget has been reached. Please try again later.",
            abstained=True,
            verified=False,
            reason="Configured API budget exhausted.",
        )
        yield "answer", response.model_dump(mode="json")


def build_qa_graph():
    """Build the conditional LangGraph QA workflow."""
    graph = StateGraph(QAState)
    graph.add_node("analyze", _analyze_node)
    graph.add_node("resolve_filters", _resolve_filters_node)
    graph.add_node("cache_lookup", _cache_lookup_node)
    graph.add_node("fast_retrieval", _fast_retrieval_node)
    graph.add_node("slow_retrieval", _slow_retrieval_node)
    graph.add_node("aggregate", _aggregate_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("store_cache", _store_cache_node)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        _route_after_analysis,
        {"finish": END, "resolve_filters": "resolve_filters"},
    )
    graph.add_edge("resolve_filters", "cache_lookup")
    graph.add_conditional_edges(
        "cache_lookup",
        _route_after_cache,
        {
            "finish": END,
            "fast_retrieval": "fast_retrieval",
            "slow_retrieval": "slow_retrieval",
        },
    )
    graph.add_edge("fast_retrieval", "aggregate")
    graph.add_edge("slow_retrieval", "aggregate")
    graph.add_edge("aggregate", "generate")
    graph.add_edge("generate", "store_cache")
    graph.add_edge("store_cache", END)
    return graph.compile()


_qa_graph = build_qa_graph()


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
    """Run query analysis, filter validation, retrieval, and generation via LangGraph."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        with request_api_budget(get_settings().max_requests_per_qa):
            result = _qa_graph.invoke(
                {
                    "question": question.strip(),
                    "report_no": report_no,
                    "top_k": top_k,
                    "section_path": section_path,
                    "chunk_type": chunk_type,
                    "use_reranker": use_reranker,
                    "model_tier": model_tier,
                    "conversation_history": conversation_history,
                }
            )
    except RequestBudgetExceeded:
        return AnswerResponse(
            answer="The configured AI API budget has been reached. Please try again later.",
            abstained=True,
            verified=False,
            reason="Configured API budget exhausted.",
        )
    return result["response"]
