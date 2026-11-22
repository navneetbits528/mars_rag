from __future__ import annotations

from typing import Any, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from marsrag.filter_resolution import ResolvedFilters, resolve_filters
from marsrag.generation import AnswerResponse, generate_answer
from marsrag.llm import RequestBudgetExceeded, request_api_budget
from marsrag.settings import get_settings
from marsrag.query_understanding import QueryPlan, analyze_query
from marsrag.retrieval.hybrid import search_hybrid

QUERY_RRF_K = 60
MAX_RETRIEVAL_QUERIES = 5


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
    return {"filters": filters}


def _route_after_analysis(state: QAState) -> str:
    if "response" in state:
        return "finish"
    return "resolve_filters"


def _route_after_filters(state: QAState) -> str:
    return "fast_retrieval" if state["plan"].route == "simple" else "slow_retrieval"


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


def build_qa_graph():
    """Build the conditional LangGraph QA workflow."""
    graph = StateGraph(QAState)
    graph.add_node("analyze", _analyze_node)
    graph.add_node("resolve_filters", _resolve_filters_node)
    graph.add_node("fast_retrieval", _fast_retrieval_node)
    graph.add_node("slow_retrieval", _slow_retrieval_node)
    graph.add_node("aggregate", _aggregate_node)
    graph.add_node("generate", _generate_node)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        _route_after_analysis,
        {"finish": END, "resolve_filters": "resolve_filters"},
    )
    graph.add_conditional_edges(
        "resolve_filters",
        _route_after_filters,
        {"fast_retrieval": "fast_retrieval", "slow_retrieval": "slow_retrieval"},
    )
    graph.add_edge("fast_retrieval", "aggregate")
    graph.add_edge("slow_retrieval", "aggregate")
    graph.add_edge("aggregate", "generate")
    graph.add_edge("generate", END)
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
            answer="This request reached its configured AI API-call limit. Please simplify the question or try again later.",
            abstained=True,
            verified=False,
            reason="Per-request API budget exhausted.",
        )
    return result["response"]
