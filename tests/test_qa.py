import pytest
from unittest.mock import patch
from types import SimpleNamespace

@pytest.fixture(autouse=True)
def mock_filter_resolution(monkeypatch):
    def fake_resolve(question, *, report_no=None, section_path=None,
                     chunk_type=None, inferred_chunk_type=None):
        return SimpleNamespace(
            report_no=report_no or "NASA-HRP-EVIDENCE",
            section_path=section_path,
            chunk_type=chunk_type if chunk_type is not None else inferred_chunk_type,
        )
    monkeypatch.setattr("marsrag.qa.resolve_filters", fake_resolve)


from marsrag.generation import AnswerResponse
from marsrag.qa import (
    _build_retrieval_queries,
    _merge_query_results,
    answer_question,
)
from marsrag.query_understanding import QueryPlan


def _plan(**overrides):
    values = {
        "route": "simple",
        "standalone_query": "What does the report say?",
        "search_queries": ["What does the report say?", "report findings"],
        "sub_questions": [],
        "clarification_question": None,
    }
    values.update(overrides)
    return QueryPlan(**values)


def test_answer_question_analyzes_and_fuses_query_results():
    first = [{"chunk_id": "chunk-1", "content": "Evidence 1"}]
    expected = AnswerResponse(
        answer="Verified answer",
        abstained=False,
        verified=True,
        citations=[],
    )
    with (
        patch("marsrag.qa.analyze_query", return_value=_plan()) as analyze,
        patch("marsrag.qa.search_hybrid", return_value=first) as retrieve,
        patch("marsrag.qa.generate_answer", return_value=expected) as generate,
    ):
        result = answer_question(
            "  What does the report say?  ",
            report_no="NASA-HRP-EVIDENCE",
            top_k=7,
            section_path="II",
            chunk_type="text",
            use_reranker=True,
            model_tier="judge",
        )

    analyze.assert_called_once_with(
        "What does the report say?",
        conversation_history=None,
    )
    assert retrieve.call_count == 1
    assert retrieve.call_args_list[0].args == ("What does the report say?",)
    for call in retrieve.call_args_list:
        assert call.kwargs == {
            "top_k": 7,
            "report_no": "NASA-HRP-EVIDENCE",
            "section_path": "II",
            "chunk_type": "text",
            "use_reranker": True,
        }
    fused_sources = generate.call_args.args[1]
    assert [item["chunk_id"] for item in fused_sources] == ["chunk-1"]
    generate.assert_called_once_with(
        "What does the report say?",
        fused_sources,
        model_tier="judge",
    )
    assert result is expected


def test_multi_hop_query_uses_decomposed_sub_questions():
    plan = _plan(
        route="multi_hop",
        standalone_query="How did the program evolve and what were its outcomes?",
        search_queries=["program evolution", "program outcomes"],
        sub_questions=[
            "What milestones changed the program?",
            "What outcomes were reported?",
        ],
    )
    with (
        patch("marsrag.qa.analyze_query", return_value=plan),
        patch(
            "marsrag.qa.search_hybrid",
            side_effect=[
                [{"chunk_id": "a", "content": "Milestones"}],
                [{"chunk_id": "b", "content": "Outcomes"}],
                [{"chunk_id": "a", "content": "Milestones"}],
                [{"chunk_id": "c", "content": "Extra"}],
                [{"chunk_id": "d", "content": "Search variant"}],
            ],
        ) as retrieve,
        patch("marsrag.qa.generate_answer") as generate,
    ):
        answer_question("How did the program evolve and what were its outcomes?")

    assert [call.args[0] for call in retrieve.call_args_list] == [
        "How did the program evolve and what were its outcomes?",
        "What milestones changed the program?",
        "What outcomes were reported?",
        "program evolution",
        "program outcomes",
    ]
    assert retrieve.call_count == 5
    evidence = generate.call_args.args[1]
    assert {item["chunk_id"] for item in evidence} == {"a", "b", "c", "d"}


def test_multi_hop_retrieval_query_count_is_bounded():
    plan = _plan(
        route="multi_hop",
        sub_questions=[f"Sub question {i}" for i in range(1, 5)],
        search_queries=["search one", "search two", "search three"],
    )
    queries = _build_retrieval_queries(plan)
    assert len(queries) == 5
    assert queries[0] == "What does the report say?"
    assert queries[1:5] == [f"Sub question {i}" for i in range(1, 5)]


def test_answer_question_returns_clarification_without_retrieval():
    plan = _plan(
        route="clarification",
        clarification_question="Which mission are you referring to?",
    )
    with (
        patch("marsrag.qa.analyze_query", return_value=plan),
        patch("marsrag.qa.search_hybrid") as retrieve,
        patch("marsrag.qa.generate_answer") as generate,
    ):
        result = answer_question("What about its timeline?")

    assert result.abstained is True
    assert result.verified is False
    assert result.answer == "Which mission are you referring to?"
    retrieve.assert_not_called()
    generate.assert_not_called()


def test_merge_query_results_deduplicates_and_ranks():
    merged = _merge_query_results(
        [
            [{"chunk_id": "a", "content": "A"}, {"chunk_id": "b", "content": "B"}],
            [{"chunk_id": "b", "content": "B"}, {"chunk_id": "c", "content": "C"}],
        ],
        top_k=3,
    )
    assert [item["chunk_id"] for item in merged] == ["b", "a", "c"]
    assert len({item["chunk_id"] for item in merged}) == 3


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_answer_question_rejects_empty_question(question):
    with patch("marsrag.qa.analyze_query") as analyze:
        with pytest.raises(ValueError, match="Question cannot be empty"):
            answer_question(question)
    analyze.assert_not_called()


def test_answer_question_applies_extracted_chunk_type_filter():
    plan = _plan(chunk_type="table")
    with (
        patch("marsrag.qa.analyze_query", return_value=plan),
        patch("marsrag.qa.search_hybrid", return_value=[]) as retrieve,
        patch("marsrag.qa.generate_answer"),
    ):
        answer_question("Show me the tables about mission risks")

    assert retrieve.call_count == 1
    assert all(call.kwargs["chunk_type"] == "table" for call in retrieve.call_args_list)


def test_explicit_chunk_type_overrides_extracted_filter():
    plan = _plan(chunk_type="table")
    with (
        patch("marsrag.qa.analyze_query", return_value=plan),
        patch("marsrag.qa.search_hybrid", return_value=[]) as retrieve,
        patch("marsrag.qa.generate_answer"),
    ):
        answer_question("Show me the tables", chunk_type="text")

    assert retrieve.call_count == 1
    assert all(call.kwargs["chunk_type"] == "text" for call in retrieve.call_args_list)


def test_answer_question_passes_history_to_query_analysis():
    history = [
        {"role": "user", "content": "What risks are identified?"},
        {"role": "assistant", "content": "Several risks are discussed."},
    ]
    with (
        patch("marsrag.qa.analyze_query", return_value=_plan()) as analyze,
        patch("marsrag.qa.search_hybrid", return_value=[]),
        patch("marsrag.qa.generate_answer"),
    ):
        answer_question("Which relate to radiation?", conversation_history=history)

    analyze.assert_called_once_with(
        "Which relate to radiation?",
        conversation_history=history,
    )
