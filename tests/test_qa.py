from unittest.mock import patch

import pytest

from marsrag.generation import AnswerResponse
from marsrag.qa import answer_question, _merge_query_results
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
    second = [
        {"chunk_id": "chunk-1", "content": "Evidence 1"},
        {"chunk_id": "chunk-2", "content": "Evidence 2"},
    ]
    expected = AnswerResponse(
        answer="Verified answer",
        abstained=False,
        verified=True,
        citations=[],
    )
    with (
        patch("marsrag.qa.analyze_query", return_value=_plan()) as analyze,
        patch("marsrag.qa.search_hybrid", side_effect=[first, second]) as retrieve,
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

    analyze.assert_called_once_with("What does the report say?")
    assert retrieve.call_count == 2
    assert retrieve.call_args_list[0].args == ("What does the report say?",)
    assert retrieve.call_args_list[1].args == ("report findings",)
    for call in retrieve.call_args_list:
        assert call.kwargs == {
            "top_k": 7,
            "report_no": "NASA-HRP-EVIDENCE",
            "section_path": "II",
            "chunk_type": "text",
            "use_reranker": True,
        }
    fused_sources = generate.call_args.args[1]
    assert [item["chunk_id"] for item in fused_sources] == ["chunk-1", "chunk-2"]
    generate.assert_called_once_with(
        "What does the report say?",
        fused_sources,
        model_tier="judge",
    )
    assert result is expected


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
