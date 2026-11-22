from unittest.mock import patch

import pytest

from marsrag.query_understanding import QueryPlan, analyze_query


def _plan(**overrides):
    values = {
        "route": "simple",
        "standalone_query": "What food safety risks are identified?",
        "search_queries": ["food safety risks"],
        "sub_questions": [],
        "clarification_question": None,
    }
    values.update(overrides)
    return QueryPlan(**values)


def test_analyze_query_returns_structured_simple_route():
    expected = _plan()
    with (
        patch("marsrag.query_understanding.build_chat_model") as build_model,
        patch("marsrag.query_understanding.invoke_chat", return_value=expected),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = analyze_query("What food safety risks are identified?")

    assert result.route == "simple"
    assert result.search_queries == ["food safety risks"]


def test_analyze_query_preserves_multi_hop_subquestions():
    expected = _plan(
        route="multi_hop",
        sub_questions=[
            "What risks are identified?",
            "What mitigation measures are described?",
        ],
    )
    with (
        patch("marsrag.query_understanding.build_chat_model") as build_model,
        patch("marsrag.query_understanding.invoke_chat", return_value=expected),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = analyze_query("Which risks have mitigation measures?")

    assert result.route == "multi_hop"
    assert len(result.sub_questions) == 2


def test_analyze_query_downgrades_invalid_multi_hop_plan():
    expected = _plan(route="multi_hop", sub_questions=["Only one step"])
    with (
        patch("marsrag.query_understanding.build_chat_model") as build_model,
        patch("marsrag.query_understanding.invoke_chat", return_value=expected),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = analyze_query("A complex question")

    assert result.route == "simple"
    assert result.sub_questions == []


def test_analyze_query_falls_back_when_model_fails():
    with (
        patch("marsrag.query_understanding.build_chat_model", side_effect=RuntimeError),
    ):
        result = analyze_query("  Original question  ")

    assert result.route == "simple"
    assert result.standalone_query == "Original question"
    assert result.search_queries == ["Original question"]


def test_analyze_query_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        analyze_query("   ")


def test_query_plan_rejects_more_than_three_search_queries():
    with pytest.raises(ValueError):
        _plan(search_queries=["one", "two", "three", "four"])


@pytest.mark.parametrize(
    "chunk_type",
    ["text", "table", "figure_caption", None],
)
def test_query_plan_accepts_supported_chunk_types(chunk_type):
    plan = _plan(chunk_type=chunk_type)
    assert plan.chunk_type == chunk_type


def test_query_plan_rejects_unsupported_chunk_type():
    with pytest.raises(ValueError):
        _plan(chunk_type="image")


def test_analyze_query_includes_recent_history_before_latest_question():
    expected = _plan(
        standalone_query="What radiation risks are described in the HRP report?",
        search_queries=["HRP report radiation risks"],
    )
    history = [
        {"role": "user", "content": "What risks does the HRP report identify?"},
        {"role": "assistant", "content": "It identifies several risks."},
    ]
    with (
        patch("marsrag.query_understanding.build_chat_model") as build_model,
        patch(
            "marsrag.query_understanding.invoke_chat",
            return_value=expected,
        ) as invoke,
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = analyze_query(
            "Which of these are related to radiation?",
            conversation_history=history,
        )

    sent_messages = invoke.call_args.args[1]
    assert sent_messages[-3].content == history[0]["content"]
    assert sent_messages[-2].content == history[1]["content"]
    assert sent_messages[-1].content == (
        "Latest user question:\nWhich of these are related to radiation?"
    )
    assert result.standalone_query == expected.standalone_query


def test_analyze_query_caps_history_to_ten_messages():
    with (
        patch("marsrag.query_understanding.build_chat_model") as build_model,
        patch("marsrag.query_understanding.invoke_chat", return_value=_plan()) as invoke,
    ):
        build_model.return_value.with_structured_output.return_value = object()
        analyze_query(
            "Latest question",
            conversation_history=[
                {"role": "user", "content": f"turn {i}"} for i in range(12)
            ],
        )

    sent_messages = invoke.call_args.args[1]
    history_messages = sent_messages[1:-1]
    assert len(history_messages) == 10
    assert history_messages[0].content == "turn 2"
    assert history_messages[-1].content == "turn 11"
