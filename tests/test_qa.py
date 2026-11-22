from unittest.mock import patch

import pytest

from marsrag.generation import AnswerResponse
from marsrag.qa import answer_question


def test_answer_question_runs_retrieval_then_generation():
    sources = [{"chunk_id": "chunk-1", "content": "Evidence"}]
    expected = AnswerResponse(
        answer="Verified answer",
        abstained=False,
        verified=True,
        citations=[],
    )
    with (
        patch("marsrag.qa.search_hybrid", return_value=sources) as retrieve,
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

    retrieve.assert_called_once_with(
        "What does the report say?",
        top_k=7,
        report_no="NASA-HRP-EVIDENCE",
        section_path="II",
        chunk_type="text",
        use_reranker=True,
    )
    generate.assert_called_once_with(
        "What does the report say?",
        sources,
        model_tier="judge",
    )
    assert result is expected


def test_answer_question_abstains_when_retrieval_returns_no_sources():
    expected = AnswerResponse(
        answer="I don't have enough evidence.",
        abstained=True,
        verified=False,
        reason="No evidence was retrieved.",
    )
    with (
        patch("marsrag.qa.search_hybrid", return_value=[]) as retrieve,
        patch("marsrag.qa.generate_answer", return_value=expected) as generate,
    ):
        result = answer_question("Question?")

    generate.assert_called_once_with("Question?", [], model_tier="main")
    assert result.abstained is True


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_answer_question_rejects_empty_question(question):
    with patch("marsrag.qa.search_hybrid") as retrieve:
        with pytest.raises(ValueError, match="Question cannot be empty"):
            answer_question(question)
    retrieve.assert_not_called()
