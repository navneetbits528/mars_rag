from unittest.mock import patch

from marsrag.generation import (
    AnswerCitation,
    ClaimCheck,
    ClaimVerification,
    GeneratedAnswer,
    generate_answer,
    verify_citations,
    verify_claim_support,
)


def _sources():
    return [
        {
            "chunk_id": "chunk-1",
            "report_no": "NASA-HRP-EVIDENCE",
            "page": 12,
            "url": "https://example.test/report.pdf",
            "section_title": "Food safety",
            "content": "The evidence report identifies microbial growth as a food safety risk.",
        }
    ]


def _generated():
    return GeneratedAnswer(
        answer="Microbial growth is identified as a food safety risk.",
        sufficient_evidence=True,
        citations=[
            AnswerCitation(
                chunk_id="chunk-1",
                quote="microbial growth as a food safety risk",
            )
        ],
    )


def test_verify_citations_accepts_exact_quote_and_known_chunk():
    citations, valid = verify_citations(
        [AnswerCitation(chunk_id="chunk-1", quote="microbial growth as a food safety risk")],
        _sources(),
    )
    assert valid is True
    assert citations[0].page == 12
    assert citations[0].report_no == "NASA-HRP-EVIDENCE"


def test_verify_citations_rejects_unknown_chunk():
    citations, valid = verify_citations(
        [AnswerCitation(chunk_id="missing", quote="microbial growth")],
        _sources(),
    )
    assert valid is False
    assert citations == []


def test_verify_citations_rejects_invented_quote():
    citations, valid = verify_citations(
        [AnswerCitation(chunk_id="chunk-1", quote="This proves a cure exists")],
        _sources(),
    )
    assert valid is False
    assert citations == []


def test_verify_claim_support_accepts_fully_supported_claims():
    citations, valid = verify_citations(_generated().citations, _sources())
    verdict = ClaimVerification(
        claims=[
            ClaimCheck(
                claim="Microbial growth is a food safety risk.",
                supported=True,
                citation_chunk_ids=["chunk-1"],
            )
        ]
    )
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", return_value=verdict),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        assert verify_claim_support(_generated().answer, citations, _sources()) is True


def test_verify_claim_support_rejects_unsupported_claim():
    citations, _ = verify_citations(_generated().citations, _sources())
    verdict = ClaimVerification(
        claims=[
            ClaimCheck(
                claim="Microbial growth causes permanent illness.",
                supported=False,
                citation_chunk_ids=["chunk-1"],
            )
        ]
    )
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", return_value=verdict),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        assert verify_claim_support("Microbial growth causes permanent illness.", citations, _sources()) is False


def test_verify_claim_support_rejects_unknown_evidence_ids():
    citations, _ = verify_citations(_generated().citations, _sources())
    verdict = ClaimVerification(
        claims=[
            ClaimCheck(
                claim="A factual claim.",
                supported=True,
                citation_chunk_ids=["not-cited"],
            )
        ]
    )
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", return_value=verdict),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        assert verify_claim_support("A factual claim.", citations, _sources()) is False


def test_generation_returns_abstention_without_sources():
    result = generate_answer("What is the risk?", [])
    assert result.abstained is True
    assert result.verified is False


def test_generation_returns_verified_answer_after_claim_check():
    verdict = ClaimVerification(
        claims=[
            ClaimCheck(
                claim="Microbial growth is a food safety risk.",
                supported=True,
                citation_chunk_ids=["chunk-1"],
            )
        ]
    )
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", side_effect=[_generated(), verdict]),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = generate_answer("What is a food safety risk?", _sources())

    assert result.abstained is False
    assert result.verified is True
    assert result.citations[0].chunk_id == "chunk-1"


def test_generation_abstains_when_model_says_evidence_is_insufficient():
    generated = GeneratedAnswer(answer="", sufficient_evidence=False, citations=[])
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", return_value=generated),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = generate_answer("What is a food safety risk?", _sources())

    assert result.abstained is True
    assert result.citations == []


def test_generation_abstains_on_fabricated_citation():
    generated = GeneratedAnswer(
        answer="Unsupported claim.",
        sufficient_evidence=True,
        citations=[AnswerCitation(chunk_id="chunk-1", quote="invented supporting quote")],
    )
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", return_value=generated),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = generate_answer("Question?", _sources())

    assert result.abstained is True
    assert result.verified is False


def test_generation_abstains_when_claim_verification_fails():
    unsupported = ClaimVerification(
        claims=[
            ClaimCheck(
                claim="An unsupported extra claim.",
                supported=False,
                citation_chunk_ids=["chunk-1"],
            )
        ]
    )
    with (
        patch("marsrag.generation.build_chat_model") as build_model,
        patch("marsrag.generation.invoke_chat", side_effect=[_generated(), unsupported]),
    ):
        build_model.return_value.with_structured_output.return_value = object()
        result = generate_answer("What is a food safety risk?", _sources())

    assert result.abstained is True
    assert result.verified is False
    assert result.reason == "Claim-level evidence verification failed."
