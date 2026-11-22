from __future__ import annotations

from marsrag.generation import AnswerResponse, generate_answer
from marsrag.retrieval.dense import DEFAULT_REPORT_NO
from marsrag.retrieval.hybrid import search_hybrid


def answer_question(
    question: str,
    *,
    report_no: str = DEFAULT_REPORT_NO,
    top_k: int = 5,
    section_path: str | None = None,
    chunk_type: str | None = None,
    use_reranker: bool = False,
    model_tier: str = "main",
) -> AnswerResponse:
    """Run hybrid retrieval followed by grounded answer generation.

    Retrieval applies the document and metadata filters before generation.
    Citation verification is performed by generate_answer against the exact
    chunks returned by retrieval.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    sources = search_hybrid(
        question.strip(),
        top_k=top_k,
        report_no=report_no,
        section_path=section_path,
        chunk_type=chunk_type,
        use_reranker=use_reranker,
    )
    return generate_answer(question.strip(), sources, model_tier=model_tier)
