from fastapi import FastAPI, HTTPException

from marsrag.db import check_database, close_pool
from marsrag.generation import AnswerResponse
from marsrag.models import AskRequest, HealthResponse
from marsrag.qa import answer_question
from marsrag.settings import get_settings

app = FastAPI(title="Mars Missions RAG API", version="0.2.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    database_ok = check_database()
    return HealthResponse(
        status="ok" if database_ok else "degraded",
        database="ok" if database_ok else "unavailable",
        corpus_version=get_settings().corpus_version,
    )


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    """Retrieve document evidence and return a citation-verified answer."""
    try:
        return answer_question(
            request.question,
            report_no=request.report_no,
            top_k=request.top_k,
            section_path=request.section_path,
            chunk_type=request.chunk_type,
            use_reranker=request.use_reranker,
            model_tier=request.model_tier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.on_event("shutdown")
def shutdown_event() -> None:
    close_pool()
