import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from marsrag.db import check_database, close_pool
from marsrag.generation import AnswerResponse
from marsrag.models import AskRequest, HealthResponse
from marsrag.qa import answer_question, stream_answer_question
from marsrag.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage resources for the lifetime of the API process."""
    try:
        yield
    finally:
        close_pool()


app = FastAPI(
    title="Mars Missions RAG API",
    version="0.2.0",
    lifespan=lifespan,
)


def _run_question(request: AskRequest) -> AnswerResponse:
    return answer_question(
        request.question,
        report_no=request.report_no,
        top_k=request.top_k,
        section_path=request.section_path,
        chunk_type=request.chunk_type,
        use_reranker=request.use_reranker,
        model_tier=request.model_tier,
        conversation_history=[
            item.model_dump() for item in request.conversation_history
        ],
    )


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
        return _run_question(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream provisional draft tokens and then the final verified answer as SSE."""

    def event_stream() -> Iterator[str]:
        try:
            for event_name, data in stream_answer_question(
                request.question,
                report_no=request.report_no,
                top_k=request.top_k,
                section_path=request.section_path,
                chunk_type=request.chunk_type,
                use_reranker=request.use_reranker,
                model_tier=request.model_tier,
                conversation_history=[
                    item.model_dump() for item in request.conversation_history
                ],
            ):
                payload = json.dumps(data, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {payload}\n\n"
            yield "event: done\ndata: {}\n\n"
        except ValueError as exc:
            payload = json.dumps({"detail": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
