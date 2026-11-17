from fastapi import FastAPI
from marsrag.db import check_database, close_pool
from marsrag.models import HealthResponse
from marsrag.settings import get_settings

app = FastAPI(title="Mars Missions RAG API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    database_ok = check_database()
    return HealthResponse(
        status="ok" if database_ok else "degraded",
        database="ok" if database_ok else "unavailable",
        corpus_version=get_settings().corpus_version,
    )


@app.on_event("shutdown")
def shutdown_event() -> None:
    close_pool()
