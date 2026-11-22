from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UsageRecord(BaseModel):
    request_type: Literal["chat", "embedding"]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    request_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]
    corpus_version: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    report_no: str = Field(default="NASA-HRP-EVIDENCE", min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=50)
    section_path: str | None = None
    chunk_type: str | None = None
    use_reranker: bool = False
    model_tier: Literal["fast", "main", "deep", "judge"] = "main"
