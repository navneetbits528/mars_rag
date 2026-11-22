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


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_history: list[ConversationMessage] = Field(default_factory=list, max_length=10)
    report_no: str | None = Field(default=None, min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=50)
    section_path: str | None = Field(default=None, min_length=1, max_length=500)
    chunk_type: str | None = Field(default=None, min_length=1, max_length=50)
    use_reranker: bool = False
    model_tier: Literal["fast", "main", "deep", "judge"] = "main"
