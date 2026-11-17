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
