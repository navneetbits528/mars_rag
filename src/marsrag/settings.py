from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)
    openai_api_key: str = Field(default="", repr=False)
    fast_model: str = "gpt-4o-mini"
    main_model: str = "gpt-4o-mini"
    deep_model: str = ""
    judge_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    database_url: str = "postgresql://marsrag:marsrag_dev_password@localhost:5432/marsrag"
    admin_api_key: str = Field(default="change-this-demo-admin-key", repr=False)
    openai_max_requests: int = 10
    max_requests_per_qa: int = 12
    daily_cost_limit_usd: float = 2.0
    max_retries: int = 2
    retry_base_seconds: float = 0.5
    request_timeout_seconds: float = 30.0
    model_context_limit: int = 128000
    input_price_per_1m_usd: float = 0.15
    output_price_per_1m_usd: float = 0.60
    embedding_price_per_1m_usd: float = 0.02
    prompt_version: str = "v1"
    corpus_version: int = 1
    log_level: str = "INFO"
    embedding_batch_token_limit: int = 150000
    embedding_batch_item_limit: int = 2048
    embedding_max_input_tokens: int = 8000
    chunk_size_chars: int = 1000
    chunk_overlap_ratio: float = 0.15
    min_chunk_chars: int = 80
    # Phase 3
    ocr_enabled: bool = True
    ocr_min_text_chars: int = 35
    ocr_language: str = "eng"
    child_chunk_tokens: int = 300
    facts_extraction_enabled: bool = False
    facts_max_chunks_per_call: int = 8

    @field_validator("embed_dim")
    @classmethod
    def valid_dim(cls, value):
        if not 1 <= value <= 2000:
            raise ValueError("EMBED_DIM must be between 1 and 2000.")
        return value

    @field_validator("openai_max_requests", "max_requests_per_qa")
    @classmethod
    def valid_budget(cls, value):
        if value < 1:
            raise ValueError("API request limits must be at least 1.")
        return value

    @field_validator("chunk_overlap_ratio")
    @classmethod
    def valid_overlap(cls, value):
        if not 0 <= value < 0.5:
            raise ValueError("CHUNK_OVERLAP_RATIO must be in [0, 0.5).")
        return value

    @field_validator("child_chunk_tokens", "facts_max_chunks_per_call", "ocr_min_text_chars", "embedding_max_input_tokens")
    @classmethod
    def positive_phase3_values(cls, value):
        if value < 1:
            raise ValueError("Phase 3 size settings must be positive.")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
