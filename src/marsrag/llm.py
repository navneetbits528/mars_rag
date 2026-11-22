from __future__ import annotations
import logging, time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import APIConnectionError, APIStatusError, RateLimitError
from marsrag.settings import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")
class RequestBudgetExceeded(RuntimeError): pass
class MissingAPIKeyError(RuntimeError): pass

_request_api_calls: ContextVar[dict[str, int] | None] = ContextVar("request_api_calls", default=None)

@contextmanager
def request_api_budget(max_requests: int):
    """Apply an isolated API-call ceiling to the current QA request."""
    if max_requests < 1:
        raise ValueError("max_requests must be at least 1.")
    state = {"used": 0, "limit": max_requests}
    token = _request_api_calls.set(state)
    try:
        yield state
    finally:
        try:
            _request_api_calls.reset(token)
        except ValueError:
            # StreamingResponse may advance a synchronous generator in different
            # copied contexts across thread-pool iterations. A ContextVar token
            # can only be reset in the context where it was created.
            _request_api_calls.set(None)

@dataclass
class UsageTotals:
    api_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0

class DemoUsageTracker:
    def __init__(self):
        self.settings = get_settings()
        self.totals = UsageTotals()

    def reserve_request(self):
        request_state = _request_api_calls.get()
        if request_state is not None:
            if request_state["used"] >= request_state["limit"]:
                raise RequestBudgetExceeded(
                    f"QA request API budget exhausted ({request_state['limit']})."
                )
            request_state["used"] += 1
        if self.totals.api_requests >= self.settings.openai_max_requests:
            raise RequestBudgetExceeded(f"Demo API request budget exhausted ({self.settings.openai_max_requests}).")
        if self.totals.estimated_cost_usd >= self.settings.daily_cost_limit_usd:
            raise RequestBudgetExceeded("Configured daily API cost ceiling reached.")
        self.totals.api_requests += 1

    def record_chat(self, model, input_tokens, output_tokens):
        s=self.settings
        cost=(input_tokens*s.input_price_per_1m_usd+output_tokens*s.output_price_per_1m_usd)/1_000_000
        self.totals.input_tokens += input_tokens
        self.totals.output_tokens += output_tokens
        self.totals.estimated_cost_usd += cost
        logger.info("chat_usage model=%s in=%d out=%d cost_usd=%.8f",model,input_tokens,output_tokens,cost)

    def record_embedding(self, model, tokens):
        cost=tokens*self.settings.embedding_price_per_1m_usd/1_000_000
        self.totals.embedding_tokens += tokens
        self.totals.estimated_cost_usd += cost
        logger.info("embedding_usage model=%s tokens=%d cost_usd=%.8f",model,tokens,cost)

usage_tracker=DemoUsageTracker()

def _key():
    key=get_settings().openai_api_key
    if not key: raise MissingAPIKeyError("Set OPENAI_API_KEY in .env.")
    return key

def build_chat_model(tier: str="main", **overrides: Any) -> ChatOpenAI:
    s=get_settings()
    names={"fast":s.fast_model,"main":s.main_model,"deep":s.deep_model,"judge":s.judge_model}
    if tier not in names: raise ValueError(f"Unknown model tier: {tier}")
    if not names[tier]: raise ValueError(f"Model tier {tier!r} is not configured.")
    args={"model":names[tier],"api_key":_key(),"timeout":s.request_timeout_seconds,"max_retries":0}
    if tier in {"main","deep","judge"}: args["temperature"]=0
    args.update(overrides)
    return ChatOpenAI(**args)

def build_embeddings() -> OpenAIEmbeddings:
    s=get_settings()
    return OpenAIEmbeddings(model=s.embed_model,dimensions=s.embed_dim,api_key=_key(),
                            max_retries=0,request_timeout=s.request_timeout_seconds,chunk_size=s.embedding_batch_item_limit)

def call_with_retry(operation: Callable[[],T]) -> T:
    s=get_settings()
    retryable=(RateLimitError,APIConnectionError,APIStatusError)
    for attempt in range(s.max_retries+1):
        usage_tracker.reserve_request()
        try: return operation()
        except retryable as exc:
            status=getattr(exc,"status_code",None)
            if attempt>=s.max_retries or (status is not None and status<500 and status!=429): raise
            time.sleep(s.retry_base_seconds*(2**attempt))
    raise RuntimeError("Retry loop exited unexpectedly.")

def invoke_chat(model: ChatOpenAI,messages: list[Any]) -> Any:
    result=call_with_retry(lambda:model.invoke(messages))
    meta=getattr(result,"usage_metadata",None) or {}
    usage_tracker.record_chat(getattr(model,"model_name","configured-model"),
                              int(meta.get("input_tokens",0)),int(meta.get("output_tokens",0)))
    return result

def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts: return []
    s=get_settings()
    vectors=call_with_retry(lambda:build_embeddings().embed_documents(texts))
    import tiktoken
    try: enc=tiktoken.encoding_for_model(s.embed_model)
    except KeyError: enc=tiktoken.get_encoding("cl100k_base")
    usage_tracker.record_embedding(s.embed_model,sum(len(enc.encode(x)) for x in texts))
    return vectors

def embed_query(text: str) -> list[float]:
    return embed_documents([text])[0]

def stream_chat(model: ChatOpenAI, messages: list[Any]) -> Iterator[str]:
    """Stream text chunks while charging one request to the active API budget."""
    usage_tracker.reserve_request()
    input_tokens = 0
    output_tokens = 0
    usage_seen = False
    try:
        for chunk in model.stream(messages):
            meta = getattr(chunk, "usage_metadata", None) or {}
            if meta:
                usage_seen = True
                input_tokens = max(input_tokens, int(meta.get("input_tokens", 0)))
                output_tokens = max(output_tokens, int(meta.get("output_tokens", 0)))
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                if content:
                    yield content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            yield str(text)
    finally:
        if usage_seen:
            usage_tracker.record_chat(
                getattr(model, "model_name", "configured-model"),
                input_tokens,
                output_tokens,
            )

