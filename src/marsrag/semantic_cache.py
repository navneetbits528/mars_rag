from __future__ import annotations

import hashlib
import json
from typing import Any

from marsrag.db import execute, fetch_one
from marsrag.generation import AnswerResponse
from marsrag.llm import embed_query

DEFAULT_SIMILARITY_THRESHOLD = 0.93


def build_cache_key(
    *,
    report_no: str,
    section_path: str | None,
    chunk_type: str | None,
    top_k: int,
    use_reranker: bool,
    model_tier: str,
    prompt_version: str,
    corpus_version: int,
) -> str:
    """Create a deterministic scope key so answers never cross retrieval contexts."""
    scope = {
        "report_no": report_no,
        "section_path": section_path,
        "chunk_type": chunk_type,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "model_tier": model_tier,
        "prompt_version": prompt_version,
        "corpus_version": corpus_version,
    }
    encoded = json.dumps(scope, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def find_cached_answer(
    question: str,
    cache_key: str,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> AnswerResponse | None:
    """Return a verified, unexpired answer only when its question is highly similar."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1].")

    vector = json.dumps(embed_query(question.strip()))
    row = fetch_one(
        """
        SELECT answer, 1 - (question_embedding <=> %s::vector) AS similarity
        FROM semantic_cache
        WHERE cache_key = %s
          AND verified = TRUE
          AND expires_at > now()
        ORDER BY question_embedding <=> %s::vector
        LIMIT 1
        """,
        (vector, cache_key, vector),
    )
    if row is None or float(row["similarity"]) < similarity_threshold:
        return None

    answer = row["answer"]
    if isinstance(answer, str):
        answer = json.loads(answer)
    response = AnswerResponse.model_validate(answer)
    if response.abstained or not response.verified:
        return None
    return response


def store_verified_answer(
    question: str,
    cache_key: str,
    response: AnswerResponse,
    *,
    ttl_seconds: int,
) -> bool:
    """Persist only fully verified, non-abstaining answers."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be at least 1.")
    if response.abstained or not response.verified:
        return False

    vector = json.dumps(embed_query(question.strip()))
    answer_json = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
    execute(
        """
        INSERT INTO semantic_cache
            (cache_key, question, question_embedding, answer, sources, verified, expires_at)
        VALUES
            (%s, %s, %s::vector, %s::jsonb, '[]'::jsonb, TRUE,
             now() + (%s * interval '1 second'))
        """,
        (cache_key, question.strip(), vector, answer_json, ttl_seconds),
    )
    return True


def delete_expired_cache_entries(*, limit: int = 1000) -> int:
    """Delete a bounded batch of expired entries to keep the cache table compact."""
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000.")
    return execute(
        """
        DELETE FROM semantic_cache
        WHERE cache_id IN (
            SELECT cache_id
            FROM semantic_cache
            WHERE expires_at <= now()
            ORDER BY expires_at
            LIMIT %s
        )
        """,
        (limit,),
    )
