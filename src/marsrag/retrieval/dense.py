from __future__ import annotations

import json
from typing import Any

from marsrag.db import fetch_all
from marsrag.llm import embed_query

DEFAULT_REPORT_NO = "NASA-HRP-EVIDENCE"
MAX_TOP_K = 50


def search_dense(
    query: str,
    top_k: int = 5,
    report_no: str = DEFAULT_REPORT_NO,
    section_path: str | None = None,
    chunk_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve similar current chunks, optionally scoped to a section/type."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    if not report_no or not report_no.strip():
        raise ValueError("report_no cannot be empty.")
    if section_path is not None and not section_path.strip():
        raise ValueError("section_path cannot be empty when provided.")
    if chunk_type is not None and not chunk_type.strip():
        raise ValueError("chunk_type cannot be empty when provided.")

    query_vector = embed_query(query.strip())
    vector_literal = json.dumps(query_vector)

    sql = """
        SELECT
            chunk_id,
            report_no,
            page,
            title,
            url,
            section_title,
            section_path,
            chunk_type,
            parent_id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        WHERE report_no = %s
          AND status = 'current'
          AND embedding IS NOT NULL
          AND (%s IS NULL OR section_path = %s)
          AND (%s IS NULL OR chunk_type = %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    results = fetch_all(
        sql,
        (
            vector_literal,
            report_no,
            section_path,
            section_path,
            chunk_type,
            chunk_type,
            vector_literal,
            top_k,
        ),
    )
    return [dict(row) for row in results]
