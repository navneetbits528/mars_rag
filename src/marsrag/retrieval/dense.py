
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
) -> list[dict[str, Any]]:
    """Retrieve the most similar chunks from a selected report."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    if not report_no or not report_no.strip():
        raise ValueError("report_no cannot be empty.")

    query_vector = embed_query(query.strip())

    # A JSON array string is also a valid pgvector literal.
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
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    results = fetch_all(
        sql,
        (vector_literal, report_no, vector_literal, top_k),
    )

    return [dict(row) for row in results]
