from __future__ import annotations

from typing import Any

from marsrag.db import fetch_all

DEFAULT_REPORT_NO = "NASA-HRP-EVIDENCE"
MAX_TOP_K = 50


def search_keyword(
    query: str,
    top_k: int = 5,
    report_no: str = DEFAULT_REPORT_NO,
    section_path: str | None = None,
    chunk_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve current chunks with PostgreSQL full-text search."""

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

    sql = r"""
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
            ts_rank_cd(
                content_tsv,
                websearch_to_tsquery(
                    'english',
                    regexp_replace(%s, '\s+', ' OR ', 'g')
                )
            ) AS keyword_score
        FROM chunks
        WHERE report_no = %s
          AND status = 'current'
          AND (%s::text IS NULL OR section_path = %s::text)
          AND (%s::text IS NULL OR chunk_type = %s::text)
          AND content_tsv @@ websearch_to_tsquery(
              'english',
              regexp_replace(%s, '\s+', ' OR ', 'g')
          )
        ORDER BY keyword_score DESC
        LIMIT %s
    """

    results = fetch_all(
        sql,
        (
            query.strip(),
            report_no,
            section_path,
            section_path,
            chunk_type,
            chunk_type,
            query.strip(),
            top_k,
        ),
    )
    return [dict(row) for row in results]
