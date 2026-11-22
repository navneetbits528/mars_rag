
from __future__ import annotations

from typing import Any

from marsrag.db import fetch_all

DEFAULT_MAX_PARENT_CHARS = 3000


def _extract_context(
    parent_content: str,
    result: dict[str, Any],
    max_chars: int,
) -> str:
    """Extract context using stored parent offsets."""

    metadata = result.get("metadata") or {}

    start = metadata.get("parent_start_char")
    end = metadata.get("parent_end_char")

    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(parent_content)
    ):
        half_window = max_chars // 2

        context_start = max(0, start - half_window)
        context_end = min(len(parent_content), context_start + max_chars)

        context_start = max(0, context_end - max_chars)

        return parent_content[context_start:context_end]

    # Backward compatibility for records ingested before offsets existed.
    child_content = (result.get("content") or "").strip()

    if child_content:
        position = parent_content.find(child_content)

        if position >= 0:
            context_start = max(0, position - max_chars // 2)
            context_end = min(
                len(parent_content),
                context_start + max_chars,
            )
            context_start = max(0, context_end - max_chars)

            return parent_content[context_start:context_end]

    # Never fall back to the beginning of an unrelated parent.
    return child_content[:max_chars]


def expand_parent_context(
    results: list[dict[str, Any]],
    max_parent_chars: int = DEFAULT_MAX_PARENT_CHARS,
) -> list[dict[str, Any]]:
    """Attach bounded, position-aware parent context to retrieval results."""

    if max_parent_chars < 1:
        raise ValueError("max_parent_chars must be at least 1.")

    if not results:
        return []

    parent_ids = list(
        dict.fromkeys(
            result["parent_id"]
            for result in results
            if result.get("parent_id")
        )
    )

    if not parent_ids:
        return [
            {
                **result,
                "parent_context": None,
                "parent_section_title": None,
                "parent_page_start": None,
                "parent_page_end": None,
            }
            for result in results
        ]

    sql = """
        SELECT
            parent_id,
            section_title,
            page_start,
            page_end,
            content
        FROM parents
        WHERE parent_id = ANY(%s)
    """

    parent_rows = fetch_all(sql, (parent_ids,))
    parents = {
        row["parent_id"]: dict(row)
        for row in parent_rows
    }

    expanded_results = []

    for result in results:
        expanded = dict(result)
        parent = parents.get(result.get("parent_id"))

        if not parent:
            expanded.update({
                "parent_context": None,
                "parent_section_title": None,
                "parent_page_start": None,
                "parent_page_end": None,
            })
            expanded_results.append(expanded)
            continue

        parent_content = parent["content"]

        if len(parent_content) <= max_parent_chars:
            context = parent_content
        else:
            context = _extract_context(
                parent_content,
                result,
                max_parent_chars,
            )

        expanded.update({
            "parent_context": context,
            "parent_section_title": parent["section_title"],
            "parent_page_start": parent["page_start"],
            "parent_page_end": parent["page_end"],
        })

        expanded_results.append(expanded)

    return expanded_results
