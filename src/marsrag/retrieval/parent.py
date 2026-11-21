
from __future__ import annotations

from typing import Any

from marsrag.db import fetch_all

DEFAULT_MAX_PARENT_CHARS = 3000


def expand_parent_context(
    results: list[dict[str, Any]],
    max_parent_chars: int = DEFAULT_MAX_PARENT_CHARS,
) -> list[dict[str, Any]]:
    """
    Attach bounded parent context to hybrid retrieval results.

    Child content and retrieval ranking fields remain unchanged.
    Duplicate parent records are fetched only once.
    """

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
        parent_id = result.get("parent_id")
        parent = parents.get(parent_id)

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
            child_content = result.get("content", "").strip()

            if not child_content:
                context = parent_content[:max_parent_chars]
            else:
                position = parent_content.find(child_content)

                if position == -1:
                    context = parent_content[:max_parent_chars]
                else:
                    half_window = max_parent_chars // 2
                    start = max(0, position - half_window)
                    end = min(
                        len(parent_content),
                        start + max_parent_chars,
                    )

                    start = max(0, end - max_parent_chars)
                    context = parent_content[start:end]

        expanded.update({
            "parent_context": context,
            "parent_section_title": parent["section_title"],
            "parent_page_start": parent["page_start"],
            "parent_page_end": parent["page_end"],
        })

        expanded_results.append(expanded)

    return expanded_results
