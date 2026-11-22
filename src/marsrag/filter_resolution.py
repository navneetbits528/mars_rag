from __future__ import annotations

import re
from dataclasses import dataclass

from marsrag.db import fetch_all
from marsrag.retrieval.dense import DEFAULT_REPORT_NO

SUPPORTED_CHUNK_TYPES = {"text", "table", "figure_caption"}


@dataclass(frozen=True)
class ResolvedFilters:
    report_no: str
    section_path: str | None
    chunk_type: str | None


def resolve_filters(
    question: str,
    *,
    report_no: str | None = None,
    section_path: str | None = None,
    chunk_type: str | None = None,
    inferred_chunk_type: str | None = None,
) -> ResolvedFilters:
    """Resolve only exact corpus metadata matches; explicit API values win."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    documents = fetch_all(
        "SELECT report_no FROM documents WHERE status = 'current' ORDER BY report_no"
    )
    known_reports = [row["report_no"] for row in documents]
    if report_no is not None:
        selected_report = report_no.strip()
        if selected_report not in known_reports:
            raise ValueError(f"Unknown or non-current report_no: {selected_report}")
    else:
        selected_report = _find_exact_value(question, known_reports) or DEFAULT_REPORT_NO
        if selected_report not in known_reports:
            raise ValueError(
                f"Default report '{DEFAULT_REPORT_NO}' is not present in the current corpus."
            )

    section_rows = fetch_all(
        """
        SELECT DISTINCT section_path
        FROM chunks
        WHERE report_no = %s AND status = 'current'
          AND section_path IS NOT NULL AND btrim(section_path) <> ''
        ORDER BY section_path
        """,
        (selected_report,),
    )
    known_sections = [row["section_path"] for row in section_rows]
    if section_path is not None:
        selected_section = section_path.strip()
        if selected_section not in known_sections:
            raise ValueError(
                f"Unknown section_path for report '{selected_report}': {selected_section}"
            )
    else:
        selected_section = _find_exact_value(question, known_sections)

    selected_type = chunk_type if chunk_type is not None else inferred_chunk_type
    if selected_type is not None:
        selected_type = selected_type.strip()
        if selected_type not in SUPPORTED_CHUNK_TYPES:
            raise ValueError(f"Unsupported chunk_type: {selected_type}")

    return ResolvedFilters(
        report_no=selected_report,
        section_path=selected_section,
        chunk_type=selected_type,
    )


def _find_exact_value(text: str, values: list[str]) -> str | None:
    """Find a corpus value only when it appears as a complete literal token."""
    matches = [
        value for value in values
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", text, re.I)
    ]
    return max(matches, key=len) if matches else None
