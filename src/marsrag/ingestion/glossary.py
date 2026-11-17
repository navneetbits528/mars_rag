from __future__ import annotations

import re
from dataclasses import dataclass

from marsrag.ingestion.models import ChunkRecord, PageText

# Captures forms such as "In Situ Resource Utilization (ISRU)".
_LONG_FORM = re.compile(r"\b([A-Z][A-Za-z0-9/&,-]*(?:\s+[A-Z]?[A-Za-z0-9/&,-]+){1,9})\s*\(([A-Z][A-Z0-9-]{1,11})\)")
# Captures definitions such as "ISRU - In Situ Resource Utilization".
_SHORT_FORM = re.compile(r"\b([A-Z][A-Z0-9-]{1,11})\s*[-–:]\s*([A-Z][A-Za-z0-9/&,-]*(?:\s+[A-Z]?[A-Za-z0-9/&,-]+){1,9})")


@dataclass(frozen=True)
class GlossaryEntry:
    acronym: str
    expansion: str
    page: int
    quote: str


def extract_glossary(pages: list[PageText]) -> list[GlossaryEntry]:
    found: dict[str, GlossaryEntry] = {}
    for page in pages:
        for line in page.text.splitlines():
            quote = line.strip()
            for match in _LONG_FORM.finditer(quote):
                expansion, acronym = match.group(1).strip(), match.group(2).strip()
                if 2 <= len(acronym) <= 12 and len(expansion.split()) >= 2:
                    found.setdefault(acronym, GlossaryEntry(acronym, expansion, page.page_number, quote))
            for match in _SHORT_FORM.finditer(quote):
                acronym, expansion = match.group(1).strip(), match.group(2).strip()
                if 2 <= len(acronym) <= 12 and len(expansion.split()) >= 2:
                    found.setdefault(acronym, GlossaryEntry(acronym, expansion, page.page_number, quote))
    return list(found.values())


def tag_chunk_acronyms(chunks: list[ChunkRecord], glossary: list[GlossaryEntry]) -> None:
    if not glossary:
        return
    for chunk in chunks:
        combined = f" {chunk.content} ".upper()
        matches = [entry for entry in glossary if re.search(
            rf"(?<![A-Z0-9]){re.escape(entry.acronym)}(?![A-Z0-9])", combined
        )]
        if matches:
            expansions = {entry.acronym: entry.expansion for entry in matches}
            chunk.metadata["acronym_expansions"] = expansions
            chunk.embedding_text += "\nAcronym expansions: " + "; ".join(
                f"{key} = {value}" for key, value in sorted(expansions.items())
            )
