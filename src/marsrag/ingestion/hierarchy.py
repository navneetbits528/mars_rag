
from __future__ import annotations

import hashlib

import tiktoken

from marsrag.ingestion.chunker import normalize_text, stable_chunk_id
from marsrag.ingestion.models import ChunkRecord, ParentRecord
from marsrag.settings import get_settings


def _encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _split_tokens(text: str, max_tokens: int, encoding) -> list[str]:
    tokens = encoding.encode(text)
    if not tokens:
        return []

    pieces = []
    for start in range(0, len(tokens), max_tokens):
        part = encoding.decode(tokens[start:start + max_tokens]).strip()
        if part:
            pieces.append(part)

    return pieces


def build_parent_child_records(
    chunks: list[ChunkRecord],
) -> tuple[list[ParentRecord], list[ChunkRecord]]:
    """Create deterministic parents and child chunks with parent offsets."""

    settings = get_settings()
    encoding = _encoding(settings.embed_model)

    groups: dict[tuple[str, str], list[tuple[int, ChunkRecord]]] = {}

    # Keep the original input position to preserve source order within a page.
    for input_position, chunk in enumerate(chunks):
        key = (chunk.report_no, chunk.section_path or "Unsectioned")
        groups.setdefault(key, []).append((input_position, chunk))

    parents: list[ParentRecord] = []
    children: list[ChunkRecord] = []

    for (report_no, section_path), entries in groups.items():
        entries.sort(key=lambda item: (item[1].page, item[0]))
        section_chunks = [chunk for _, chunk in entries]

        parent_parts = []
        chunk_positions = {}
        parent_cursor = 0

        for index, chunk in enumerate(section_chunks):
            if index:
                parent_cursor += 2  # Length of the "\n\n" separator.

            chunk_positions[id(chunk)] = parent_cursor
            parent_parts.append(chunk.content)
            parent_cursor += len(chunk.content)

        parent_content = "\n\n".join(parent_parts)
        normalized_parent = normalize_text(parent_content)

        parent_id = hashlib.sha256(
            f"{report_no}|{section_path}|{normalized_parent.casefold()}".encode("utf-8")
        ).hexdigest()

        parents.append(
            ParentRecord(
                parent_id=parent_id,
                report_no=report_no,
                page_start=min(item.page for item in section_chunks),
                page_end=max(item.page for item in section_chunks),
                section_title=section_chunks[0].section_title,
                section_path=section_path,
                content=parent_content,
                metadata={"child_count": len(section_chunks)},
            )
        )

        for chunk in section_chunks:
            parts = (
                [chunk.content]
                if chunk.chunk_type in {"table", "figure_caption"}
                else _split_tokens(
                    chunk.content,
                    settings.child_chunk_tokens,
                    encoding,
                )
            )

            local_cursor = 0
            chunk_parent_start = chunk_positions[id(chunk)]

            for part in parts:
                normalized = normalize_text(part)
                if not normalized:
                    continue

                # Find this part in the original chunk, respecting split order.
                local_start = chunk.content.find(part, local_cursor)

                if local_start >= 0:
                    parent_start = chunk_parent_start + local_start
                    parent_end = parent_start + len(part)
                    local_cursor = local_start + len(part)
                else:
                    # Do not invent an offset if exact matching fails.
                    parent_start = None
                    parent_end = None

                child_id = stable_chunk_id(
                    report_no,
                    chunk.page,
                    f"{chunk.chunk_type}:{normalized}",
                )

                metadata = {
                    **chunk.metadata,
                    "parent_id": parent_id,
                    "ocr": bool(chunk.metadata.get("ocr")),
                    "parent_start_char": parent_start,
                    "parent_end_char": parent_end,
                }

                children.append(
                    ChunkRecord(
                        chunk_id=child_id,
                        report_no=chunk.report_no,
                        page=chunk.page,
                        title=chunk.title,
                        year=chunk.year,
                        status=chunk.status,
                        url=chunk.url,
                        section_title=chunk.section_title,
                        section_path=chunk.section_path,
                        chunk_type=chunk.chunk_type,
                        content=part,
                        embedding_text=(
                            f"Section: {chunk.section_title}\n"
                            f"Type: {chunk.chunk_type}\n{part}"
                        ),
                        content_sha256=hashlib.sha256(
                            normalized.encode()
                        ).hexdigest(),
                        metadata=metadata,
                        parent_id=parent_id,
                    )
                )

    unique_parents = {item.parent_id: item for item in parents}
    unique_children = {item.chunk_id: item for item in children}

    return list(unique_parents.values()), list(unique_children.values())
