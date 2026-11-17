from dataclasses import dataclass, field


@dataclass
class PageText:
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    report_no: str
    page: int
    title: str
    year: int
    status: str
    url: str
    section_title: str
    section_path: str
    chunk_type: str
    content: str
    embedding_text: str
    content_sha256: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


@dataclass
class ParentRecord:
    parent_id: str
    report_no: str
    page_start: int
    page_end: int
    section_title: str
    section_path: str
    content: str
    metadata: dict = field(default_factory=dict)
