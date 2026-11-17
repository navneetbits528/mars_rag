CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    report_no TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'superseded')),
    supersedes TEXT,
    file_path TEXT NOT NULL,
    file_sha256 TEXT,
    ocr BOOLEAN NOT NULL DEFAULT FALSE,
    corpus_version INTEGER NOT NULL DEFAULT 1,
    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,
    report_no TEXT NOT NULL REFERENCES documents(report_no) ON DELETE CASCADE,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section_title TEXT,
    section_path TEXT,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    report_no TEXT NOT NULL REFERENCES documents(report_no) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'superseded')),
    url TEXT NOT NULL,
    section_title TEXT,
    section_path TEXT,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    parent_id TEXT REFERENCES parents(parent_id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    embedding VECTOR({{EMBED_DIM}}) NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_sha256 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin_idx ON chunks USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS chunks_report_page_idx ON chunks(report_no, page);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON chunks USING gin(metadata);

CREATE TABLE IF NOT EXISTS glossary (
    acronym TEXT PRIMARY KEY,
    expansion TEXT NOT NULL,
    report_no TEXT REFERENCES documents(report_no) ON DELETE CASCADE,
    page INTEGER,
    quote TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    parameter TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT,
    context TEXT,
    report_no TEXT NOT NULL REFERENCES documents(report_no) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    source_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    quote TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS facts_parameter_idx ON facts(parameter);

CREATE TABLE IF NOT EXISTS semantic_cache (
    cache_id BIGSERIAL PRIMARY KEY,
    cache_key TEXT NOT NULL,
    question TEXT NOT NULL,
    question_embedding VECTOR({{EMBED_DIM}}) NOT NULL,
    answer JSONB NOT NULL,
    sources JSONB NOT NULL,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS semantic_cache_embedding_hnsw_idx
    ON semantic_cache USING hnsw (question_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS semantic_cache_expiry_idx ON semantic_cache(expires_at);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    question TEXT NOT NULL,
    answer JSONB,
    rating SMALLINT CHECK (rating IN (-1, 1)),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_log (
    request_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    intent TEXT,
    retrieved_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    abstained BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage (
    usage_id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    request_type TEXT NOT NULL CHECK (request_type IN ('chat', 'embedding')),
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    embedding_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC(14, 8) NOT NULL DEFAULT 0,
    api_requests INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingest_files (
    file_path TEXT PRIMARY KEY,
    report_no TEXT NOT NULL REFERENCES documents(report_no) ON DELETE CASCADE,
    file_sha256 TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
