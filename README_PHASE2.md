# Phase 2 — PDF ingestion v1

Includes page-wise PyMuPDF extraction, repeated margin/header/footer removal (>40% pages),
page-number and TOC leader cleanup, bibliography cutoff, heading-aware chunking, recursive
splitting (~1000 characters, 15% overlap), deterministic chunk IDs, SHA-256 incremental
ingestion, token-counted embedding batches, preflight cost/request-budget estimates and pgvector
persistence with stale-chunk cleanup.

## Setup
```powershell
uv sync --extra dev
docker compose up -d db
uv run python -m marsrag.db migrate
```
This is an overlay/patch for your already-working Phase 1 repository, not a replacement for its
existing DB/API/SQL files. Copy these files over the same paths in your Phase 1 folder. Keep your
existing `db.py`, SQL migration, Docker files, API files and `manifest.json`. Place actual NASA PDFs
under `data/pdfs/` and update your existing manifest. PDFs are not bundled.

## Preflight (no OpenAI calls or DB writes)
```powershell
uv run python -m marsrag.ingestion.cli --estimate-only
```

## Ingest
```powershell
uv run python -m marsrag.ingestion.cli --migrate
```
Single report:
```powershell
uv run python -m marsrag.ingestion.cli --report-no NASA/SP-2009-566
```
Force reprocess:
```powershell
uv run python -m marsrag.ingestion.cli --force
```

Verify:
```powershell
docker exec -it mars-rag-db psql -U marsrag -d marsrag
```
```sql
SELECT report_no,title,year,status FROM documents;
SELECT report_no,COUNT(*) chunks,MIN(page) first_page,MAX(page) last_page FROM chunks GROUP BY report_no;
SELECT chunk_id,report_no,page,section_title,LEFT(content,120) FROM chunks ORDER BY report_no,page LIMIT 10;
```

Tests:
```powershell
uv run pytest -q
```

## Limitations
OCR, table extraction, figure captions, parent-child storage, acronym glossary and fact extraction
are Phase 3. Heading detection is conservative pattern matching, not PDF font/layout analysis.
Bibliography cleanup stops at recognized headings; inspect reports with appendices. Embedding token
usage is estimated locally. The request counter is process-local and resets for each CLI run. With the 10-call demo budget,
start with one reasonably sized PDF; `--estimate-only` shows whether the selected corpus fits.
VERIFY LangChain/OpenAI embedding signatures and provider batch limits against current docs on
dependency upgrades.
