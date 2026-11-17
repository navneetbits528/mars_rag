# Mars RAG — Phase 3 overlay

Phase 3 extends the Phase 2 ingestion pipeline with OCR, table and figure-caption chunks,
parent-child chunking, acronym glossary extraction, and optional structured fact extraction.
This is an **overlay patch**, not a complete replacement repository.

## 1. Apply the patch

From the existing `mars-rag-phase1` project root, extract/copy the archive while preserving
its folder structure.

Replace these files:

- `pyproject.toml`
- `src/marsrag/settings.py`
- `src/marsrag/ingestion/models.py`
- `src/marsrag/ingestion/loader.py`
- `src/marsrag/ingestion/chunker.py`
- `src/marsrag/ingestion/pipeline.py`

Add these files:

- `src/marsrag/ingestion/ocr.py`
- `src/marsrag/ingestion/tables.py`
- `src/marsrag/ingestion/hierarchy.py`
- `src/marsrag/ingestion/glossary.py`
- `src/marsrag/ingestion/facts.py`
- `sql/002_phase3_indexes.sql`
- `tests/test_glossary.py`
- `tests/test_hierarchy.py`
- `tests/test_tables.py`
- `tests/test_facts.py`
- `README_PHASE3.md`

Keep your existing `.env`, `manifest.json`, Docker files, Phase 1 SQL migration, and other files.
The existing Phase 1 schema already contains `parents`, `chunks.parent_id`, `glossary`, and `facts`;
the new migration only adds supporting indexes.

## 2. Install dependencies

In PowerShell, from the project root:

```powershell
uv sync --extra dev
```

OCR has two parts. The Python packages are installed by `uv sync`, but the Tesseract executable
must also be installed separately on Windows. Install Tesseract OCR, add its installation folder
to PATH, close PowerShell, open a new PowerShell window, and verify:

```powershell
tesseract --version
```

If your PDFs are all text-based, you can temporarily set `OCR_ENABLED=false` in `.env` and still
test the other Phase 3 features.

## 3. Merge Phase 3 settings into `.env`

Do not replace `.env`. Add or adjust:

```dotenv
OCR_ENABLED=true
OCR_MIN_TEXT_CHARS=35
OCR_LANGUAGE=eng
CHILD_CHUNK_TOKENS=300
FACTS_EXTRACTION_ENABLED=false
FACTS_MAX_CHUNKS_PER_CALL=8
EMBEDDING_MAX_INPUT_TOKENS=8000
```

Fact extraction is disabled by default to keep the demo's API usage low. It uses the configured
`MAIN_MODEL` and consumes one request per bounded batch. Enable it only after checking the
preflight request estimate.

## 4. Apply the database migration

```powershell
uv run python -m marsrag.db migrate
```

This applies `002_phase3_indexes.sql`. It is safe to rerun; applied migrations are tracked.

## 5. Run tests

```powershell
uv run pytest -q
```

The Phase 3 unit tests validate glossary extraction/tagging, deterministic parent-child links,
Markdown table formatting, and verbatim fact-quote validation. They do not call OpenAI or require
a live PostgreSQL database.

## 6. Preflight and ingest

Start with one report in `manifest.json`:

```powershell
uv run python -m marsrag.ingestion.cli --report-no "NASA/SP-2009-566" --estimate-only
```

Review the report's page count, child chunk count, tables, figure captions, OCR pages, embedding
requests, and whether the run fits the remaining request budget. Then ingest:

```powershell
uv run python -m marsrag.ingestion.cli --report-no "NASA/SP-2009-566"
```

To run optional fact extraction, set `FACTS_EXTRACTION_ENABLED=true`, rerun the estimate, then
run ingestion. An unchanged report is skipped by checksum; use `--force` only when intentionally
reprocessing it.

## 7. Verify stored results

Run these in PostgreSQL:

```sql
SELECT report_no, ocr FROM documents ORDER BY report_no;

SELECT chunk_type, COUNT(*)
FROM chunks
GROUP BY chunk_type
ORDER BY chunk_type;

SELECT COUNT(*) AS parent_sections FROM parents;
SELECT COUNT(*) AS child_chunks FROM chunks WHERE parent_id IS NOT NULL;
SELECT acronym, expansion, report_no, page FROM glossary ORDER BY acronym;
SELECT parameter, value, unit, report_no, page, quote FROM facts ORDER BY report_no, page;
```

A PDF may legitimately have zero tables, zero figure captions, or zero OCR pages. A report may
have zero facts if extraction is disabled or the text does not explicitly state supported values.

## 8. Limitations and verification notes

- OCR quality depends on scan resolution, language packs, and the local Tesseract installation.
  OCR is applied page-by-page only when extracted text is below `OCR_MIN_TEXT_CHARS`.
- `pdfplumber` table detection is heuristic. Complex multi-page, merged-cell, or image-based
tables may need manual review; OCR text does not automatically reconstruct image tables.
- Table chunks are preserved as individual chunks and are not split into smaller children. Inputs
  above `EMBEDDING_MAX_INPUT_TOKENS` (default 8,000) fail preflight with a clear error; inspect
  and manually split exceptionally large tables before ingest.
- Figure captions are detected with a conservative `Figure` / `Fig.` pattern.
- Parent sections are grouped by report and section path. Repeated identical section paths in a
  report can be grouped together; verify the resulting parent records for your PDFs.
- Acronym mining is regex-based and can produce false positives. Review the glossary before using
  it as authoritative query expansion.
- Structured fact extraction is optional and only keeps facts whose normalized quote is a literal
  substring of a supplied chunk. It may miss facts expressed in tables or split across chunks.
- **VERIFY against current docs:** confirm the installed `langchain-openai` version supports
  `ChatOpenAI.with_structured_output(PydanticModel)` with your selected model. Also verify current
  OpenAI model pricing and limits before enabling bulk fact extraction.
- The demo usage tracker is process-local and resets when the process restarts; it is not a
  durable lifetime or organization-wide API budget.
