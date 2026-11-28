# NASA Mars Missions — Agentic RAG Application

An end-to-end Retrieval-Augmented Generation (RAG) application for exploring NASA human-spaceflight and Mars-mission documents. The system ingests PDF reports, retrieves relevant evidence, generates answers, and verifies citations and factual claims before presenting document-grounded responses.

It includes a FastAPI backend, PostgreSQL with pgvector, a LangGraph question-answering workflow, server-sent event (SSE) streaming, and a Streamlit chat interface.

> **Evidence policy:** A provisional streamed draft is not a verified answer. For document-grounded responses, the final answer is shown as verified only after citation and claim-level checks pass. Conversational responses are explicitly marked as not verified against NASA documents. Out-of-scope questions receive a scope explanation rather than fabricated citations.

## Features

### Document ingestion
- Page-level PDF text extraction with PyMuPDF.
- OCR fallback for pages with insufficient extractable text.
- Table extraction, figure-caption handling, and document structure metadata.
- Parent-child chunking, acronym/glossary support, and optional structured fact extraction.
- Deterministic chunk identifiers and incremental ingestion using document checksums.
- Batched embeddings and ingestion preflight estimates.

### Retrieval
- Dense vector retrieval using PostgreSQL/pgvector.
- Keyword retrieval and hybrid retrieval.
- Reciprocal Rank Fusion (RRF) to combine ranked results.
- Optional LLM-based reranking.
- Report, section, and content-type filters.
- Parent-context expansion and multi-query retrieval support.

### Question answering
- Query understanding for simple, multi-hop, comparison, clarification, conversational, and out-of-scope questions.
- Follow-up question condensation using conversation history.
- LangGraph orchestration with fast and multi-step retrieval paths.
- Structured answer generation.
- Exact citation ID and quote validation.
- Claim-level evidence verification and safe abstention when evidence is insufficient.
- Separate conversational response path, clearly labelled as not document-verified.

### Performance and controls
- Semantic answer cache with configurable similarity threshold and TTL.
- Per-request API-call budget and global cost controls.
- Configurable fast, main, deep, and judge model tiers.
- Streaming draft tokens and final answer events over SSE.

### Evaluation
- Curated, labelled retrieval evaluation dataset.
- Recall@K, MRR, answer matching, citation validity, faithfulness proxy, and abstention metrics.
- Retrieval ablation runner comparing dense, keyword, hybrid, reranking, and query-expansion configurations.
- Semantic-cache threshold and live PostgreSQL cache-latency benchmarks.
- Consolidated JSON and CSV evaluation report.

## Architecture

```text
                         +----------------------+
                         |   Streamlit Chat UI  |
                         +----------+-----------+
                                    |
                              HTTP / SSE
                                    |
                         +----------v-----------+
                         |      FastAPI API     |
                         |   /ask, /ask/stream  |
                         +----------+-----------+
                                    |
                         +----------v-----------+
                         |   LangGraph QA Flow  |
                         +----------+-----------+
                                    |
              +---------------------+----------------------+
              |                     |                      |
      Query understanding    Filter resolution      Cache lookup
              |                     |                      |
              +---------------------+----------------------+
                                    |
                         +----------v-----------+
                         |   Retrieval Layer    |
                         | Dense + Keyword + RRF|
                         | Optional reranking   |
                         +----------+-----------+
                                    |
                         +----------v-----------+
                         | Answer generation    |
                         | Citation validation  |
                         | Claim verification   |
                         +----------+-----------+
                                    |
                         +----------v-----------+
                         | Verified response /  |
                         | safe abstention      |
                         +----------------------+

  PDF reports --> extraction / OCR --> chunking --> embeddings
                                                   |
                                          PostgreSQL + pgvector
```

## Technology stack

- **Language:** Python 3.11+
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit
- **Orchestration:** LangGraph, LangChain
- **LLM and embeddings:** OpenAI API (configurable model names)
- **Database:** PostgreSQL 16, pgvector, psycopg
- **PDF processing:** PyMuPDF, pdfplumber, pytesseract, Pillow
- **Validation:** Pydantic
- **Testing:** pytest, pytest-asyncio, httpx
- **Environment and packaging:** uv
- **Containerization:** Docker, Docker Compose

## Prerequisites

Install the following before starting:

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/).
- Docker Desktop with Docker Compose.
- An OpenAI API key with access to the configured chat and embedding models.
- Tesseract OCR installed and available on PATH if OCR fallback is required.

## Quick start (Windows PowerShell)

Run commands from the repository root.

### 1. Install dependencies

Install the application, development, and Streamlit dependencies:

```powershell
uv sync --all-extras
```

### 2. Configure environment variables

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set your API key and model names:

```env
OPENAI_API_KEY=your_openai_api_key
FAST_MODEL=gpt-4o-mini
MAIN_MODEL=gpt-4o-mini
DEEP_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
DATABASE_URL=postgresql://marsrag:marsrag_dev_password@localhost:5432/marsrag
```

The example uses the same chat model for all three answer tiers. You can change the model names to models available to your API account. Keep `.env` private and never commit API keys.

### 3. Start PostgreSQL with pgvector

```powershell
docker compose up -d db
docker compose ps
```

Wait until the database service is healthy. You can check readiness with:

```powershell
docker compose exec db pg_isready -U marsrag -d marsrag
```

### 4. Apply database migrations

```powershell
uv run python -m marsrag.db migrate
```

### 5. Add NASA PDF files and update the manifest

Place the PDF reports under `data/pdfs/` and update `manifest.json` to reference the files and their report metadata. The PDF files are not bundled with this repository.

Estimate ingestion before making embedding API calls:

```powershell
uv run python -m marsrag.ingestion.cli --estimate-only
```

Ingest the reports listed in the manifest:

```powershell
uv run python -m marsrag.ingestion.cli
```

To ingest one report:

```powershell
uv run python -m marsrag.ingestion.cli --report-no NASA-HRP-EVIDENCE
```

To force reprocessing:

```powershell
uv run python -m marsrag.ingestion.cli --report-no NASA-HRP-EVIDENCE --force
```

To run migrations and ingestion in one command:

```powershell
uv run python -m marsrag.ingestion.cli --migrate
```

### 6. Start the FastAPI backend

In terminal 1:

```powershell
uv run uvicorn marsrag.api.main:app --reload
```

API documentation is available at:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health endpoint: http://127.0.0.1:8000/health

### 7. Start the Streamlit frontend

In terminal 2:

```powershell
uv run streamlit run streamlit_app.py
```

Open http://localhost:8501.

The Streamlit UI connects to `http://127.0.0.1:8000` by default. To use another API host, set `MARS_RAG_API_URL` before starting Streamlit:

```powershell
$env:MARS_RAG_API_URL = "http://127.0.0.1:8000"
uv run streamlit run streamlit_app.py
```

## Docker Compose

The Compose file defines PostgreSQL and an API service. After configuring `.env` and ensuring the database schema and document corpus are prepared, you can start the services with:

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

The Streamlit UI is run separately on the host using the command in the previous section.

Stop the Compose services:

```powershell
docker compose down
```

To also remove the persisted PostgreSQL volume (this permanently deletes the local database data):

```powershell
docker compose down -v
```

## API usage

### Synchronous question

`POST /ask`

Example request:

```json
{
  "question": "What are the major risks associated with human spaceflight?",
  "report_no": "NASA-HRP-EVIDENCE",
  "top_k": 5,
  "use_reranker": false,
  "model_tier": "main"
}
```

### Streaming question

`POST /ask/stream`

Accepts the same request fields and returns server-sent events. The stream may include status updates, provisional token events, a final structured answer event, and a completion event. Only the final answer's verification status and citations are authoritative.

### Request fields

| Field | Description |
|---|---|
| `question` | User question (required) |
| `conversation_history` | Up to 10 prior user/assistant messages |
| `report_no` | Optional report filter |
| `top_k` | Number of retrieved results (1–50; default 5) |
| `section_path` | Optional section filter |
| `chunk_type` | Optional content type filter |
| `use_reranker` | Enable optional reranking (default false) |
| `model_tier` | `fast`, `main`, `deep`, or `judge` |

### Response interpretation

- `verified=true`: the document-grounded answer passed citation and claim-level checks.
- `abstained=true`: the system could not safely provide a document-grounded answer.
- `verified=false` with `abstained=false`: a conversational answer was returned and is not verified against NASA documents.
- `reason`: explains abstention or identifies a non-document-verified response.

The health endpoint checks database connectivity and reports the corpus version.

## Evaluation

The checked-in evaluation workflow is scoped to the NASA-HRP-EVIDENCE corpus. Evaluation datasets and outputs are stored under `eval/`.

Validate the curated dataset:

```powershell
uv run python -m marsrag.evaluation.dataset validate --input eval/dataset.json
```

Run retrieval ablations:

```powershell
uv run python -m marsrag.evaluation.ablation --input eval/dataset.json --output eval/ablation_results.json
```

Evaluate semantic-cache thresholds:

```powershell
uv run python -m marsrag.evaluation.cache --input eval/cache_dataset.json --output eval/cache_results.json
```

Benchmark live PostgreSQL cache lookup:

```powershell
uv run python -m marsrag.evaluation.cache_benchmark --repeats 30 --threshold 0.93 --output eval/cache_benchmark_results.json
```

Build the consolidated evaluation report:

```powershell
uv run python -m marsrag.evaluation.report --ablation eval/ablation_results.json --cache-thresholds eval/cache_results.json --cache-benchmark eval/cache_benchmark_results.json --json-output eval/final_evaluation_report.json --csv-output eval/final_evaluation_report.csv
```

See the evaluation dataset and its annotation requirements in `eval/`. Evaluation results depend on the selected corpus, dataset labels, model configuration, and runtime environment. Review the generated reports before publishing benchmark claims.

## Tests and code quality

Run the full test suite:

```powershell
uv run pytest -v
```

Run lint checks:

```powershell
uv run ruff check .
```

## Configuration reference

Important environment variables are documented in `.env.example`. Common settings include:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | API credential |
| `FAST_MODEL`, `MAIN_MODEL`, `DEEP_MODEL` | Answer-generation model tiers |
| `JUDGE_MODEL` | Claim-verification/reranking model |
| `EMBED_MODEL`, `EMBED_DIM` | Embedding model and vector dimension |
| `DATABASE_URL` | PostgreSQL connection |
| `OPENAI_MAX_REQUESTS` | Global API request budget |
| `DAILY_COST_LIMIT_USD` | Configured daily cost limit |
| `MAX_RETRIES`, `REQUEST_TIMEOUT_SECONDS` | API retry and timeout controls |
| `SEMANTIC_CACHE_ENABLED` | Enable or disable semantic cache |
| `SEMANTIC_CACHE_TTL_SECONDS` | Cache entry lifetime |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | Minimum similarity for cache reuse |
| `MARS_RAG_API_URL` | Streamlit backend URL |

Check `.env.example` and the settings module for the complete set of supported options and defaults.

## Repository layout

```text
.
├── data/
│   └── pdfs/                 # Add source NASA PDF reports here
├── eval/                     # Evaluation datasets, templates, and reports
├── sql/                      # Database schema and migrations
├── src/marsrag/
│   ├── api/                  # FastAPI application
│   ├── evaluation/           # Metrics, benchmarks, and reports
│   ├── ingestion/            # PDF extraction and ingestion pipeline
│   ├── retrieval/            # Dense, keyword, hybrid retrieval and reranking
│   ├── generation.py         # Answer generation and verification
│   ├── query_understanding.py
│   ├── qa.py                 # LangGraph QA workflow
│   ├── semantic_cache.py
│   └── settings.py
├── tests/                    # Automated tests
├── streamlit_app.py          # Streamlit chat UI
├── manifest.json             # Source document manifest
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## Security and operational notes

- Never commit `.env`, API keys, database credentials, or private source documents.
- The Compose database password is a development default. Replace it before shared or production use.
- Configure a strong `ADMIN_API_KEY` where administrative functionality is enabled.
- Restrict network exposure and configure authentication, authorization, TLS, secret management, backups, and monitoring before production deployment.
- OCR requires a working Tesseract installation and the requested language data.
- LLM responses and retrieval quality depend on the ingested corpus and model configuration.
- Semantic-cache thresholds are experimental settings and should be validated against representative, human-reviewed question pairs before production use.
- Conversational responses are not document-verified and should not be presented as cited NASA research answers.

## Current scope and limitations

This repository is an evaluated project application and development/demo setup. It is not a NASA product or an official NASA service. It does not provide personal medical, travel, astronaut-selection, or mission eligibility decisions. The knowledge available to the application is limited to the documents ingested into its database, except for the separately labelled conversational response path.

## License

Add the license and attribution terms applicable to your source documents and project before public distribution.
