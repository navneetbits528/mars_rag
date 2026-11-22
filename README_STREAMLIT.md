# Streamlit UI

The Streamlit frontend connects to the existing FastAPI service and consumes
server-sent events from `POST /ask/stream`.

## Install

From the project root, install the UI extra:

```powershell
uv sync --extra ui
```

If the project environment is managed with an existing lock file, this updates
the lock file as needed.

## Run the backend

In terminal 1:

```powershell
uv run uvicorn marsrag.api.main:app --reload
```

## Run the frontend

In terminal 2:

```powershell
uv run streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (normally
`http://localhost:8501`).

## Configuration

Set `MARS_RAG_API_URL` to point the UI at a different FastAPI service.
The default is `http://127.0.0.1:8000`.

The UI streams provisional draft text first. It then replaces that draft with
the final structured response from the API. Only the final response's
`verified` flag and citations are treated as authoritative. Provisional tokens
are explicitly unverified and are never stored as assistant conversation turns.
