from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("MARS_RAG_API_URL", "http://127.0.0.1:8000")


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for index, citation in enumerate(citations, start=1):
            st.markdown(
                f"**{index}. {citation.get('report_no', 'Unknown report')} "
                f"— page {citation.get('page', 'N/A')}**"
            )
            st.caption(f"Chunk: {citation.get('chunk_id', 'N/A')}")
            if citation.get("quote"):
                st.markdown(f"> {citation['quote']}")
            if citation.get("url"):
                st.markdown(f"[Open source]({citation['url']})")


def _stream_question(
    api_url: str,
    payload: dict[str, Any],
    answer_placeholder: Any,
    status_placeholder: Any,
) -> dict[str, Any]:
    draft = ""
    final_answer: dict[str, Any] | None = None
    current_event = "message"
    data_lines: list[str] = []

    with requests.post(
        f"{api_url.rstrip('/')}/ask/stream",
        json=payload,
        stream=True,
        timeout=(10, 300),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line or ""
            if not line:
                if data_lines:
                    data = json.loads("\n".join(data_lines))
                    if current_event == "status":
                        stage = data.get("stage", "working")
                        status_placeholder.update(
                            label=f"Working: {stage.replace('_', ' ')}",
                            state="running",
                        )
                    elif current_event == "token":
                        draft += data.get("text", "")
                        answer_placeholder.markdown(draft + "▌")
                    elif current_event == "answer":
                        final_answer = data
                    elif current_event == "error":
                        raise RuntimeError(data.get("detail", "Streaming request failed."))
                current_event = "message"
                data_lines = []
                continue

            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())

    if final_answer is None:
        raise RuntimeError("The stream ended without a final answer event.")
    status_placeholder.update(label="Response received", state="complete")
    return final_answer


def main() -> None:
    st.set_page_config(page_title="Mars Missions RAG", page_icon="🚀", layout="wide")
    st.title("🚀 Mars Missions RAG")
    st.caption("Ask questions about NASA human-spaceflight and Mars mission documents.")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL)
        st.divider()
        st.header("Retrieval settings")
        report_no = st.text_input("Report number", value="NASA-HRP-EVIDENCE")
        top_k = st.slider("Evidence chunks (top_k)", min_value=1, max_value=20, value=5)
        use_reranker = st.checkbox("Use reranker", value=False)
        model_tier = st.selectbox("Generation model tier", ["fast", "main", "deep"], index=1)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("abstained"):
                st.warning(message.get("reason") or "This response could not be verified against the document evidence.")
            elif message.get("verified"):
                st.success("Verified against retrieved document evidence.")
            elif message.get("reason"):
                st.info(message["reason"])
            _render_citations(message.get("citations", []))

    question = st.chat_input("Ask a question about the documents...")
    if not question:
        return

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
        if item["role"] in {"user", "assistant"}
    ][-10:]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Connecting to Mars RAG API...", expanded=False)
        draft_placeholder = st.empty()
        try:
            result = _stream_question(
                api_url,
                {
                    "question": question,
                    "conversation_history": history,
                    "report_no": report_no.strip() or None,
                    "top_k": top_k,
                    "use_reranker": use_reranker,
                    "model_tier": model_tier,
                },
                draft_placeholder,
                status,
            )
            draft_placeholder.empty()
            st.markdown(result.get("answer", "No answer returned."))
            if result.get("abstained"):
                st.warning(result.get("reason") or "This response could not be verified against the document evidence.")
            elif result.get("verified"):
                st.success("Verified against retrieved document evidence.")
            elif result.get("reason"):
                st.info(result["reason"])
            _render_citations(result.get("citations", []))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "verified": result.get("verified", False),
                    "abstained": result.get("abstained", False),
                    "citations": result.get("citations", []),
                    "reason": result.get("reason"),
                }
            )
        except requests.RequestException as exc:
            status.update(label="API connection failed", state="error")
            st.error(f"Could not connect to the FastAPI service: {exc}")
        except (ValueError, RuntimeError) as exc:
            status.update(label="Streaming failed", state="error")
            st.error(str(exc))


if __name__ == "__main__":
    main()
