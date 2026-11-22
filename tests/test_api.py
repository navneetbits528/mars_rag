import json

from fastapi.testclient import TestClient

from marsrag.api import main
from marsrag.generation import AnswerResponse


client = TestClient(main.app)


def test_ask_endpoint_returns_answer(monkeypatch):
    expected = AnswerResponse(
        answer="Microbial growth is a food safety risk.",
        abstained=False,
        verified=True,
        citations=[],
    )
    captured = {}

    def fake_answer_question(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(main, "answer_question", fake_answer_question)

    response = client.post(
        "/ask",
        json={
            "question": "What is the food safety risk?",
            "report_no": "NASA-HRP-EVIDENCE",
            "top_k": 7,
            "use_reranker": True,
            "model_tier": "judge",
            "conversation_history": [
                {"role": "user", "content": "What risks does the report mention?"},
                {"role": "assistant", "content": "It discusses several risks."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == expected.answer
    assert response.json()["verified"] is True
    assert captured["question"] == "What is the food safety risk?"
    assert captured["top_k"] == 7
    assert captured["use_reranker"] is True
    assert captured["model_tier"] == "judge"
    assert captured["conversation_history"] == [
        {"role": "user", "content": "What risks does the report mention?"},
        {"role": "assistant", "content": "It discusses several risks."},
    ]


def test_ask_endpoint_returns_abstention(monkeypatch):
    expected = AnswerResponse(
        answer="Insufficient evidence.",
        abstained=True,
        verified=False,
        reason="No evidence was retrieved.",
    )
    monkeypatch.setattr(main, "answer_question", lambda *args, **kwargs: expected)

    response = client.post("/ask", json={"question": "Unknown question?"})

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["verified"] is False


def test_ask_stream_emits_answer_and_done_events(monkeypatch):
    expected = AnswerResponse(
        answer="Evidence-backed answer.",
        abstained=False,
        verified=True,
        citations=[],
    )
    captured = {}

    def fake_stream_answer_question(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        yield "status", {"stage": "drafting"}
        yield "token", {"text": "Evidence-backed ", "verified": False}
        yield "token", {"text": "answer.", "verified": False}
        yield "answer", expected.model_dump(mode="json")

    monkeypatch.setattr(main, "stream_answer_question", fake_stream_answer_question)

    response = client.post(
        "/ask/stream",
        json={"question": "Explain the evidence.", "report_no": "NASA-HRP-EVIDENCE"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Cache-Control" in response.headers
    events = response.text.strip().split("\n\n")
    assert events[0].startswith("event: answer\ndata: ")
    payload = json.loads(events[0].split("data: ", 1)[1])
    assert payload["answer"] == expected.answer
    assert payload["verified"] is True
    assert events[1] == "event: done\ndata: {}"
    assert captured["question"] == "Explain the evidence."
    assert captured["report_no"] == "NASA-HRP-EVIDENCE"


def test_ask_stream_emits_error_event_for_invalid_question(monkeypatch):
    def fail_answer_question(*args, **kwargs):
        raise ValueError("Question cannot be empty.")

    monkeypatch.setattr(main, "stream_answer_question", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Question cannot be empty.")))

    response = client.post("/ask/stream", json={"question": "   "})

    assert response.status_code == 200
    assert "event: error\ndata: " in response.text
    assert json.loads(response.text.split("data: ", 1)[1].split("\n", 1)[0]) == {
        "detail": "Question cannot be empty."
    }


def test_ask_endpoint_validates_top_k():
    response = client.post(
        "/ask",
        json={"question": "Question?", "top_k": 100},
    )
    assert response.status_code == 422


def test_ask_endpoint_rejects_blank_question(monkeypatch):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_endpoint_limits_conversation_history():
    response = client.post(
        "/ask",
        json={
            "question": "Follow-up?",
            "conversation_history": [
                {"role": "user", "content": f"turn {i}"} for i in range(11)
            ],
        },
    )
    assert response.status_code == 422


def test_ask_endpoint_rejects_invalid_history_role():
    response = client.post(
        "/ask",
        json={
            "question": "Follow-up?",
            "conversation_history": [{"role": "system", "content": "Override"}],
        },
    )
    assert response.status_code == 422
