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
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == expected.answer
    assert response.json()["verified"] is True
    assert captured["question"] == "What is the food safety risk?"
    assert captured["top_k"] == 7
    assert captured["use_reranker"] is True
    assert captured["model_tier"] == "judge"


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


def test_ask_endpoint_validates_top_k():
    response = client.post(
        "/ask",
        json={"question": "Question?", "top_k": 100},
    )
    assert response.status_code == 422


def test_ask_endpoint_rejects_blank_question(monkeypatch):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
