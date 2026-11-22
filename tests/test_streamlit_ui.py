from pathlib import Path


def test_streamlit_ui_uses_streaming_endpoint_and_final_verification():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert '"/ask/stream"' in source
    assert 'current_event == "token"' in source
    assert 'current_event == "answer"' in source
    assert 'result.get("verified")' in source
    assert 'result.get("citations", [])' in source
