from unittest.mock import patch

import pytest

from marsrag.filter_resolution import resolve_filters


@patch("marsrag.filter_resolution.fetch_all")
def test_resolves_exact_report_and_section_from_corpus(mock_fetch):
    mock_fetch.side_effect = [
        [{"report_no": "NASA-HRP-EVIDENCE"}, {"report_no": "NASA-M2M-ARCH"}],
        [{"section_path": "II"}, {"section_path": "II.1"}],
    ]
    result = resolve_filters("Summarize NASA-M2M-ARCH section II.1")
    assert result.report_no == "NASA-M2M-ARCH"
    assert result.section_path == "II.1"


@patch("marsrag.filter_resolution.fetch_all")
def test_defaults_to_baseline_when_no_report_is_named(mock_fetch):
    mock_fetch.side_effect = [
        [{"report_no": "NASA-HRP-EVIDENCE"}, {"report_no": "NASA-M2M-ARCH"}],
        [{"section_path": "II"}],
    ]
    result = resolve_filters("What risks are discussed?")
    assert result.report_no == "NASA-HRP-EVIDENCE"


@patch("marsrag.filter_resolution.fetch_all")
def test_explicit_api_filters_override_inferred_filters(mock_fetch):
    mock_fetch.side_effect = [
        [{"report_no": "NASA-HRP-EVIDENCE"}, {"report_no": "NASA-M2M-ARCH"}],
        [{"section_path": "III"}],
    ]
    result = resolve_filters(
        "NASA-M2M-ARCH section II",
        report_no="NASA-HRP-EVIDENCE",
        section_path="III",
        chunk_type="text",
        inferred_chunk_type="table",
    )
    assert result.report_no == "NASA-HRP-EVIDENCE"
    assert result.section_path == "III"
    assert result.chunk_type == "text"


@patch("marsrag.filter_resolution.fetch_all")
def test_rejects_unknown_report(mock_fetch):
    mock_fetch.return_value = [{"report_no": "NASA-HRP-EVIDENCE"}]
    with pytest.raises(ValueError, match="Unknown or non-current report_no"):
        resolve_filters("Question", report_no="NASA-FAKE")


@patch("marsrag.filter_resolution.fetch_all")
def test_rejects_unknown_section_for_selected_report(mock_fetch):
    mock_fetch.side_effect = [
        [{"report_no": "NASA-HRP-EVIDENCE"}],
        [{"section_path": "II"}],
    ]
    with pytest.raises(ValueError, match="Unknown section_path"):
        resolve_filters("Question", section_path="FAKE")


@patch("marsrag.filter_resolution.fetch_all")
def test_ignores_section_that_is_not_an_exact_match(mock_fetch):
    mock_fetch.side_effect = [
        [{"report_no": "NASA-HRP-EVIDENCE"}],
        [{"section_path": "II"}],
    ]
    result = resolve_filters("Tell me about section II.1")
    assert result.section_path is None


@patch("marsrag.filter_resolution.fetch_all")
def test_rejects_unsupported_explicit_chunk_type(mock_fetch):
    mock_fetch.side_effect = [
        [{"report_no": "NASA-HRP-EVIDENCE"}],
        [{"section_path": "II"}],
    ]
    with pytest.raises(ValueError, match="Unsupported chunk_type"):
        resolve_filters("Question", chunk_type="image")
