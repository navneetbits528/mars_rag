import pytest

from marsrag.llm import (
    DemoUsageTracker,
    RequestBudgetExceeded,
    UsageTotals,
    request_api_budget,
)


def test_request_budget_allows_calls_up_to_limit(monkeypatch):
    tracker = DemoUsageTracker()
    tracker.totals = UsageTotals()
    monkeypatch.setattr(tracker.settings, "openai_max_requests", 100)

    with request_api_budget(2) as state:
        tracker.reserve_request()
        tracker.reserve_request()
        assert state["used"] == 2
        with pytest.raises(RequestBudgetExceeded, match="QA request API budget exhausted"):
            tracker.reserve_request()

    assert tracker.totals.api_requests == 2


def test_request_budgets_are_isolated_between_requests(monkeypatch):
    tracker = DemoUsageTracker()
    tracker.totals = UsageTotals()
    monkeypatch.setattr(tracker.settings, "openai_max_requests", 100)

    with request_api_budget(1):
        tracker.reserve_request()
        with pytest.raises(RequestBudgetExceeded):
            tracker.reserve_request()

    with request_api_budget(1):
        tracker.reserve_request()

    assert tracker.totals.api_requests == 2


def test_global_budget_still_applies(monkeypatch):
    tracker = DemoUsageTracker()
    tracker.totals = UsageTotals()
    monkeypatch.setattr(tracker.settings, "openai_max_requests", 1)

    with request_api_budget(5):
        tracker.reserve_request()
        with pytest.raises(RequestBudgetExceeded, match="Demo API request budget exhausted"):
            tracker.reserve_request()
