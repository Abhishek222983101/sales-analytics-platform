"""Tests for the InsightJSON, evidence-cited recommendations, and LLM fallbacks.

These run OFFLINE — the LLM paths are asserted to return None without a key, so no
network call happens in CI. (A real Groq key only lights up the summary/chat.)
"""
import pytest

from core import llm, nlp
from core.diagnostic import (
    compute_rfm,
    detect_anomalies,
    segment_customers,
    segment_summary,
)
from core.forecasting import run_forecast
from core.ingestion import load_demo
from core.insight_bus import build_insight
from core.prescriptive import executive_summary, recommend

_RANK = {"High": 3, "Medium": 2, "Low": 1}


@pytest.fixture(scope="module")
def full():
    data = load_demo().canonical_df
    fc = run_forecast(data)
    seg = segment_summary(segment_customers(compute_rfm(data)))
    an = detect_anomalies(data).anomalies
    return data, build_insight(data, forecast=fc, segments=seg, anomalies=an)


def test_build_insight_is_complete_and_serialisable(full):
    _, ins = full
    assert ins.total_revenue > 0
    assert ins.concentration is not None
    assert ins.segments is not None and 0 <= ins.segments.at_risk_revenue_share <= 1
    assert ins.forecast is not None
    assert ins.model_dump_json()          # pydantic round-trips


def test_recommendations_cite_evidence_and_are_ranked(full):
    _, ins = full
    actions = recommend(ins)
    assert actions
    ranks = [_RANK[a.impact] for a in actions]
    assert ranks == sorted(ranks, reverse=True)          # sorted by impact
    for a in actions:
        assert a.title and a.why and a.evidence            # every action is grounded
        assert a.impact in _RANK


def test_recommend_is_deterministic_on_minimal_insight():
    data = load_demo().canonical_df
    ins = build_insight(data)                             # no forecast/segments/anomalies
    assert [a.title for a in recommend(ins)] == [a.title for a in recommend(ins)]


def test_executive_summary_falls_back_without_key(monkeypatch, full):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    _, ins = full
    assert llm.available() is False
    assert executive_summary(ins) is None


def test_quick_facts_always_available():
    facts = nlp.quick_facts(load_demo().canonical_df)
    assert "Total revenue" in facts and "Best month" in facts


def test_answer_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert nlp.answer("anything", load_demo().canonical_df) is None
