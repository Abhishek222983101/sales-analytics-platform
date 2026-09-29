"""Prescriptive analytics: turn the InsightJSON into ranked, evidence-cited actions.

The recommendations are **rule-based and deterministic** — every action names the
exact InsightJSON field that justifies it, and the whole thing works with no LLM
key at all. The LLM (when configured) only adds a warm natural-language summary on
top; it never invents the actions. This is the "explainable recommendation" idea.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core import llm
from core.insight_bus import Insight

_IMPACT_RANK = {"High": 3, "Medium": 2, "Low": 1}


@dataclass
class Action:
    title: str
    why: str            # plain-English rationale, quoting the numbers
    impact: str         # "High" | "Medium" | "Low"
    evidence: str       # the InsightJSON field path + value that justifies it


def recommend(insight: Insight) -> list[Action]:
    """Ranked, evidence-cited actions derived purely from the InsightJSON."""
    actions: list[Action] = []

    seg = insight.segments
    if seg and seg.at_risk_revenue_share > 0.03:
        actions.append(Action(
            "Win back your At-Risk customers",
            f"{seg.at_risk_customers} customers who together hold "
            f"{seg.at_risk_revenue_share:.0%} of revenue have gone quiet. A friendly "
            f"check-in or a small offer could re-activate them.",
            "High" if seg.at_risk_revenue_share >= 0.15 else "Medium",
            f"segments.at_risk_revenue_share = {seg.at_risk_revenue_share}",
        ))

    conc = insight.concentration
    if conc and conc.top20_share > 0.60:
        actions.append(Action(
            "Protect your key accounts",
            f"Your top 20% of {conc.entity.replace('_', ' ')}s drive "
            f"{conc.top20_share:.0%} of revenue — that's concentration risk. Give them "
            f"dedicated care so a single loss doesn't sting.",
            "High" if conc.top20_share > 0.75 else "Medium",
            f"concentration.top20_share = {conc.top20_share}",
        ))

    for m in [m for m in insight.movers if m.change_pct <= -10][:2]:
        actions.append(Action(
            f"Investigate the dip in {m.name}",
            f"{m.name} ({m.dimension}) revenue fell {m.change_pct:.0f}% last quarter "
            f"versus the one before — worth a look before it compounds.",
            "High" if m.change_pct <= -20 else "Medium",
            f"movers[{m.dimension}={m.name}].change_pct = {m.change_pct}%",
        ))

    for m in [m for m in insight.movers if m.change_pct >= 15][:1]:
        actions.append(Action(
            f"Double down on {m.name}",
            f"{m.name} ({m.dimension}) grew {m.change_pct:.0f}% last quarter — "
            f"lean into what's working there.",
            "Medium",
            f"movers[{m.dimension}={m.name}].change_pct = +{m.change_pct}%",
        ))

    fc = insight.forecast
    if fc and fc.delta_pct <= -5:
        actions.append(Action(
            f"Plan for a softer {fc.horizon_months} months",
            f"The {fc.winner} model projects revenue {fc.delta_pct:.0f}% below the prior "
            f"{fc.horizon_months} months. Budget conservatively and line up demand now.",
            "High" if fc.delta_pct <= -15 else "Medium",
            f"forecast.delta_pct = {fc.delta_pct}%",
        ))
    elif fc and fc.delta_pct >= 5:
        actions.append(Action(
            f"Get ready for growth over {fc.horizon_months} months",
            f"The {fc.winner} model projects revenue {fc.delta_pct:+.0f}% versus the prior "
            f"period — make sure stock and staffing can keep up.",
            "Medium",
            f"forecast.delta_pct = +{fc.delta_pct}%",
        ))

    an = insight.anomalies
    if an and an.count > 0 and an.latest_type == "drop":
        actions.append(Action(
            f"Follow up on the {an.latest_month} dip",
            f"{an.latest_month} was flagged as an unusual revenue drop. Check whether it "
            f"was a one-off or the start of a trend.",
            "Medium",
            f"anomalies.latest = {an.latest_month} (drop)",
        ))

    actions.sort(key=lambda a: _IMPACT_RANK[a.impact], reverse=True)
    return actions[:5]


_SUMMARY_SYSTEM = (
    "You are Lumi, a warm and concise sales analyst. You are given a JSON of analytics "
    "findings about a business's sales. Write a 3-4 sentence executive summary in plain, "
    "encouraging English that a non-technical owner could act on. Reference the actual "
    "numbers from the JSON. No preamble, no bullet points, no markdown headings."
)


def executive_summary(insight: Insight) -> Optional[str]:
    """A warm natural-language summary of the InsightJSON, or None if no LLM key."""
    if not llm.available():
        return None
    return llm.complete(_SUMMARY_SYSTEM, insight.model_dump_json(indent=2),
                        max_tokens=1200, temperature=0.4)
