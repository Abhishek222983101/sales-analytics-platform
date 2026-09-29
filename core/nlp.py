"""Ask-your-data: answer natural-language questions grounded in pre-computed facts.

Safety by design: we never execute model-generated code. Instead we build a compact
CONTEXT of aggregates and let the LLM answer from it (RAG-free, grounded). Without a
key, :func:`quick_facts` still gives useful computed answers.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from core import descriptive as D, llm
from core.insight_bus import Insight


def quick_facts(data: pd.DataFrame) -> dict[str, str]:
    """A handful of always-available computed answers (no LLM needed)."""
    k = D.compute_kpis(data)
    facts = {
        "Total revenue": f"${k.total_revenue:,.0f}",
        "Orders": f"{k.n_orders:,}",
        "Avg order value": f"${k.avg_order_value:,.2f}",
        "Best month": k.best_period_label or "—",
    }
    if k.revenue_growth_pct is not None:
        facts["Latest MoM growth"] = f"{k.revenue_growth_pct:+.1f}%"
    for dim, label in (("region", "Top region"), ("category", "Top category")):
        if dim in data.columns:
            top = D.breakdown(data, dim).iloc[0]
            facts[label] = f"{top[dim]} ({top['share']:.0%})"
    return facts


def data_context(data: pd.DataFrame, insight: Optional[Insight] = None) -> str:
    """Compact, factual context the LLM must answer from."""
    lines = [f"Columns available: {', '.join(data.columns)}."]
    for label, value in quick_facts(data).items():
        lines.append(f"{label}: {value}.")
    for dim in ("region", "category", "segment"):
        if dim in data.columns:
            bd = D.breakdown(data, dim).head(6)
            parts = [f"{r[dim]} ${r['revenue']:,.0f} ({r['share']:.0%})" for _, r in bd.iterrows()]
            lines.append(f"Revenue by {dim}: " + "; ".join(parts) + ".")
    if insight is not None:
        lines.append("Structured findings JSON:")
        lines.append(insight.model_dump_json())
    return "\n".join(lines)


_ASK_SYSTEM = (
    "You are Lumi, a friendly data assistant. Answer ONLY from the CONTEXT provided about "
    "the user's sales data. Use the numbers in the context. If the answer isn't in the "
    "context, say so warmly and suggest what they could look at instead. Be concise and "
    "kind. Never invent figures."
)


def answer(question: str, data: pd.DataFrame, insight: Optional[Insight] = None) -> Optional[str]:
    """LLM answer grounded in the data context, or None if no LLM key."""
    if not llm.available():
        return None
    user = f"CONTEXT:\n{data_context(data, insight)}\n\nQUESTION: {question}"
    return llm.complete(_ASK_SYSTEM, user, max_tokens=1200, temperature=0.3)
