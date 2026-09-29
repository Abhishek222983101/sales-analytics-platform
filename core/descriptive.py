"""Descriptive analytics: KPIs, revenue time-series, breakdowns, and Pareto.

Pure functions over the canonical dataframe (columns defined in schema_mapper).
Everything here answers "what happened?" in financial/revenue terms. No Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Human label -> pandas resample rule.
GRAINS: dict[str, str] = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}

# Candidate categorical dimensions, in the order we like to show them.
_DIMENSION_ORDER = ("category", "sub_category", "region", "segment", "state", "city")


@dataclass
class Kpis:
    total_revenue: float
    n_orders: int
    n_customers: Optional[int]
    avg_order_value: float
    revenue_growth_pct: Optional[float]
    best_period_label: Optional[str]


def available_dimensions(df: pd.DataFrame) -> list[str]:
    """Categorical columns present in this dataset, in display order."""
    return [d for d in _DIMENSION_ORDER if d in df.columns]


def filter_df(df: pd.DataFrame, filters: dict[str, list]) -> pd.DataFrame:
    """Apply {column: [allowed values]} filters (used for cross-filtering)."""
    out = df
    for col, allowed in filters.items():
        if allowed and col in out.columns:
            out = out[out[col].isin(allowed)]
    return out


def revenue_series(df: pd.DataFrame, grain: str = "Monthly") -> pd.DataFrame:
    """Revenue resampled to the chosen grain. Returns columns [period, revenue]."""
    rule = GRAINS.get(grain, "MS")
    s = (
        df.set_index("date")["revenue"]
        .resample(rule)
        .sum()
        .reset_index()
    )
    s.columns = ["period", "revenue"]
    return s


def compute_kpis(df: pd.DataFrame) -> Kpis:
    total = float(df["revenue"].sum())
    n_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else int(len(df))
    n_customers = int(df["customer_id"].nunique()) if "customer_id" in df.columns else None
    aov = total / n_orders if n_orders else 0.0

    monthly = revenue_series(df, "Monthly")
    growth = None
    if len(monthly) >= 2 and monthly["revenue"].iloc[-2] > 0:
        growth = float(
            (monthly["revenue"].iloc[-1] - monthly["revenue"].iloc[-2])
            / monthly["revenue"].iloc[-2] * 100.0
        )

    best_label = None
    if not monthly.empty:
        best = monthly.loc[monthly["revenue"].idxmax(), "period"]
        best_label = f"{best:%b %Y}"

    return Kpis(total, n_orders, n_customers, aov, growth, best_label)


def breakdown(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Revenue by a categorical dimension, sorted desc, with share of total."""
    g = (
        df.groupby(dim, dropna=True)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    g.columns = [dim, "revenue"]
    total = g["revenue"].sum()
    g["share"] = g["revenue"] / total if total else 0.0
    return g


def top_products(df: pd.DataFrame, n: int = 10) -> Optional[pd.DataFrame]:
    col = "product_name" if "product_name" in df.columns else (
        "product_id" if "product_id" in df.columns else None
    )
    if col is None:
        return None
    return breakdown(df, col).head(n)


def pareto(df: pd.DataFrame, entity: str = "customer_id") -> Optional[dict]:
    """Revenue-concentration curve for an entity (customer/product).

    Returns the cumulative curve plus the headline 'top 20% -> X% of revenue'.
    """
    if entity not in df.columns:
        return None
    g = df.groupby(entity)["revenue"].sum().sort_values(ascending=False)
    n = len(g)
    total = float(g.sum())
    if n == 0 or total <= 0:
        return None
    k = max(1, round(0.2 * n))
    top20_share = float(g.iloc[:k].sum() / total)
    curve = pd.DataFrame(
        {
            "rank_pct": [(i + 1) / n for i in range(n)],
            "cum_share": (g.cumsum() / total).to_numpy(),
        }
    )
    return {"entity": entity, "n": n, "top20_share": top20_share, "curve": curve}
