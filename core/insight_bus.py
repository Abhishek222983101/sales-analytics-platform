"""The InsightJSON spine — one validated object assembled from every layer's output.

This is the heart of the project's contribution: the prescriptive LLM reasons over
this *combined* structure (forecast + segments + anomalies + concentration + movers),
so each recommendation can cite the exact field that justifies it. Pure Python.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from core import descriptive as D


class Concentration(BaseModel):
    entity: str
    top20_share: float
    top_category: Optional[str] = None


class Mover(BaseModel):
    dimension: str
    name: str
    change_pct: float


class ForecastInsight(BaseModel):
    winner: str
    horizon_months: int
    delta_pct: float
    mape: float
    top_drivers: list[str]


class SegmentInsight(BaseModel):
    at_risk_customers: int
    at_risk_revenue_share: float
    champions_revenue_share: float


class AnomalyInsight(BaseModel):
    count: int
    latest_month: Optional[str] = None
    latest_type: Optional[str] = None


class Insight(BaseModel):
    total_revenue: float
    date_range: list[str]
    concentration: Optional[Concentration] = None
    movers: list[Mover] = Field(default_factory=list)
    forecast: Optional[ForecastInsight] = None
    segments: Optional[SegmentInsight] = None
    anomalies: Optional[AnomalyInsight] = None


def _movers(data: pd.DataFrame, dims=("region", "category"),
            months: int = 3, min_pct: float = 8.0) -> list[Mover]:
    """Biggest revenue swings by dimension: last `months` vs the prior `months`."""
    last = data["date"].max()
    cut1 = last - pd.DateOffset(months=months)
    cut2 = last - pd.DateOffset(months=2 * months)
    recent = data[data["date"] > cut1]
    prior = data[(data["date"] <= cut1) & (data["date"] > cut2)]

    movers: list[Mover] = []
    for dim in dims:
        if dim not in data.columns:
            continue
        r = recent.groupby(dim)["revenue"].sum()
        p = prior.groupby(dim)["revenue"].sum()
        for name in set(r.index) | set(p.index):
            pv, rv = float(p.get(name, 0.0)), float(r.get(name, 0.0))
            if pv > 0:
                change = (rv - pv) / pv * 100
                if abs(change) >= min_pct:
                    movers.append(Mover(dimension=dim, name=str(name), change_pct=round(change, 1)))
    movers.sort(key=lambda m: abs(m.change_pct), reverse=True)
    return movers[:6]


def build_insight(data: pd.DataFrame, forecast=None, segments=None, anomalies=None) -> Insight:
    """Assemble the InsightJSON from the canonical data + any computed layer outputs."""
    date_range = [f"{data['date'].min():%b %Y}", f"{data['date'].max():%b %Y}"]

    par = D.pareto(data, "customer_id") or D.pareto(data, "product_name")
    concentration = None
    if par:
        top_cat = None
        if "category" in data.columns:
            top_cat = str(D.breakdown(data, "category").iloc[0]["category"])
        concentration = Concentration(
            entity=par["entity"], top20_share=round(par["top20_share"], 3), top_category=top_cat
        )

    forecast_insight = None
    if forecast is not None:
        forecast_insight = ForecastInsight(
            winner=forecast.winner,
            horizon_months=forecast.horizon,
            delta_pct=round(forecast.delta_pct, 1),
            mape=round(forecast.winner_result.metrics["mape"], 1),
            top_drivers=list(forecast.shap_importance["feature"].head(3)),
        )

    segment_insight = None
    if segments is not None and not segments.empty:
        at_risk_mask = segments["segment"].isin(["At-Risk", "Lost"])
        champ_mask = segments["segment"] == "Champions"
        segment_insight = SegmentInsight(
            at_risk_customers=int(segments.loc[at_risk_mask, "customers"].sum()),
            at_risk_revenue_share=round(float(segments.loc[at_risk_mask, "revenue_share"].sum()), 3),
            champions_revenue_share=round(float(segments.loc[champ_mask, "revenue_share"].sum()), 3),
        )

    anomaly_insight = None
    if anomalies is not None and not anomalies.empty:
        latest = anomalies.sort_index().iloc[-1]
        anomaly_insight = AnomalyInsight(
            count=int(len(anomalies)),
            latest_month=f"{anomalies.index.max():%b %Y}",
            latest_type=str(latest["type"]),
        )

    return Insight(
        total_revenue=round(float(data["revenue"].sum()), 2),
        date_range=date_range,
        concentration=concentration,
        movers=_movers(data),
        forecast=forecast_insight,
        segments=segment_insight,
        anomalies=anomaly_insight,
    )
