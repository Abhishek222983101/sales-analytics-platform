"""Diagnostic analytics: RFM customer segmentation (KMeans) + revenue anomalies.

Answers "why did it happen?" — who the customers are (Champions … Lost) and which
months behaved abnormally. Pure Python — no Streamlit. Degrades gracefully when
optional columns (customer_id) are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Best -> worst; KMeans clusters are ranked and mapped onto these.
RFM_LABELS = ["Champions", "Loyal", "At-Risk", "Lost"]
_AT_RISK = {"At-Risk", "Lost"}


# --------------------------------------------------------------------------- #
# RFM segmentation                                                            #
# --------------------------------------------------------------------------- #
def has_customers(df: pd.DataFrame) -> bool:
    return "customer_id" in df.columns and df["customer_id"].notna().any()


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Recency (days since last order), Frequency (orders), Monetary (Σ revenue)."""
    snapshot = df["date"].max() + pd.Timedelta(days=1)
    grp = df.groupby("customer_id")
    recency = grp["date"].max().apply(lambda d: (snapshot - d).days)
    frequency = grp["order_id"].nunique() if "order_id" in df.columns else grp.size()
    monetary = grp["revenue"].sum()
    return pd.DataFrame(
        {"recency": recency, "frequency": frequency, "monetary": monetary}
    ).dropna()


def segment_customers(rfm: pd.DataFrame, k: int = 4, random_state: int = 42) -> pd.DataFrame:
    """Cluster customers on scaled RFM and name clusters best->worst."""
    feats = rfm.copy()
    k = max(1, min(k, feats["monetary"].nunique(), len(feats)))

    matrix = np.column_stack([
        -feats["recency"].to_numpy(float),          # recent = better
        np.log1p(feats["frequency"].to_numpy(float)),
        np.log1p(feats["monetary"].to_numpy(float)),
    ])
    scaled = StandardScaler().fit_transform(matrix)
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(scaled)
    feats["cluster"] = labels

    prof = feats.groupby("cluster").agg(
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    )
    prof["score"] = (
        prof["monetary"].rank() + prof["frequency"].rank() - prof["recency"].rank()
    )
    order = prof["score"].sort_values(ascending=False).index.tolist()
    name_map = {
        cl: (RFM_LABELS[i] if i < len(RFM_LABELS) else f"Group {i + 1}")
        for i, cl in enumerate(order)
    }
    feats["segment"] = feats["cluster"].map(name_map)
    return feats


def segment_summary(seg: pd.DataFrame) -> pd.DataFrame:
    """Per-segment customer count, revenue and share, ordered best->worst."""
    s = (
        seg.groupby("segment")
        .agg(
            customers=("monetary", "size"),
            revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
        )
        .reset_index()
    )
    total = s["revenue"].sum()
    s["revenue_share"] = s["revenue"] / total if total else 0.0
    order = {name: i for i, name in enumerate(RFM_LABELS)}
    s["_o"] = s["segment"].map(lambda x: order.get(x, 99))
    return s.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def at_risk_revenue_share(summary: pd.DataFrame) -> float:
    """Share of revenue held by At-Risk + Lost customers."""
    mask = summary["segment"].isin(_AT_RISK)
    return float(summary.loc[mask, "revenue_share"].sum())


# --------------------------------------------------------------------------- #
# Anomaly detection                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyResult:
    series: pd.DataFrame          # index=month, cols: revenue, pct_change, anomaly, score
    anomalies: pd.DataFrame       # only flagged months, with a 'type' column


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.08) -> AnomalyResult:
    """Flag abnormal monthly revenue with an Isolation Forest."""
    monthly = df.set_index("date")["revenue"].resample("MS").sum().astype(float)
    frame = pd.DataFrame({"revenue": monthly})
    frame["pct_change"] = frame["revenue"].pct_change().fillna(0.0)

    matrix = StandardScaler().fit_transform(frame[["revenue", "pct_change"]].to_numpy())
    iso = IsolationForest(contamination=contamination, random_state=42)
    flags = iso.fit_predict(matrix)
    frame["anomaly"] = flags == -1
    frame["score"] = iso.decision_function(matrix)

    anomalies = frame[frame["anomaly"]].copy()
    anomalies["type"] = np.where(anomalies["pct_change"] >= 0, "spike", "drop")
    return AnomalyResult(frame, anomalies)
