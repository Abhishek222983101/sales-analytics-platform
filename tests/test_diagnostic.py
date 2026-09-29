"""Unit tests for RFM segmentation and anomaly detection."""
import pandas as pd

from core.diagnostic import (
    RFM_LABELS,
    at_risk_revenue_share,
    compute_rfm,
    detect_anomalies,
    segment_customers,
    segment_summary,
)
from core.ingestion import load_demo


def _demo():
    return load_demo().canonical_df


def test_rfm_known_values():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-02-01"]),
            "revenue": [100.0, 50.0, 200.0],
            "customer_id": ["A", "A", "B"],
            "order_id": ["o1", "o2", "o3"],
        }
    )
    rfm = compute_rfm(df)                 # snapshot = 2020-02-02
    assert rfm.loc["A", "frequency"] == 2
    assert rfm.loc["A", "monetary"] == 150
    assert rfm.loc["B", "recency"] == 1
    assert rfm.loc["B", "monetary"] == 200


def test_segments_are_named_and_varied():
    seg = segment_customers(compute_rfm(_demo()))
    assert "segment" in seg.columns
    assert set(seg["segment"].unique()) <= set(RFM_LABELS)
    assert seg["segment"].nunique() >= 2


def test_summary_shares_sum_to_one():
    summ = segment_summary(segment_customers(compute_rfm(_demo())))
    assert abs(summ["revenue_share"].sum() - 1.0) < 1e-9


def test_champions_lead_revenue_share():
    summ = segment_summary(segment_customers(compute_rfm(_demo())))
    top = summ.sort_values("revenue_share", ascending=False).iloc[0]["segment"]
    assert top == "Champions"


def test_at_risk_share_in_range():
    summ = segment_summary(segment_customers(compute_rfm(_demo())))
    assert 0.0 <= at_risk_revenue_share(summ) <= 1.0


def test_anomalies_flagged_and_typed():
    an = detect_anomalies(_demo())
    assert "anomaly" in an.series.columns
    assert 1 <= len(an.anomalies) <= len(an.series) * 0.2
    assert set(an.anomalies["type"].unique()) <= {"spike", "drop"}
