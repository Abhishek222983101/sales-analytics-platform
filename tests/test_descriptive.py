"""Unit tests for the descriptive analytics functions."""
from core.descriptive import (
    available_dimensions,
    breakdown,
    compute_kpis,
    filter_df,
    pareto,
    revenue_series,
    top_products,
)
from core.ingestion import load_demo


def _demo():
    return load_demo().canonical_df


def test_kpis_on_demo():
    k = compute_kpis(_demo())
    assert k.total_revenue > 0
    assert k.n_orders > 0
    assert k.n_customers and k.n_customers > 0
    assert k.avg_order_value > 0
    assert k.best_period_label


def test_revenue_series_grains_nonempty():
    df = _demo()
    for grain in ("Daily", "Weekly", "Monthly"):
        s = revenue_series(df, grain)
        assert list(s.columns) == ["period", "revenue"]
        assert len(s) > 0 and s["revenue"].sum() > 0


def test_revenue_series_conserves_total():
    df = _demo()
    total = float(df["revenue"].sum())
    monthly = float(revenue_series(df, "Monthly")["revenue"].sum())
    assert abs(monthly - total) < 1.0   # resample-sum conserves revenue (to the dollar)


def test_breakdown_sorted_and_shares_sum_to_one():
    bd = breakdown(_demo(), "region")
    assert abs(bd["share"].sum() - 1.0) < 1e-9
    rev = bd["revenue"].to_numpy()
    assert (rev[:-1] >= rev[1:]).all()   # sorted descending


def test_top_products_present():
    tp = top_products(_demo(), 10)
    assert tp is not None and len(tp) == 10


def test_pareto_share_in_range():
    par = pareto(_demo(), "customer_id")
    assert par and 0 < par["top20_share"] <= 1


def test_filter_df_narrows():
    df = _demo()
    sub = filter_df(df, {"region": ["West"]})
    assert set(sub["region"].dropna().unique()) <= {"West"}
    assert 0 < len(sub) < len(df)


def test_available_dimensions_on_demo():
    dims = available_dimensions(_demo())
    assert "region" in dims and "category" in dims and "segment" in dims
