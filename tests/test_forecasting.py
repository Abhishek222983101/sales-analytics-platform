"""Unit tests for the forecasting engine: leakage, metrics, consistency, output."""
import numpy as np
import pandas as pd
import pytest

from core.forecasting import (
    FEATURE_COLUMNS,
    MAX_HORIZON,
    ForecastError,
    _feature_row,
    _metrics,
    make_features,
    monthly_revenue,
    run_forecast,
    time_split,
)
from core.ingestion import load_demo


def _series():
    return monthly_revenue(load_demo().canonical_df)


def test_time_split_has_no_leakage():
    tr, te = time_split(make_features(_series()))
    assert tr.index.max() < te.index.min()   # test strictly after train
    assert len(te) > 0


def test_metrics_formulas():
    assert _metrics([100, 200], [100, 200])["rmse"] == 0
    m = _metrics([100, 100], [90, 110])
    assert round(m["mae"], 2) == 10 and round(m["rmse"], 2) == 10 and round(m["mape"], 2) == 10


def test_feature_row_matches_vectorized_features():
    # The recursive forecaster must build features identically to training,
    # or multi-step forecasts silently drift.
    s = _series()
    feats = make_features(s)
    ts = feats.index[20]
    hist = list(s[s.index < ts].to_numpy())
    row = _feature_row(hist, ts)
    for col in FEATURE_COLUMNS:
        assert np.isclose(row[col], feats.loc[ts, col]), col


def test_run_forecast_shapes_and_sanity():
    rep = run_forecast(load_demo().canonical_df)
    assert rep.winner in ("XGBoost", "SARIMA")
    assert len(rep.future_index) == MAX_HORIZON
    assert len(rep.winner_result.future_pred) == MAX_HORIZON
    for metrics in (rep.xgb.metrics, rep.sarima.metrics):
        assert np.isfinite(metrics["rmse"])
        assert 0 < metrics["mape"] < 100          # credible monthly error
    assert set(rep.shap_importance["feature"]) == set(FEATURE_COLUMNS)
    assert rep.sarima.future_lower is not None
    assert (rep.sarima.future_upper >= rep.sarima.future_lower).all()


def test_short_series_raises_forecast_error():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "revenue": [10.0, 20.0]}
    )
    with pytest.raises(ForecastError):
        run_forecast(df)
