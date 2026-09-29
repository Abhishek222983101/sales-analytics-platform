"""Predictive analytics: monthly revenue forecasting with XGBoost and SARIMA.

Design choices that matter for correctness and credibility:
  * Revenue is aggregated to a **monthly** series — stable, seasonal, and
    business-readable ("next 6 months"). Weekly was too noisy and made SARIMA's
    52-period seasonality diverge.
  * The train/test split is **time-based** (never random) — no look-ahead.
  * The backtest is **recursive multi-step** for BOTH models, so the XGBoost-vs-
    SARIMA comparison is apples-to-apples (XGBoost never peeks at test-window
    actuals for its lag features).
  * Explainability via SHAP (falls back to XGBoost gain importance).

Pure Python — no Streamlit. Raises ``ForecastError`` with a kind message when the
series is too short to model.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "year", "month", "quarter",
    "lag_1", "lag_2", "lag_3", "lag_12",
    "roll_mean_3", "roll_std_3",
]
MIN_MONTHS = 24         # below this we don't pretend to forecast
SEASONAL_PERIOD = 12    # monthly seasonality
MAX_HORIZON = 12        # always compute this many months; the UI slices it
GRAIN_LABEL = "month"


class ForecastError(RuntimeError):
    """Raised when the data can't support a forecast (shown kindly in the UI)."""


# --------------------------------------------------------------------------- #
# Series prep & features                                                      #
# --------------------------------------------------------------------------- #
def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    """Aggregate canonical rows to a gap-free monthly revenue series."""
    return df.set_index("date")["revenue"].resample("MS").sum().astype(float)


def make_features(series: pd.Series) -> pd.DataFrame:
    """Build the supervised feature table (target 'revenue' + FEATURE_COLUMNS)."""
    f = pd.DataFrame({"revenue": series})
    idx = f.index
    f["year"] = idx.year
    f["month"] = idx.month
    f["quarter"] = idx.quarter
    for lag in (1, 2, 3, 12):
        f[f"lag_{lag}"] = f["revenue"].shift(lag)
    f["roll_mean_3"] = f["revenue"].shift(1).rolling(3).mean()
    f["roll_std_3"] = f["revenue"].shift(1).rolling(3).std()
    return f.dropna()


def time_split(features: pd.DataFrame, test_frac: float = 0.2):
    n = len(features)
    k = max(1, int(round(n * (1 - test_frac))))
    return features.iloc[:k], features.iloc[k:]


def _feature_row(hist: list[float], ts: pd.Timestamp) -> dict:
    """One feature row for timestamp ``ts`` given monthly history up to ts-1."""
    return {
        "year": ts.year,
        "month": ts.month,
        "quarter": ts.quarter,
        "lag_1": hist[-1],
        "lag_2": hist[-2],
        "lag_3": hist[-3],
        "lag_12": hist[-12],
        "roll_mean_3": float(np.mean(hist[-3:])),
        "roll_std_3": float(np.std(hist[-3:], ddof=1)),
    }


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    freq = series.index.freq or "MS"
    return pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def _fit_xgb(train: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror",
        # Pin to CPU: the default probes for a GPU and costs ~7s of cold start.
        device="cpu", tree_method="hist", n_jobs=2,
    )
    model.fit(train[FEATURE_COLUMNS], train["revenue"])
    return model


def _forecast_xgb_recursive(model, history: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Recursive multi-step forecast: each step feeds its own prediction forward."""
    hist = [float(v) for v in history.to_numpy()]
    preds = []
    for ts in index:
        row = pd.DataFrame([_feature_row(hist, ts)])[FEATURE_COLUMNS]
        yhat = float(model.predict(row)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.asarray(preds)


def _fit_sarima(series: pd.Series):
    """Fit the stable 'airline' SARIMA (0,1,1)(0,1,1,12); fall back to non-seasonal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return SARIMAX(
                series, order=(0, 1, 1),
                seasonal_order=(0, 1, 1, SEASONAL_PERIOD),
            ).fit(disp=False, maxiter=50)
        except Exception:
            return SARIMAX(series, order=(0, 1, 1)).fit(disp=False, maxiter=50)


# --------------------------------------------------------------------------- #
# Metrics & explainability                                                     #
# --------------------------------------------------------------------------- #
def _metrics(actual, pred) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    err = a - p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = a != 0
    mape = float(np.mean(np.abs(err[mask] / a[mask])) * 100) if mask.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def _shap_importance(model, X: pd.DataFrame):
    try:
        import shap
        sample = X.sample(min(200, len(X)), random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = shap.TreeExplainer(model).shap_values(sample)
        importance = np.abs(values).mean(axis=0)
        method = "SHAP (mean |impact|)"
    except Exception:
        importance = model.feature_importances_
        method = "XGBoost gain importance"
    out = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": np.asarray(importance, float)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return out, method


# --------------------------------------------------------------------------- #
# Result containers + orchestration                                            #
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    name: str
    metrics: dict
    future_pred: np.ndarray
    future_lower: Optional[np.ndarray] = None
    future_upper: Optional[np.ndarray] = None


@dataclass
class ForecastReport:
    series: pd.Series
    test_index: pd.DatetimeIndex
    test_actual: np.ndarray
    xgb_test_pred: np.ndarray
    sarima_test_pred: np.ndarray
    future_index: pd.DatetimeIndex
    xgb: ModelResult
    sarima: ModelResult
    winner: str
    horizon: int
    delta_pct: float
    shap_importance: pd.DataFrame
    shap_method: str

    @property
    def winner_result(self) -> ModelResult:
        return self.xgb if self.winner == "XGBoost" else self.sarima


def run_forecast(df: pd.DataFrame, horizon: int = MAX_HORIZON, test_frac: float = 0.2) -> ForecastReport:
    series = monthly_revenue(df)
    if len(series) < MIN_MONTHS:
        raise ForecastError(
            f"I need at least ~{MIN_MONTHS} months of history to forecast honestly — "
            f"this slice only has {len(series)}."
        )

    feats = make_features(series)
    train, test = time_split(feats, test_frac)
    cutoff = test.index[0]
    s_train = series[series.index < cutoff]
    s_test = series[series.index >= cutoff]

    # --- XGBoost: recursive backtest, then recursive future -----------------
    xgb_bt = _fit_xgb(train)
    xgb_test_pred = _forecast_xgb_recursive(xgb_bt, s_train, s_test.index)
    xgb_metrics = _metrics(s_test.to_numpy(), xgb_test_pred)

    xgb_full = _fit_xgb(feats)
    future_index = _future_index(series, horizon)
    xgb_future = _forecast_xgb_recursive(xgb_full, series, future_index)

    # --- SARIMA: multi-step backtest, then future with 80% CI ---------------
    sar_bt = _fit_sarima(s_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sar_test_pred = np.asarray(sar_bt.get_forecast(steps=len(s_test)).predicted_mean)
    sar_metrics = _metrics(s_test.to_numpy(), sar_test_pred)

    sar_full = _fit_sarima(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sfc = sar_full.get_forecast(steps=horizon)
        sar_future = np.asarray(sfc.predicted_mean)
        ci = sfc.conf_int(alpha=0.2)
    sar_low = np.asarray(ci.iloc[:, 0])
    sar_up = np.asarray(ci.iloc[:, 1])

    winner = "XGBoost" if xgb_metrics["rmse"] <= sar_metrics["rmse"] else "SARIMA"

    recent = float(series.iloc[-horizon:].sum())
    winner_future = xgb_future if winner == "XGBoost" else sar_future
    delta_pct = float((winner_future.sum() - recent) / recent * 100) if recent > 0 else float("nan")

    shap_df, shap_method = _shap_importance(xgb_full, feats[FEATURE_COLUMNS])

    return ForecastReport(
        series=series,
        test_index=s_test.index,
        test_actual=s_test.to_numpy(),
        xgb_test_pred=xgb_test_pred,
        sarima_test_pred=sar_test_pred,
        future_index=future_index,
        xgb=ModelResult("XGBoost", xgb_metrics, xgb_future),
        sarima=ModelResult("SARIMA", sar_metrics, sar_future, sar_low, sar_up),
        winner=winner,
        horizon=horizon,
        delta_pct=delta_pct,
        shap_importance=shap_df,
        shap_method=shap_method,
    )
