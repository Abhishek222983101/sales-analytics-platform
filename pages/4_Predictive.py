"""Predictive layer — "what's coming next?" XGBoost vs SARIMA, explained."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.branding import APP_NAME, oops
from core.forecasting import ForecastError, run_forecast

st.set_page_config(page_title=f"Predictive · {APP_NAME}", page_icon="🔮", layout="wide")
st.title("🔮 Predictive — what's coming next?")

data = st.session_state.get("data")
if data is None:
    st.info("Pop over to 📄 Upload first and load some data — then I can look ahead. 💛")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📄")
    st.stop()


@st.cache_data(show_spinner="Training XGBoost & SARIMA on your revenue… (just once) ✨")
def _forecast(df: pd.DataFrame):
    return run_forecast(df)


try:
    rep = _forecast(data)
except ForecastError as exc:
    st.warning(oops(str(exc)))
    st.stop()

# Store for the prescriptive layer (Phase 5 reads this).
st.session_state["forecast"] = rep

# --- horizon (slices a pre-computed 12-month forecast, so it's instant) ----
h = st.slider("How many months ahead?", 3, len(rep.future_index), min(6, len(rep.future_index)))
fut_idx = rep.future_index[:h]
xgb_f = rep.xgb.future_pred[:h]
sar_f = rep.sarima.future_pred[:h]
sar_lo = rep.sarima.future_lower[:h] if rep.sarima.future_lower is not None else None
sar_hi = rep.sarima.future_upper[:h] if rep.sarima.future_upper is not None else None

winner_f = xgb_f if rep.winner == "XGBoost" else sar_f
recent = float(rep.series.iloc[-h:].sum())
proj = float(np.sum(winner_f))
delta = (proj - recent) / recent * 100 if recent > 0 else float("nan")

# --- headline KPIs ---------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Best model", rep.winner, help="Chosen by lowest backtest RMSE.")
c2.metric(f"{rep.winner} error (MAPE)", f"{rep.winner_result.metrics['mape']:.1f}%")
c3.metric(f"Projected next {h} months", f"${proj:,.0f}",
          delta=f"{delta:+.1f}% vs prior {h}mo")
st.divider()

# --- forecast chart --------------------------------------------------------
st.subheader("Revenue — history & forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=rep.series.index, y=rep.series.to_numpy(),
                         name="Actual", line=dict(color="#2B2B33")))
if sar_lo is not None:
    fig.add_trace(go.Scatter(x=fut_idx, y=sar_hi, name="SARIMA 80% high",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut_idx, y=sar_lo, name="SARIMA 80% range",
                             fill="tonexty", fillcolor="rgba(142,125,190,0.18)",
                             line=dict(width=0)))
fig.add_trace(go.Scatter(x=fut_idx, y=xgb_f, name="XGBoost forecast",
                         line=dict(color="#E8497B", dash="dash")))
fig.add_trace(go.Scatter(x=fut_idx, y=sar_f, name="SARIMA forecast",
                         line=dict(color="#8E7DBE", dash="dot")))
fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=420,
                  yaxis_title="Revenue ($)", legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)

# --- what-if simulator -----------------------------------------------------
st.subheader("What-if")
st.caption("Nudge the forecast to sanity-check a scenario (a promo, a slow quarter…).")
adj = st.slider("Adjust the forecast by", -30, 30, 0, format="%d%%")
if adj != 0:
    adj_total = proj * (1 + adj / 100)
    st.metric(f"Adjusted projection ({adj:+d}%)", f"${adj_total:,.0f}",
              delta=f"{(adj_total - proj):+,.0f} vs base")

# --- model comparison + drivers -------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Model comparison")
    star = {"XGBoost": "", "SARIMA": ""}
    star[rep.winner] = " ⭐"
    tbl = pd.DataFrame(
        {
            "Model": [f"XGBoost{star['XGBoost']}", f"SARIMA{star['SARIMA']}"],
            "RMSE": [rep.xgb.metrics["rmse"], rep.sarima.metrics["rmse"]],
            "MAE": [rep.xgb.metrics["mae"], rep.sarima.metrics["mae"]],
            "MAPE %": [rep.xgb.metrics["mape"], rep.sarima.metrics["mape"]],
        }
    )
    st.dataframe(
        tbl.style.format({"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "MAPE %": "{:.1f}%"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("Lower is better · ⭐ = chosen model · errors measured on a held-out "
               "time-based backtest (no peeking ahead).")

with right:
    st.subheader("What drives the forecast")
    imp = rep.shap_importance.head(8)
    fig2 = px.bar(imp, x="importance", y="feature", orientation="h",
                  color_discrete_sequence=["#C64B8C"])
    fig2.update_layout(margin=dict(t=6, b=0, l=0, r=0), height=320,
                       yaxis=dict(categoryorder="total ascending"),
                       xaxis_title=rep.shap_method, yaxis_title=None)
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("How well did it do on the backtest? (actual vs predicted)"):
    bt = go.Figure()
    bt.add_trace(go.Scatter(x=rep.test_index, y=rep.test_actual, name="Actual",
                            line=dict(color="#2B2B33")))
    bt.add_trace(go.Scatter(x=rep.test_index, y=rep.xgb_test_pred, name="XGBoost",
                            line=dict(color="#E8497B", dash="dash")))
    bt.add_trace(go.Scatter(x=rep.test_index, y=rep.sarima_test_pred, name="SARIMA",
                            line=dict(color="#8E7DBE", dash="dot")))
    bt.update_layout(margin=dict(t=6, b=0, l=0, r=0), height=300,
                     yaxis_title="Revenue ($)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(bt, use_container_width=True)

st.divider()
st.page_link("pages/5_Prescriptive.py", label="Next: what should we do? →", icon="💡")
