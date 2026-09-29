"""Diagnostic layer — "why did it happen?" Segments + anomalies."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import diagnostic as D
from core.branding import APP_NAME

st.set_page_config(page_title=f"Diagnostic · {APP_NAME}", page_icon="🔍", layout="wide")
st.title("🔍 Diagnostic — why did it happen?")

SEG_COLORS = {"Champions": "#E8497B", "Loyal": "#8E7DBE", "At-Risk": "#F7C948", "Lost": "#9AA0A6"}

data = st.session_state.get("data")
if data is None:
    st.info("Pop over to 📄 Upload first and load some data — then I can dig in. 💛")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📄")
    st.stop()

# --- customer segments (needs customer_id) ---------------------------------
st.subheader("Who your customers are")
if D.has_customers(data):
    rfm = D.compute_rfm(data)
    seg = D.segment_customers(rfm)
    summary = D.segment_summary(seg)
    st.session_state["segments"] = summary   # for the prescriptive layer

    cols = st.columns(len(summary))
    for col, row in zip(cols, summary.itertuples()):
        col.metric(row.segment, f"{row.customers:,}", delta=f"{row.revenue_share:.0%} of revenue",
                   delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        fig = px.scatter(
            seg, x="recency", y="monetary", color="segment", size="frequency",
            color_discrete_map=SEG_COLORS, category_orders={"segment": D.RFM_LABELS},
            labels={"recency": "Days since last order", "monetary": "Lifetime revenue ($)"},
            log_y=True, opacity=0.7,
        )
        fig.update_layout(margin=dict(t=6, b=0, l=0, r=0), height=380,
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        show = summary.assign(
            revenue=summary["revenue"].map(lambda v: f"${v:,.0f}"),
            share=summary["revenue_share"].map(lambda s: f"{s:.1%}"),
        )[["segment", "customers", "revenue", "share"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        risk = D.at_risk_revenue_share(summary)
        st.metric("Revenue at risk (At-Risk + Lost)", f"{risk:.1%}")
        st.caption("These customers were valuable but have gone quiet — worth a gentle nudge.")
else:
    st.caption("No customer column in this dataset, so I'll skip segmentation — "
               "the anomaly scan below still works. 💛")

st.divider()

# --- anomalies (needs only date + revenue) ---------------------------------
st.subheader("Months that behaved unusually")
an = D.detect_anomalies(data)
st.session_state["anomalies"] = an.anomalies

frame = an.series
fig = go.Figure()
fig.add_trace(go.Scatter(x=frame.index, y=frame["revenue"], name="Monthly revenue",
                         line=dict(color="#2B2B33")))
if not an.anomalies.empty:
    spikes = an.anomalies[an.anomalies["type"] == "spike"]
    drops = an.anomalies[an.anomalies["type"] == "drop"]
    fig.add_trace(go.Scatter(x=spikes.index, y=spikes["revenue"], mode="markers",
                             name="Spike", marker=dict(color="#2E9E7B", size=12, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=drops.index, y=drops["revenue"], mode="markers",
                             name="Drop", marker=dict(color="#E8497B", size=12, symbol="triangle-down")))
fig.update_layout(margin=dict(t=6, b=0, l=0, r=0), height=340,
                  yaxis_title="Revenue ($)", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

if an.anomalies.empty:
    st.caption("Nothing looked out of the ordinary — steady as she goes. ✨")
else:
    tbl = an.anomalies.reset_index()
    tbl["month"] = tbl["date"].dt.strftime("%b %Y")
    tbl["revenue"] = tbl["revenue"].map(lambda v: f"${v:,.0f}")
    tbl["change"] = tbl["pct_change"].map(lambda p: f"{p:+.0%}")
    st.dataframe(tbl[["month", "revenue", "change", "type"]],
                 use_container_width=True, hide_index=True)

# --- forecast drivers (reuse SHAP from the Predictive layer if available) ---
rep = st.session_state.get("forecast")
if rep is not None:
    st.divider()
    st.subheader("What drives the revenue forecast")
    imp = rep.shap_importance.head(8)
    fig2 = px.bar(imp, x="importance", y="feature", orientation="h",
                  color_discrete_sequence=["#C64B8C"])
    fig2.update_layout(margin=dict(t=6, b=0, l=0, r=0), height=300,
                       yaxis=dict(categoryorder="total ascending"),
                       xaxis_title=rep.shap_method, yaxis_title=None)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.page_link("pages/4_Predictive.py", label="Next: what's coming next? →", icon="🔮")
