"""Descriptive layer — "what happened?" A cross-filterable revenue dashboard."""
import plotly.express as px
import streamlit as st

from core import descriptive as D
from core.branding import APP_NAME

st.set_page_config(page_title=f"Descriptive · {APP_NAME}", page_icon="🌸", layout="wide")

# Soft, on-brand chart palette.
ROSE = ["#E8497B", "#F4A6C0", "#C64B8C", "#F7C948", "#8E7DBE", "#5AA7A7", "#E9A178"]
px.defaults.color_discrete_sequence = ROSE
px.defaults.template = "plotly_white"

st.title("🌸 Descriptive — what happened?")

data = st.session_state.get("data")
if data is None:
    st.info("Pop over to 📄 Upload first and load some data — then this comes alive. 💛")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📄")
    st.stop()

# --- sidebar cross-filters -------------------------------------------------
st.sidebar.header("Filter the view")
st.sidebar.caption("Everything below updates together. ✨")

filters: dict[str, list] = {}
for dim in D.available_dimensions(data):
    values = sorted(v for v in data[dim].dropna().unique().tolist())
    picked = st.sidebar.multiselect(dim.replace("_", " ").title(), values, default=[])
    if picked:
        filters[dim] = picked

dmin, dmax = data["date"].min().date(), data["date"].max().date()
date_range = st.sidebar.slider(
    "Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax)
)

df = D.filter_df(data, filters)
df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

if df.empty:
    st.warning("No rows match these filters — try loosening one? 🙈")
    st.stop()

# --- KPI cards -------------------------------------------------------------
k = D.compute_kpis(df)
c1, c2, c3, c4 = st.columns(4)
growth = f"{k.revenue_growth_pct:+.1f}% MoM" if k.revenue_growth_pct is not None else None
c1.metric("Total revenue", f"${k.total_revenue:,.0f}", delta=growth)
c2.metric("Orders", f"{k.n_orders:,}")
c3.metric("Avg order value", f"${k.avg_order_value:,.2f}")
if k.n_customers is not None:
    c4.metric("Customers", f"{k.n_customers:,}")
elif k.best_period_label:
    c4.metric("Best month", k.best_period_label)

st.caption(f"Showing **{len(df):,}** rows · best month so far: **{k.best_period_label or '—'}**.")
st.divider()

# --- revenue trend ---------------------------------------------------------
st.subheader("Revenue over time")
grain = st.radio("Grain", list(D.GRAINS), horizontal=True, index=2, label_visibility="collapsed")
series = D.revenue_series(df, grain)
fig = px.area(series, x="period", y="revenue", markers=(grain != "Daily"))
fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), yaxis_title="Revenue ($)", xaxis_title=None)
fig.update_traces(line_color="#E8497B", fillcolor="rgba(232,73,123,0.15)")
st.plotly_chart(fig, use_container_width=True)

# --- breakdowns ------------------------------------------------------------
dims = D.available_dimensions(df)
if dims:
    st.subheader("Where the revenue comes from")
    tabs = st.tabs([d.replace("_", " ").title() for d in dims])
    for tab, dim in zip(tabs, dims):
        with tab:
            bd = D.breakdown(df, dim)
            bd = bd.assign(label=bd["share"].map(lambda s: f"{s:.0%}"))
            fig = px.bar(bd, x="revenue", y=dim, orientation="h", text="label")
            fig.update_layout(margin=dict(t=6, b=0, l=0, r=0),
                              yaxis=dict(categoryorder="total ascending"),
                              xaxis_title="Revenue ($)", yaxis_title=None, height=360)
            st.plotly_chart(fig, use_container_width=True)

# --- top products + Pareto -------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Top products")
    tp = D.top_products(df, 10)
    if tp is not None:
        tp = tp.rename(columns={tp.columns[0]: "product"})
        tp["revenue"] = tp["revenue"].map(lambda v: f"${v:,.0f}")
        tp["share"] = tp["share"].map(lambda s: f"{s:.1%}")
        st.dataframe(tp, use_container_width=True, hide_index=True)
    else:
        st.caption("No product column in this dataset — that's okay. 💛")

with right:
    st.subheader("Revenue concentration")
    par = D.pareto(df, "customer_id") or D.pareto(df, "product_name")
    if par:
        share = par["top20_share"]
        st.metric(f"Top 20% of {par['entity'].replace('_', ' ')}s", f"{share:.0%} of revenue")
        fig = px.line(par["curve"], x="rank_pct", y="cum_share")
        fig.update_layout(margin=dict(t=6, b=0, l=0, r=0), height=300,
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="Share of entities", yaxis_title="Share of revenue")
        fig.update_traces(line_color="#C64B8C")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The steeper the early climb, the more revenue leans on a few names.")
    else:
        st.caption("Need a customer or product column to show concentration.")

st.divider()
st.page_link("pages/3_Diagnostic.py", label="Next: why did it happen? →", icon="🔍")
