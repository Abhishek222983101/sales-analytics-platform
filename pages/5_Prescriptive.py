"""Prescriptive layer — "what should we do?" Ranked, evidence-cited AI actions."""
import streamlit as st

from core import llm
from core.branding import APP_NAME
from core.insight_bus import build_insight
from core.prescriptive import executive_summary, recommend

st.set_page_config(page_title=f"Prescriptive · {APP_NAME}", page_icon="💡", layout="wide")

try:
    llm.bridge_secrets(st.secrets)
except Exception:
    pass

st.title("💡 Prescriptive — what should we do?")

data = st.session_state.get("data")
if data is None:
    st.info("Pop over to 📄 Upload first and load some data — then I can advise. 💛")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📄")
    st.stop()

IMPACT_CHIP = {"High": "🔴 High impact", "Medium": "🟡 Medium impact", "Low": "⚪ Low impact"}


@st.cache_data(show_spinner=False)
def _segments(df):
    from core.diagnostic import compute_rfm, has_customers, segment_customers, segment_summary
    if not has_customers(df):
        return None
    return segment_summary(segment_customers(compute_rfm(df)))


@st.cache_data(show_spinner=False)
def _anomalies(df):
    from core.diagnostic import detect_anomalies
    return detect_anomalies(df).anomalies


@st.cache_data(show_spinner="Looking ahead…")
def _forecast(df):
    from core.forecasting import ForecastError, run_forecast
    try:
        return run_forecast(df)
    except ForecastError:
        return None


with st.spinner("Reading everything the other layers found… ✨"):
    insight = build_insight(data, forecast=_forecast(data),
                            segments=_segments(data), anomalies=_anomalies(data))
    st.session_state["insight"] = insight

actions = recommend(insight)

# --- ranked action cards ---------------------------------------------------
if not actions:
    st.success("Honestly? Things look healthy — no urgent actions jumped out. 💛")
else:
    st.caption("Ranked by impact. Each one shows the finding it's based on — no black box.")
    for i, a in enumerate(actions, 1):
        with st.container(border=True):
            top = st.columns([5, 1])
            top[0].markdown(f"**{i}. {a.title}**")
            top[1].markdown(IMPACT_CHIP[a.impact])
            st.write(a.why)
            st.caption(f"📎 Based on: `{a.evidence}`")

st.divider()

# --- optional LLM executive summary (behind a button, cached in session) ---
st.subheader("Executive summary")
if llm.available():
    if st.button("✨ Write me a summary", type="primary"):
        with st.spinner("Thinking it through…"):
            st.session_state["exec_summary"] = executive_summary(insight)
    summary = st.session_state.get("exec_summary")
    if summary:
        st.info(summary)
    elif "exec_summary" in st.session_state:
        st.caption("The summary came back empty — the action cards above have you covered. 💛")
else:
    st.caption("Add a Groq/OpenAI key in secrets to get a warm written summary here. "
               "The evidence-based actions above work without it. 💛")

# --- the InsightJSON spine -------------------------------------------------
with st.expander("See the findings this is built from (InsightJSON)"):
    st.json(insight.model_dump())

st.divider()
st.page_link("pages/6_Ask_Your_Data.py", label="Next: ask your data anything →", icon="💬")
