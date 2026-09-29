"""Lumi — Intelligent Sales Analytics Platform.

Landing page. The real work lives in the numbered pages in ``pages/`` (each one
analytics layer) and the pure-Python services in ``core/``.
"""
import streamlit as st

from core.branding import APP_EMOJI, APP_NAME, APP_SUBTITLE, APP_TAGLINE, LAYERS

st.set_page_config(
    page_title=f"{APP_NAME} — Sales Analytics",
    page_icon=APP_EMOJI,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"{APP_EMOJI} {APP_NAME}")
st.markdown(f"#### {APP_TAGLINE}")
st.caption(APP_SUBTITLE)

st.write("")
st.markdown(
    "Upload a sales spreadsheet and I'll walk you through **what happened**, **why**, "
    "**what's coming**, and **what to do about it** — gently, and in plain English. "
    "No data-science degree required. 💛"
)

st.write("")
cols = st.columns(4)
for col, (title, question, blurb) in zip(cols, LAYERS):
    with col:
        st.markdown(f"##### {title}")
        st.caption(question)
        st.write(blurb)

st.write("")
st.divider()

left, right = st.columns([1, 2])
with left:
    st.page_link("pages/1_Upload.py", label="Start here — bring your data", icon="📄")
with right:
    st.caption(
        "New here? The demo dataset is one click away on the upload page — "
        "no file needed to explore."
    )

if st.session_state.get("data") is not None:
    n = len(st.session_state["data"])
    st.success(
        f"You've already loaded {n:,} rows this session — pick any layer from the sidebar. ✨"
    )
