"""Upload page: bring a sales CSV (or use demo data), review the auto-detected
column mapping, correct anything, and load it into the session for every layer.
"""
from io import BytesIO

import pandas as pd
import streamlit as st

from core.branding import APP_NAME, cheer, oops
from core.ingestion import build_from_mapping, load_csv, load_demo
from core.schema_mapper import CANONICAL_SCHEMA, REQUIRED_FIELDS, suggest_mapping

st.set_page_config(page_title=f"Upload · {APP_NAME}", page_icon="📄", layout="wide")

st.title("📄 Bring your sales data")
st.write(
    "Drop a CSV and I'll figure out the columns for you. "
    "It doesn't need to be perfectly formatted — promise. 💛"
)

NONE_LABEL = "— none —"


@st.cache_data(show_spinner=False)
def _read_raw(file_bytes: bytes) -> pd.DataFrame:
    return load_csv(BytesIO(file_bytes))


def _mapping_editor(raw: pd.DataFrame) -> dict:
    """Render a selectbox per canonical field; return the chosen mapping dict."""
    suggestion = suggest_mapping(raw)
    options = [NONE_LABEL] + list(raw.columns)
    chosen: dict = {}

    st.subheader("Column mapping")
    st.caption(
        "I've guessed these — change any that look off. "
        "Fields marked \\* are the only ones I truly need."
    )

    grid = st.columns(3)
    for i, fdef in enumerate(CANONICAL_SCHEMA):
        match = suggestion.match_for(fdef.name)
        default = match.source if (match and match.source in raw.columns) else None
        index = options.index(default) if default in options else 0
        star = " \\*" if fdef.required else ""
        with grid[i % 3]:
            pick = st.selectbox(
                f"{fdef.name}{star}",
                options,
                index=index,
                help=fdef.description,
                key=f"map_{fdef.name}",
            )
            chosen[fdef.name] = None if pick == NONE_LABEL else pick
    return chosen


def _load_into_session(canonical_df, mapping) -> None:
    st.session_state["data"] = canonical_df
    st.session_state["mapping"] = mapping
    # any stale downstream results are invalidated by a fresh load
    st.session_state.pop("insight", None)


# --- input row -------------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    up = st.file_uploader("Upload a sales CSV", type=["csv"], label_visibility="collapsed")
with c2:
    use_demo = st.button(
        "✨ Use demo data",
        use_container_width=True,
        help="Explore with a real Superstore dataset — no file needed.",
    )

# --- demo path -------------------------------------------------------------
if use_demo:
    demo = load_demo()
    if demo.ok:
        _load_into_session(demo.canonical_df, demo.mapping.mapping if demo.mapping else None)
        st.success(cheer("demo data loaded — have a wander!"))
        for issue in demo.infos:
            st.caption(issue.message)
        st.dataframe(demo.canonical_df.head(15), use_container_width=True)
        st.page_link("pages/2_Descriptive.py", label="See what happened →", icon="🌸")
    else:
        for issue in demo.errors:
            st.error(oops(issue.message))

# --- uploaded-file path ----------------------------------------------------
raw_df = None
if up is not None:
    try:
        raw_df = _read_raw(up.getvalue())
    except Exception as exc:  # noqa: BLE001 - be kind about any read failure
        st.error(oops(f"I couldn't read that file — {exc}."))

if raw_df is not None:
    st.success(f"Read **{len(raw_df):,}** rows and **{raw_df.shape[1]}** columns.")
    with st.expander("Preview raw file", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True)

    mapping = _mapping_editor(raw_df)
    missing = [f for f in REQUIRED_FIELDS if not mapping.get(f)]
    if missing:
        st.warning(
            f"I still need a column for: **{', '.join(missing)}**. "
            f"Pick it above and we're good to go."
        )

    if st.button("Looks good — load it 💛", type="primary", disabled=bool(missing)):
        result = build_from_mapping(raw_df, mapping)
        if not result.ok:
            for issue in result.errors:
                st.error(oops(issue.message))
        else:
            _load_into_session(result.canonical_df, mapping)
            for issue in result.warnings:
                st.warning(issue.message)
            for issue in result.infos:
                st.info(issue.message)
            st.success(cheer("your data's loaded and tidy!"))
            st.dataframe(result.canonical_df.head(15), use_container_width=True)
            st.page_link("pages/2_Descriptive.py", label="See what happened →", icon="🌸")

# --- already-loaded hint ---------------------------------------------------
if raw_df is None and not use_demo and st.session_state.get("data") is not None:
    n = len(st.session_state["data"])
    st.info(f"You've got {n:,} rows loaded already — head to any layer in the sidebar, "
            f"or upload something new above. ✨")
