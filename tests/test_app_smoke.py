"""Headless render smoke tests — every page must run without raising.

Pages are driven through the real entrypoint (``app.py``) and ``switch_page``,
which mirrors how a user navigates and gives ``st.page_link`` its nav context.
This guards against pages that import-compile fine but blow up at runtime.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.ingestion import load_demo

ROOT = Path(__file__).resolve().parent.parent


def _demo_df():
    return load_demo().canonical_df


def _entry() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"))


def test_landing_runs():
    at = _entry().run(timeout=30)
    assert not at.exception


def test_upload_page_runs():
    at = _entry()
    at.run(timeout=30)
    at.switch_page("pages/1_Upload.py")
    at.run(timeout=30)
    assert not at.exception


def test_descriptive_runs_with_data():
    at = _entry()
    at.session_state["data"] = _demo_df()
    at.run(timeout=30)
    at.switch_page("pages/2_Descriptive.py")
    at.run(timeout=60)
    assert not at.exception


def test_descriptive_prompts_without_data():
    at = _entry()
    at.run(timeout=30)
    at.switch_page("pages/2_Descriptive.py")
    at.run(timeout=30)
    assert not at.exception   # shows the "upload first" prompt, no crash


def test_diagnostic_runs_with_data():
    at = _entry()
    at.session_state["data"] = _demo_df()
    at.run(timeout=30)
    at.switch_page("pages/3_Diagnostic.py")
    at.run(timeout=60)
    assert not at.exception


def test_predictive_runs_with_data():
    at = _entry()
    at.session_state["data"] = _demo_df()
    at.run(timeout=30)
    at.switch_page("pages/4_Predictive.py")
    at.run(timeout=120)          # trains XGBoost + SARIMA on first run
    assert not at.exception


def test_prescriptive_runs_with_data():
    at = _entry()
    at.session_state["data"] = _demo_df()
    at.run(timeout=30)
    at.switch_page("pages/5_Prescriptive.py")
    at.run(timeout=120)          # builds InsightJSON; LLM summary is behind a button
    assert not at.exception


def test_ask_page_runs_with_data():
    at = _entry()
    at.session_state["data"] = _demo_df()
    at.run(timeout=30)
    at.switch_page("pages/6_Ask_Your_Data.py")
    at.run(timeout=60)           # quick facts render; no question submitted
    assert not at.exception


def test_stub_pages_run():
    for page in ("3_Diagnostic", "4_Predictive", "5_Prescriptive", "6_Ask_Your_Data"):
        at = _entry()
        at.run(timeout=30)
        at.switch_page(f"pages/{page}.py")   # 4 without data just shows the prompt
        at.run(timeout=30)
        assert not at.exception, f"{page} raised"
