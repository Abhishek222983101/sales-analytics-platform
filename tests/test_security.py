"""Security checks for the ingestion surface — the only untrusted-input path so far.

Covers: (1) no code-execution sinks in core, (2) CSV/formula-injection payloads
are preserved as inert text (never evaluated), (3) malformed input fails kindly
without leaking a traceback.
"""
from pathlib import Path

import pandas as pd

from core.ingestion import build_from_mapping

CORE = Path(__file__).resolve().parent.parent / "core"


def test_no_dangerous_sinks_in_core():
    banned = ("eval(", "exec(", "os.system", "subprocess", "pickle.load", "__import__", "os.popen")
    for py in CORE.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in src, f"{py.name} contains banned sink {token!r}"


def test_formula_injection_cells_stay_inert_text():
    # Classic CSV-injection payloads must survive verbatim, never be evaluated.
    raw = pd.DataFrame(
        {
            "Order Date": ["08/11/2017", "09/11/2017"],
            "Sales": [10.0, 20.0],
            "Category": ["=cmd|'/c calc'!A1", "@SUM(1+9)*cmd"],
        }
    )
    res = build_from_mapping(
        raw, {"date": "Order Date", "revenue": "Sales", "category": "Category"}
    )
    assert res.ok
    cats = list(res.canonical_df["category"])
    assert cats[0].startswith("=cmd")     # preserved, not executed
    assert cats[1].startswith("@SUM")


def test_malformed_input_fails_gracefully():
    raw = pd.DataFrame({"Order Date": ["nope", "nah"], "Sales": ["x", "y"]})
    res = build_from_mapping(raw, {"date": "Order Date", "revenue": "Sales"})
    assert not res.ok
    assert res.errors                      # a kind message, not a raised traceback


def test_llm_context_excludes_raw_customer_ids():
    # The Ask-your-data context sent to the LLM must be aggregates only — no per-
    # customer PII should ever leave the app.
    from core.ingestion import load_demo
    from core.nlp import data_context

    data = load_demo().canonical_df
    ctx = data_context(data)
    for cid in data["customer_id"].dropna().unique()[:10]:
        assert str(cid) not in ctx
