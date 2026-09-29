"""Unit tests for ingestion: cleaning, day-first dates, and graceful failures."""
import pandas as pd

from core.ingestion import build_from_mapping, load_demo


def test_demo_ingests_ok():
    res = load_demo()
    assert res.ok
    df = res.canonical_df
    assert {"date", "revenue"} <= set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["revenue"])
    assert len(df) > 9000
    assert df["date"].is_monotonic_increasing  # sorted ascending


def test_dayfirst_parsing():
    # 16/06/2017 must parse as 16 June (day-first), not month 16 (invalid).
    raw = pd.DataFrame({"Order Date": ["16/06/2017"], "Sales": [10.0]})
    res = build_from_mapping(raw, {"date": "Order Date", "revenue": "Sales"})
    assert res.ok
    d = res.canonical_df["date"].iloc[0]
    assert (d.day, d.month, d.year) == (16, 6, 2017)


def test_clean_drops_unparseable_rows():
    raw = pd.DataFrame(
        {
            "Order Date": ["08/11/2017", "not a date", "09/11/2017"],
            "Sales": ["100", "200", "oops"],
        }
    )
    res = build_from_mapping(raw, {"date": "Order Date", "revenue": "Sales"})
    assert res.ok
    assert len(res.canonical_df) == 1        # only the first row is fully valid
    assert res.warnings                      # and we warned about the drop


def test_build_from_mapping_missing_required():
    raw = pd.DataFrame({"Order Date": ["08/11/2017"], "Notes": ["hi"]})
    res = build_from_mapping(raw, {"date": "Order Date"})  # no revenue mapped
    assert not res.ok
    assert res.errors
