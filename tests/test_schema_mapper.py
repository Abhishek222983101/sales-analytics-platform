"""Unit tests for the schema mapper — the generalisation claim lives or dies here."""
import pandas as pd

from core.schema_mapper import coerce_numeric, suggest_mapping


def _superstore_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Row ID": [1, 2, 3],
            "Order ID": ["CA-2017-1", "CA-2017-1", "CA-2017-2"],
            "Order Date": ["08/11/2017", "08/11/2017", "12/06/2017"],
            "Ship Date": ["11/11/2017", "11/11/2017", "16/06/2017"],
            "Ship Mode": ["Second Class", "Second Class", "First Class"],
            "Customer ID": ["CG-12520", "CG-12520", "DV-13045"],
            "Customer Name": ["Claire Gute", "Claire Gute", "Darrin Van"],
            "Segment": ["Consumer", "Consumer", "Corporate"],
            "Country": ["United States", "United States", "United States"],
            "City": ["Henderson", "Henderson", "Los Angeles"],
            "State": ["Kentucky", "Kentucky", "California"],
            "Postal Code": [42420, 42420, 90036],
            "Region": ["South", "South", "West"],
            "Product ID": ["FUR-BO-1", "FUR-CH-2", "OFF-LA-3"],
            "Category": ["Furniture", "Furniture", "Office Supplies"],
            "Sub-Category": ["Bookcases", "Chairs", "Labels"],
            "Product Name": ["Bookcase", "Stacking Chairs", "Labels"],
            "Sales": [261.96, 731.94, 14.62],
        }
    )


def test_superstore_maps_core_fields():
    m = suggest_mapping(_superstore_sample()).mapping
    assert m["date"] == "Order Date"
    assert m["revenue"] == "Sales"
    assert m["customer_id"] == "Customer ID"
    assert m["category"] == "Category"
    assert m["sub_category"] == "Sub-Category"
    assert m["region"] == "Region"
    assert m["segment"] == "Segment"
    assert m["order_id"] == "Order ID"


def test_date_does_not_grab_ship_date():
    # "Order Date" must win over "Ship Date" for the canonical date field.
    m = suggest_mapping(_superstore_sample()).mapping
    assert m["date"] == "Order Date"


def test_renamed_columns_still_map():
    df = _superstore_sample().rename(
        columns={"Sales": "amount", "Order Date": "invoice_date", "Customer ID": "client id"}
    )
    m = suggest_mapping(df).mapping
    assert m["revenue"] == "amount"
    assert m["date"] == "invoice_date"
    assert m["customer_id"] == "client id"


def test_dtype_fallback_for_unnamed_columns():
    df = pd.DataFrame(
        {
            "foo": ["08/11/2017", "09/11/2017", "10/11/2017"],
            "bar": [100.0, 200.0, 300.0],
            "baz": ["x", "y", "z"],
        }
    )
    m = suggest_mapping(df).mapping
    assert m["date"] == "foo"      # detected by parseability
    assert m["revenue"] == "bar"   # detected as the largest-sum numeric column


def test_coerce_numeric_handles_currency_and_parens():
    out = coerce_numeric(pd.Series(["$1,200.50", "(30)", "45"]))
    assert out.iloc[0] == 1200.50
    assert out.iloc[1] == -30
    assert out.iloc[2] == 45


def test_required_missing_is_reported():
    df = pd.DataFrame({"Order Date": ["08/11/2017"], "Notes": ["hello"]})
    result = suggest_mapping(df)
    assert "revenue" in result.missing_required()
