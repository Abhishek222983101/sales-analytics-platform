"""Map an arbitrary uploaded sales CSV onto the platform's canonical schema.

The canonical schema is the single internal contract every downstream layer
(descriptive, diagnostic, predictive, prescriptive) reads from. Mapping uses
(1) fuzzy name matching against a synonym dictionary, then (2) dtype heuristics
for the two required fields (date, revenue). Pure functions only — no Streamlit,
so this stays fully unit-testable.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz


# --------------------------------------------------------------------------- #
# Canonical schema definition                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanonicalField:
    name: str
    kind: str            # "datetime" | "numeric" | "text"
    required: bool
    synonyms: tuple[str, ...]
    description: str


CANONICAL_SCHEMA: tuple[CanonicalField, ...] = (
    CanonicalField(
        "date", "datetime", True,
        ("order date", "date", "invoice date", "transaction date",
         "purchase date", "sale date", "order dt", "created at"),
        "When the order was placed",
    ),
    CanonicalField(
        "revenue", "numeric", True,
        ("sales", "revenue", "amount", "total", "gmv", "sale amount",
         "net sales", "total sales", "sales amount", "order value", "turnover"),
        "Money value of the order line",
    ),
    CanonicalField(
        "order_id", "text", False,
        ("order id", "order number", "order no", "invoice id", "invoice no",
         "invoice number", "bill no", "receipt no"),
        "Identifier grouping the lines of one order",
    ),
    CanonicalField(
        "customer_id", "text", False,
        ("customer id", "customer", "cust id", "client id", "buyer id",
         "customer name", "client", "account id"),
        "Identifier for the buyer (enables RFM segmentation)",
    ),
    CanonicalField(
        "category", "text", False,
        ("category", "product category", "department", "product type"),
        "Top-level product grouping",
    ),
    CanonicalField(
        "sub_category", "text", False,
        ("sub-category", "subcategory", "sub category", "subcat"),
        "Second-level product grouping",
    ),
    CanonicalField(
        "region", "text", False,
        ("region", "zone", "area", "territory", "market"),
        "Geographic grouping",
    ),
    CanonicalField(
        "segment", "text", False,
        ("segment", "customer segment", "client segment", "buyer segment"),
        "Customer segment",
    ),
    CanonicalField(
        "product_id", "text", False,
        ("product id", "sku", "item id", "item code", "product code"),
        "Product identifier",
    ),
    CanonicalField(
        "product_name", "text", False,
        ("product name", "item name", "item", "product", "description"),
        "Human-readable product name",
    ),
    CanonicalField(
        "state", "text", False,
        ("state", "province"),
        "State / province",
    ),
    CanonicalField(
        "city", "text", False,
        ("city", "town"),
        "City",
    ),
)

FIELDS_BY_NAME: dict[str, CanonicalField] = {f.name: f for f in CANONICAL_SCHEMA}
REQUIRED_FIELDS: tuple[str, ...] = tuple(f.name for f in CANONICAL_SCHEMA if f.required)
OPTIONAL_FIELDS: tuple[str, ...] = tuple(f.name for f in CANONICAL_SCHEMA if not f.required)

NAME_ACCEPT = 72.0       # minimum fuzzy score to accept a name-based match
NAME_STRONG = 90.0       # score treated as a confident match


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _norm(name: object) -> str:
    """Normalise a column name for comparison: lowercase, unpunctuated, single-spaced."""
    s = str(name).strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion tolerating currency symbols, commas and (parens)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype("string")
        .str.replace(r"[,$₹€£%\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (123) -> -123
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _looks_like_id(norm_name: str) -> bool:
    return any(tok in norm_name.split() for tok in
               ("id", "postal", "zip", "row", "code", "year", "phone", "no"))


def _score_name(field: CanonicalField, norm_name: str) -> float:
    """Fuzzy score in [0, 100] for how well a normalised column name fits a field."""
    best = 0.0
    for syn in field.synonyms:
        if norm_name == syn:
            return 100.0
        if re.search(rf"\b{re.escape(syn)}\b", norm_name):
            best = max(best, 90.0)
        best = max(best, float(fuzz.token_sort_ratio(norm_name, syn)))
    return best


# --------------------------------------------------------------------------- #
# Mapping results                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class FieldMatch:
    canonical: str
    source: Optional[str]
    score: float
    method: str          # "name" | "dtype" | "user" | "none"

    @property
    def confidence(self) -> str:
        if self.source is None:
            return "none"
        if self.method == "user":
            return "manual"
        if self.score >= NAME_STRONG:
            return "high"
        if self.score >= NAME_ACCEPT:
            return "medium"
        return "low"


@dataclass
class MappingResult:
    matches: list[FieldMatch]
    unmapped_sources: list[str]

    @property
    def mapping(self) -> dict[str, Optional[str]]:
        return {m.canonical: m.source for m in self.matches}

    def source_for(self, canonical: str) -> Optional[str]:
        return self.mapping.get(canonical)

    def match_for(self, canonical: str) -> Optional[FieldMatch]:
        return next((m for m in self.matches if m.canonical == canonical), None)

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not self.mapping.get(f)]


# --------------------------------------------------------------------------- #
# Dtype heuristics (fallback for required fields)                              #
# --------------------------------------------------------------------------- #
def _detect_date_column(df: pd.DataFrame, exclude: set, sample: int = 250) -> Optional[str]:
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col].dropna().astype("string").head(sample)
        if s.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if float(parsed.notna().mean()) >= 0.8:
            return col
    return None


def _detect_revenue_column(df: pd.DataFrame, exclude: set) -> Optional[str]:
    best, best_sum = None, -1.0
    for col in df.columns:
        if col in exclude or _looks_like_id(_norm(col)):
            continue
        vals = coerce_numeric(df[col])
        if float(vals.notna().mean()) < 0.8:
            continue
        total = float(vals.abs().sum())
        if total > best_sum:
            best, best_sum = col, total
    return best


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def suggest_mapping(df: pd.DataFrame) -> MappingResult:
    """Suggest a canonical -> source-column mapping for a raw dataframe."""
    sources = list(df.columns)
    norms = {c: _norm(c) for c in sources}
    used: set = set()
    chosen: dict[str, FieldMatch] = {}

    # 1) name matching, in schema priority order (each source used at most once)
    for f in CANONICAL_SCHEMA:
        best_src, best_score = None, 0.0
        for c in sources:
            if c in used:
                continue
            sc = _score_name(f, norms[c])
            if sc > best_score:
                best_src, best_score = c, sc
        if best_src is not None and best_score >= NAME_ACCEPT:
            chosen[f.name] = FieldMatch(f.name, best_src, best_score, "name")
            used.add(best_src)
        else:
            chosen[f.name] = FieldMatch(f.name, None, 0.0, "none")

    # 2) dtype fallback for the two required fields
    if chosen["date"].source is None:
        src = _detect_date_column(df, used)
        if src:
            chosen["date"] = FieldMatch("date", src, 60.0, "dtype")
            used.add(src)
    if chosen["revenue"].source is None:
        src = _detect_revenue_column(df, used)
        if src:
            chosen["revenue"] = FieldMatch("revenue", src, 60.0, "dtype")
            used.add(src)

    matches = [chosen[f.name] for f in CANONICAL_SCHEMA]
    unmapped = [c for c in sources if c not in used]
    return MappingResult(matches, unmapped)


def apply_mapping(df: pd.DataFrame, mapping: dict[str, Optional[str]]) -> pd.DataFrame:
    """Project a raw dataframe onto canonical columns (type coercion happens later)."""
    out = pd.DataFrame(index=df.index)
    for canonical, source in mapping.items():
        if source and source in df.columns:
            out[canonical] = df[source].values
    return out
