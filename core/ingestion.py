"""Load, validate, and clean an uploaded sales file into the canonical schema.

Pipeline: read CSV -> auto-map columns -> validate required fields -> clean
(parse day-first dates, coerce revenue, drop unusable rows, de-duplicate, sort).
Returns an ``IngestResult`` carrying the cleaned dataframe plus a list of gentle,
human-readable ``Issue`` messages the UI can show. Pure Python — no Streamlit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from core.schema_mapper import (
    MappingResult,
    REQUIRED_FIELDS,
    apply_mapping,
    coerce_numeric,
    suggest_mapping,
)

DEMO_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_superstore.csv"
_TEXT_DTYPE = "string"


@dataclass
class Issue:
    level: str            # "error" | "warning" | "info"
    message: str


@dataclass
class IngestResult:
    ok: bool
    raw_df: Optional[pd.DataFrame]
    canonical_df: Optional[pd.DataFrame]
    mapping: Optional[MappingResult]
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "info"]


def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like buffer, with an encoding fallback."""
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="latin-1")


def clean_canonical(cdf: pd.DataFrame, issues: list[Issue]) -> pd.DataFrame:
    """Coerce types, drop unusable rows, de-duplicate and sort a canonical frame."""
    df = cdf.copy()

    if "date" in df.columns:
        # We intentionally accept mixed/day-first formats across arbitrary uploads;
        # silence pandas' "could not infer format" nudge to keep deploy logs clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    if "revenue" in df.columns:
        df["revenue"] = coerce_numeric(df["revenue"])
    for col in df.columns:
        if col not in ("date", "revenue"):
            df[col] = df[col].astype(_TEXT_DTYPE).str.strip()

    required_here = [c for c in REQUIRED_FIELDS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required_here)
    dropped = before - len(df)
    if dropped:
        issues.append(Issue(
            "warning",
            f"{dropped:,} row(s) had a missing or unreadable date/revenue — "
            f"I gently set those aside so they don't skew the numbers.",
        ))

    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates()
        issues.append(Issue("info", f"Tidied up {dups:,} exact duplicate row(s)."))

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _summary_issue(df: pd.DataFrame) -> Issue:
    span = f"{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}"
    return Issue("info", f"All set — {len(df):,} clean rows spanning {span}. 💛")


def ingest(source) -> IngestResult:
    """Full pipeline: read -> auto-map -> validate -> clean."""
    issues: list[Issue] = []
    try:
        raw = load_csv(source)
    except Exception as exc:  # noqa: BLE001 - surface any read failure kindly
        return IngestResult(False, None, None, None,
                            [Issue("error", f"I couldn't read that file — {exc}")])

    if raw is None or raw.empty:
        return IngestResult(False, raw, None, None,
                            [Issue("error", "That file looks empty — mind checking it?")])

    mapping = suggest_mapping(raw)
    missing = mapping.missing_required()
    if missing:
        issues.append(Issue(
            "error",
            f"I couldn't confidently find a column for: {', '.join(missing)}. "
            f"No worries — you can point me to it on the upload page.",
        ))
        return IngestResult(False, raw, None, mapping, issues)

    canonical = clean_canonical(apply_mapping(raw, mapping.mapping), issues)
    if canonical.empty:
        issues.append(Issue("error", "After cleaning, no usable rows were left."))
        return IngestResult(False, raw, canonical, mapping, issues)

    issues.append(_summary_issue(canonical))
    return IngestResult(True, raw, canonical, mapping, issues)


def build_from_mapping(raw: pd.DataFrame, mapping: dict) -> IngestResult:
    """Apply a (possibly user-corrected) mapping dict and clean."""
    issues: list[Issue] = []
    missing = [f for f in REQUIRED_FIELDS if not mapping.get(f)]
    if missing:
        return IngestResult(False, raw, None, None,
                            [Issue("error", f"Please choose a column for: {', '.join(missing)}.")])

    canonical = clean_canonical(apply_mapping(raw, mapping), issues)
    ok = not canonical.empty
    if not ok:
        issues.append(Issue("error", "After cleaning, no usable rows were left."))
    else:
        issues.append(_summary_issue(canonical))
    return IngestResult(ok, raw, canonical, None, issues)


def load_demo() -> IngestResult:
    """Ingest the bundled Superstore sample so the app works with zero upload."""
    return ingest(DEMO_PATH)
