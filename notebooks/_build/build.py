"""Build + execute the DWM Colab notebook.

Run from the project root:   .venv/bin/python notebooks/_build/build.py
Produces notebooks/DWM_Sales_Analytics.ipynb with all outputs embedded, so it can be
read straight away and re-run top-to-bottom in Google Colab.
"""
import io
import os
import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd
import requests
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent
sys.path.insert(0, str(HERE))

import s1_intro_prep, s2_warehouse_eda, s3_models, s4_mining_wrap  # noqa: E402

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population"
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "District of Columbia",
]


def scrape_fallback_csv() -> str:
    """Scrape once at build time so the notebook can embed a safety-net copy."""
    html = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (DWM college project)"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    t.columns = [" | ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    sc = next(c for c in t.columns if c.startswith("State or territory"))
    pc = next(c for c in t.columns if "April 1, 2020" in c)
    d = t[[sc, pc]].copy()
    d.columns = ["state", "population_2020"]
    d["state"] = d["state"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    d["population_2020"] = pd.to_numeric(d["population_2020"].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
    d = d[d["state"].isin(US_STATES)].dropna().astype({"population_2020": int})
    assert len(d) == 51, f"expected 51 rows, got {len(d)}"
    return d.to_csv(index=False).strip()


cells = []


def md(text: str):
    # Colab/Jupyter treat $...$ as LaTeX, which would mangle money amounts — escape every dollar sign.
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n").replace("$", r"\$")))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


s1_intro_prep.add(md, code, scrape_fallback_csv())
s2_warehouse_eda.add(md, code)
s3_models.add(md, code)
s4_mining_wrap.add(md, code)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True},
}
print(f"Built {len(cells)} cells ({sum(c.cell_type == 'code' for c in cells)} code). Executing…")

out_path = Path(os.environ.get("NB_OUT", OUT_DIR / "DWM_Sales_Analytics.ipynb"))
ep = ExecutePreprocessor(timeout=1500, kernel_name="python3")
try:
    ep.preprocess(nb, {"metadata": {"path": str(OUT_DIR)}})
    status = "OK"
except CellExecutionError as exc:
    status = "FAILED"
    print(str(exc)[-3000:])
finally:
    nbf.write(nb, out_path)

print(f"\n{status} — wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
sys.exit(0 if status == "OK" else 1)
