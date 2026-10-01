# DWM project notebook

**`DWM_Sales_Analytics.ipynb`** — the complete submission: web scraping, pre-processing, star schema + OLAP,
EDA, classification, regression, clustering, Apriori. Outputs are already embedded, so it can be read as-is.
`DWM_Sales_Analytics.html` is the same thing as a web page (no Colab needed to read it).

## Run it in Google Colab
1. Go to <https://colab.research.google.com> → **File ▸ Upload notebook** → pick `DWM_Sales_Analytics.ipynb`
2. **Runtime ▸ Run all** (≈ 3–5 minutes). The dataset downloads itself and the scrape runs live.

## Rebuild it (developers)
```bash
pip install -r requirements-notebook.txt
python notebooks/_build/build.py      # regenerates + executes the notebook, then check for errors
```
The cells live in `notebooks/_build/s1…s4_*.py`; `build.py` assembles and runs them.
