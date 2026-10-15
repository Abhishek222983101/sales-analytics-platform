# DWM Course Project — Sales Analytics with Data Warehousing & Mining

**This folder is the graded submission.** One self-contained notebook that covers the whole course pipeline on the
Superstore Sales dataset (9,800 order lines, 2015–2018).

| File | What it is |
|---|---|
| **`DWM_Sales_Analytics.ipynb`** | The notebook — outputs are already embedded, so it can be read as-is |
| `DWM_Sales_Analytics.html` | The same notebook as a web page (open in any browser, no Colab needed) |
| `build/` | Source cells + the script that generates and executes the notebook (not needed to read or run it) |
| `requirements.txt` | Only needed to rebuild locally; Colab already has everything (the notebook installs `mlxtend` itself) |

## What the notebook covers

| § | Section | Highlights |
|---|---|---|
| 1 | Data collection | Download, data dictionary, summary |
| 2 | **Web scraping** | Wikipedia state populations via `requests` + `pandas.read_html`, with an embedded fallback copy |
| 3 | **Pre-processing** | Audit, date format detection, imputation (11 ZIPs), duplicate, consistency checks, outliers (IQR), skew + log transform, encoding, scaling, before/after |
| 4 | **Star schema + OLAP** | 1 fact + 5 dimensions in SQLite with PK/FK, reconciliation checks, roll-up / drill-down / slice / dice / pivot |
| 5 | **EDA** | Univariate, trends, seasonality heat-map, Pareto, relationships, key findings, plus a **head-to-head comparison: Technology vs Office Supplies** (data trace → measures → charts → statistical tests) |
| 6 | **Classification** | The two classes defined (1 = high-value, 0 = standard — like spam / not spam), 6 algorithms, cross-validation, ROC, Random-Forest confusion matrix (counts and %), per-class precision/recall, real example predictions, feature importance, decision tree, grid search |
| 7 | **Regression** | 5 regressors vs a baseline, plus monthly revenue forecast (XGBoost vs SARIMA) |
| 8 | **Clustering** | RFM, elbow + silhouette + Davies–Bouldin, K-Means, hierarchical + dendrogram |
| 9 | **Apriori** | Support / confidence / lift, library + from-scratch implementation verified identical |
| 10 | Bonus | Anomaly detection (Isolation Forest) + link to the live web app |
| 11 | Conclusions | Results table, business insights, limitations, references |

## Run it in Google Colab
1. Open <https://colab.research.google.com> → **File ▸ Upload notebook** → choose `DWM_Sales_Analytics.ipynb`
2. **Runtime ▸ Run all** (about 3–5 minutes). The dataset downloads itself and the scrape runs live.
3. Use Colab's left sidebar (the "Table of contents" icon) to jump between sections.

## Headline results (from the executed notebook)
* **Pre-processing:** 9,800 → 9,799 rows; skew of `Sales` 12.98 → 0.28 after log transform; 1,145 outliers kept and flagged.
* **Star schema:** 9,799 fact rows reconcile exactly with the clean table ($2,261,255.41); 0 foreign-key violations.
* **Classification:** four algorithms tie at F1 ≈ 0.62 (ROC-AUC ≈ 0.85); `Sub-Category` is the dominant driver.
* **Regression:** R² ≈ 0.45 (limited because the data has no quantity or price); monthly forecast MAPE 19.6% (XGBoost) vs 29.1% (SARIMA).
* **Clustering:** k = 4 RFM segments (silhouette 0.26 — customers form a continuum, not sharp islands).
* **Apriori:** 21 rules at 1% support; maximum lift 1.25 — sub-categories are bought almost independently.

## Rebuild from source (developers)
```bash
pip install -r dwm-project/requirements.txt
python dwm-project/build/build.py          # regenerates and executes the notebook; exit code 0 = every cell ran
```
Set `NB_OUT=/some/path.ipynb` to write the result somewhere else instead of overwriting the committed notebook.
