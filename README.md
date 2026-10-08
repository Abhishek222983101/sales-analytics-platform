# Sales Analytics — DWM Course Project + Live Web App

Data Warehousing & Mining project on the **Superstore Sales** dataset
([Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) · 9,800 order lines · 2015–2018).

| | What | Where |
|---|---|---|
| 📓 **Main deliverable** | A single Colab notebook: web scraping, pre-processing, star schema + OLAP, EDA, classification, regression, clustering, Apriori | [`dwm-project/`](./dwm-project) |
| 🌐 **Bonus** | "Lumi" — the same ideas as a live, interactive web app with AI-written recommendations | **https://lumi-sales-analytics.onrender.com** *(free hosting — first load after idle can take ~1 minute)* |

## Repository map
```
dwm-project/        ← THE GRADED WORK: notebook (.ipynb + .html), its README and build scripts
docs/               ← design notes: SYSTEM-ARCHITECTURE.md, BUILD-PLAN.md, DATA_PROFILE.md
app.py, pages/      ← the web app (Streamlit); must stay at the repo root for deployment
core/               ← pure-Python services behind the app (unit-tested, no Streamlit)
tests/              ← 48 tests: unit + headless page rendering + security
data/               ← the bundled Superstore sample
scripts/            ← helper scripts (data profiling)
render.yaml, requirements.txt, runtime.txt, .streamlit/   ← deployment config
```

## 📓 The DWM notebook
Start with **[`dwm-project/README.md`](./dwm-project/README.md)** — it lists every section, how to open the notebook in Google Colab,
and the headline results. The notebook already contains its outputs, so it can also be read straight from
`dwm-project/DWM_Sales_Analytics.html` in any browser.

## 🌐 The web app (Lumi)
A gentle, self-service app that takes any sales CSV and walks through **Descriptive → Diagnostic → Predictive → Prescriptive**
analytics, ending with ranked recommendations that each cite the finding behind them (they work with no API key; a Groq key adds
a written summary and free-form "Ask your data" chat).

| Page | Does |
|---|---|
| 📄 Upload | Auto-maps columns of any sales CSV, or loads the demo data |
| 🌸 Descriptive | KPIs, trends, breakdowns, Pareto, cross-filtering |
| 🔍 Diagnostic | RFM → K-Means customer segments, Isolation-Forest anomalies |
| 🔮 Predictive | XGBoost vs SARIMA monthly forecast, backtest, what-if |
| 💡 Prescriptive | Ranked, evidence-cited actions + optional AI summary |
| 💬 Ask your data | Chat grounded on pre-computed aggregates (no code execution) |

Stack: Streamlit · pandas · Plotly · scikit-learn · XGBoost · statsmodels · Groq (`openai/gpt-oss-120b`) with a rule-based fallback. Python 3.11.

### Run the app locally
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: add LLM_API_KEY="gsk_..." to .streamlit/secrets.toml (git-ignored)
streamlit run app.py
```

### Test
```bash
pip install -r requirements-dev.txt
pytest -q          # 48 tests
```

### Deploy
Deployed on **Render** (free plan) from `render.yaml`; every push to `main` redeploys. Set `LLM_API_KEY` as a secret in the Render
dashboard. The free plan has 512 MB RAM — the Predictive page peaks around 475 MB, so avoid adding heavy dependencies (`shap` was removed for this reason).
Streamlit Community Cloud also works (1 GB). Vercel cannot host this app: it is a long-running Python/ML process.

## Docs
* [`docs/SYSTEM-ARCHITECTURE.md`](./docs/SYSTEM-ARCHITECTURE.md) — design, the InsightJSON contract, deployment options
* [`docs/BUILD-PLAN.md`](./docs/BUILD-PLAN.md) — the phase-by-phase build plan for the app
* [`docs/DATA_PROFILE.md`](./docs/DATA_PROFILE.md) — dataset profile; confirms `Sales` is the only numeric measure
