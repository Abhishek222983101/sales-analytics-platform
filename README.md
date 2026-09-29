# 💛 Lumi — Intelligent Sales Analytics Platform

A gentle, self-service web app that takes any sales CSV and walks a non-technical
user through all four analytics layers — **Descriptive → Diagnostic → Predictive →
Prescriptive** — in one interactive dashboard, ending with an AI that turns the
models' output into ranked, **evidence-cited** recommendations.

Final-year prototype · dataset: [Superstore Sales (Kaggle)](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) · single dataset, finance/revenue framing.

## ✨ The one-line innovation
An LLM prescriptive layer that reasons over the **combined** output of all four
models (forecast delta + anomaly + segment + SHAP driver, assembled into one
`InsightJSON`) and emits each recommendation with a traceable "why" — *explainable
recommendations, not just explainable predictions*. It works **with no API key**
(deterministic rule-based actions); a Groq/OpenAI key adds a warm written summary
and free-form "Ask your data" chat.

## What each layer does
| Layer | Answers | How |
|---|---|---|
| 🌸 Descriptive | What happened? | Revenue KPIs, trend, breakdowns, Pareto, cross-filtering |
| 🔍 Diagnostic | Why? | RFM → KMeans segments (Champions…Lost), IsolationForest anomalies |
| 🔮 Predictive | What's next? | XGBoost vs SARIMA monthly forecast, backtest, what-if, SHAP |
| 💡 Prescriptive | What to do? | Ranked, evidence-cited actions + optional LLM summary |
| 💬 Ask your data | Anything | Grounded LLM chat over aggregates (no code execution) |

## Stack
Streamlit · pandas · Plotly · scikit-learn · XGBoost · statsmodels (SARIMA) · SHAP ·
Groq (`openai/gpt-oss-120b`) with a rule-based fallback. Python 3.11.

## Project structure
```
app.py                 landing page
pages/                 one Streamlit page per layer (1_Upload … 6_Ask_Your_Data)
core/                  pure-Python services (no Streamlit — unit-tested)
  schema_mapper.py     auto-map any CSV → canonical schema
  ingestion.py         load / validate / clean
  descriptive.py       KPIs, series, breakdowns, Pareto
  forecasting.py       XGBoost + SARIMA + SHAP
  diagnostic.py        RFM/KMeans + IsolationForest
  insight_bus.py       assembles the InsightJSON spine
  prescriptive.py      evidence-cited actions + LLM summary
  nlp.py               Ask-your-data (grounded, safe)
  llm.py               provider-agnostic adapter + fallback
data/                  bundled Superstore sample
tests/                 48 tests (unit + headless render + security)
```

## Run locally
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: echo 'LLM_API_KEY="gsk_..."' >> .streamlit/secrets.toml   (git-ignored)
streamlit run app.py
```
Then click **✨ Use demo data** — no upload needed to explore.

## Test
```bash
pip install -r requirements-dev.txt
pytest -q          # 48 tests, all green
```

## Deploy

**Recommended — Streamlit Community Cloud (free, 1 GB):**
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch, and `app.py`.
3. In **Advanced → Secrets**, paste:
   ```toml
   LLM_PROVIDER = "groq"
   LLM_MODEL = "openai/gpt-oss-120b"
   LLM_API_KEY = "gsk_..."
   ```
4. Deploy. Every `git push` redeploys.

**Alternative — Render** (`render.yaml` blueprint is included): Render dashboard →
**New → Blueprint** → pick this repo → set `LLM_API_KEY` as a secret. Note the free
plan's 512 MB RAM is tight for this ML stack; prefer Streamlit Cloud if it OOMs.

> Not Vercel: it can't run a long-lived Python/ML process (Streamlit + XGBoost +
> SARIMA + SHAP). Vercel is for JS front-ends only.

## Notes for the report
See `SYSTEM-ARCHITECTURE.md` (design + the InsightJSON contract + synopsis edits)
and `BUILD-PLAN.md` (the phase-by-phase build). `DATA_PROFILE.md` documents the
dataset and confirms the revenue-only constraint.
