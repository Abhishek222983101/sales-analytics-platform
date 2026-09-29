# Build Plan (detailed) — Intelligent Sales Analytics Platform

Phase-by-phase **with sub-phases**. Pair with `SYSTEM-ARCHITECTURE.md` (the *what/why*); this is the *in what order + done-when*.

**Guiding principle:** always have a deployable, demoable app. Prove the deploy pipeline on Day 1 (Phase 0), then add one working layer per phase. If time runs out, only the last layer is at risk.

**Estimates** are focused-effort ranges, not calendar dates. Total ≈ **3–3.5 wks part-time / ~10 days full-time.** Each sub-phase block: **build · file(s) · done-when.**

**Progress key:** `[ ]` todo · `[~]` in progress · `[x]` done

---

## PHASE 0 — Foundations & prove-the-deploy  ·  ½–1 day
**Goal:** empty app live on a public URL + dataset profiled. De-risk deployment *before* any features.

- **0.1 Repo & environment** · `.git/`, folder tree
  Build: `git init`; create the structure from Arch §6; Python 3.11 venv.
  Done: `tree` matches the architecture; venv activates.
- **0.2 Dependencies & config** · `requirements.txt`, `runtime.txt`, `.gitignore`
  Build: pin deps (streamlit, pandas, numpy, plotly, scikit-learn, xgboost, statsmodels, shap, rapidfuzz, pydantic, python-dotenv, groq/openai, pytest); `runtime.txt`=python-3.11; gitignore secrets.
  Done: `pip install -r requirements.txt` succeeds clean.
- **0.3 Get & profile the data** · `data/sample_superstore.csv`, `DATA_PROFILE.md`
  Build: download the Kaggle Superstore CSV; run `.info()`, `.describe()`, `.isnull().sum()`, `.nunique()`; write results to `DATA_PROFILE.md`.
  Done: **column list confirmed** (only `Sales` numeric — no Profit/Qty/Discount); real row count + date range recorded.
- **0.4 Hello app** · `app.py`
  Build: landing page (title, one-line pitch), sidebar nav placeholder, a "Use demo data" stub button.
  Done: `streamlit run app.py` shows the landing page locally.
- **0.5 First deploy** · GitHub + Streamlit Cloud
  Build: push to GitHub → share.streamlit.io → New app → pick repo/`app.py` → deploy.
  Done: **public `*.streamlit.app` URL loads**, and a `git push` auto-redeploys.

✅ **Checkpoint:** live app skeleton on the internet.

---

## PHASE 1 — Ingestion + schema-mapper + canonical df  ·  1–2 days
**Goal:** upload *any* sales CSV → validated, cleaned, canonical dataframe + a visible mapping report. Works with zero upload via bundled demo data.

- **1.1 Ingestion module** · `core/ingestion.py`
  Build: `read()`, validate (≥1 date-parseable col, ≥1 numeric col), clean (parse dates, drop dupes, coerce dtypes, handle missing), return `(df, issues[])`.
  Done: loads Superstore **and** a deliberately messy CSV without crashing; returns an issues list.
- **1.2 Canonical schema + synonym dictionary** · `core/schema_mapper.py`
  Build: define canonical schema (Arch §3); synonym dict (`revenue`←sales/amount/gmv…, `date`←order date/invoice date…, etc.).
  Done: dict covers date/revenue/customer/order/category/region/segment.
- **1.3 Fuzzy name-match + dtype heuristics** · `core/schema_mapper.py`
  Build: `rapidfuzz` header matching; fallback dtype heuristics (first date-parseable→`date`; highest-sum numeric→`revenue`); output `mapping` + per-field `confidence`.
  Done: maps Superstore correctly **and** a CSV with `Sales`→`amount`, `Order Date`→`invoice_date`.
- **1.4 Upload page UI** · `pages/1_Upload.py`
  Build: file uploader + "Use demo data" button; show detected mapping in editable `st.selectbox`es (user override); preview `canonical_df.head()`; stash df + mapping in `st.session_state`.
  Done: user can upload or use demo, see + correct the mapping, and proceed.
- **1.5 Caching + tests** · `tests/test_schema_mapper.py`
  Build: `@st.cache_data` keyed by file bytes; pytest for the renamed-column CSV.
  Done: re-runs are instant; `pytest` green.

✅ **Checkpoint:** upload any sales CSV → it's correctly understood, with a mapping report.

---

## PHASE 2 — Descriptive layer  ·  2–3 days
**Goal:** a genuinely useful finance dashboard (do this before ML — early visible payoff).

- **2.1 KPI computations** · `core/descriptive.py`
  Build: total revenue, avg order value, MoM & YoY growth %, active customers, order count.
  Done: functions return correct values on the demo df (spot-check by hand).
- **2.2 Time-series aggregation** · `core/descriptive.py`
  Build: `revenue_series(df, grain)` for day/week/month.
  Done: three grains produce sensible, gap-free series.
- **2.3 Breakdowns** · `core/descriptive.py`
  Build: revenue by Region / Category / Sub-Category / Segment; top-N products.
  Done: each returns a sorted, aggregated frame.
- **2.4 Pareto / revenue concentration** · `core/descriptive.py`
  Build: 80/20 curve for customers & products → feeds InsightJSON `revenue_concentration`.
  Done: "top X% = Y% of revenue" computed.
- **2.5 Descriptive page UI** · `pages/2_Descriptive.py`
  Build: KPI cards (`st.metric`), trend chart with **grain toggle**, breakdown bar/treemap (Plotly), Pareto chart. Finance vocabulary throughout.
  Done: all charts render on demo data.
- **2.6 Cross-filtering** · `pages/2_Descriptive.py`
  Build: sidebar slicers (region/category/segment/date-range) that re-filter **every** chart + KPI.
  Done: changing a slicer updates the whole page; hides a slicer whose column is absent.

✅ **Checkpoint:** a Power BI–style, cross-filterable dashboard on live data.

---

## PHASE 3 — Predictive layer  ·  3–4 days  ·  *technical core*
**Goal:** benchmarked revenue forecast (XGBoost vs SARIMA) + CI + metrics + what-if + SHAP.

- **3.1 Series prep** · `core/forecasting.py`
  Build: aggregate to **weekly revenue**; fill missing weeks; optional slice (region/category).
  Done: continuous weekly series (~208 pts) produced.
- **3.2 Feature engineering** · `core/forecasting.py`
  Build: `year, month, weekofyear, quarter, lag_1/4/8, rolling_mean_4, rolling_std_4` (+ optional one-hot slice).
  Done: feature matrix has no NaNs after warm-up rows dropped.
- **3.3 Time-based split** · `core/forecasting.py`
  Build: last ~20% as test; **never random**.
  Done: test dates strictly after train (asserted in test 3.9).
- **3.4 XGBoost model** · `core/forecasting.py`
  Build: train `XGBRegressor`; recursive multi-step forecast over the horizon.
  Done: produces an N-week forecast array.
- **3.5 SARIMA model** · `core/forecasting.py`
  Build: `statsmodels.SARIMAX` (seasonal 52, or monthly/period-12 fallback); forecast + confidence interval.
  Done: forecast + CI returned; no fit error on the series.
- **3.6 Metrics & model comparison** · `core/forecasting.py`
  Build: RMSE / MAE / **MAPE** on test for both; pick winner → InsightJSON `forecast`.
  Done: comparison table populated; winner flagged.
- **3.7 SHAP explainability** · `core/forecasting.py`
  Build: `shap.TreeExplainer` on XGBoost → global importance → InsightJSON `drivers`.
  Done: ranked driver list produced (reused by Phase 4).
- **3.8 Predictive page UI** · `pages/4_Predictive.py`
  Build: comparison table; overlaid actual-vs-forecast chart with **CI band**; horizon selector; **what-if simulator** (growth/horizon sliders); SHAP panel.
  Done: all controls change the output live.
- **3.9 Tests** · `tests/test_forecasting.py`
  Build: no-leakage assertion + metric-formula check on a toy series.
  Done: `pytest` green.

✅ **Checkpoint:** "forecast next 8 weeks of revenue — two models compared, here's why."

---

## PHASE 4 — Diagnostic layer  ·  2–3 days
**Goal:** explain the *why* — customer segments + anomaly flags.

- **4.1 RFM build** · `core/diagnostic.py`
  Build: per `customer_id` → Recency (days since last order), Frequency (order count), Monetary (Σ revenue).
  Done: RFM table correct on demo data (spot-check one customer).
- **4.2 Clustering** · `core/diagnostic.py`
  Build: StandardScaler → choose k via elbow/silhouette (expect ~4) → KMeans → map clusters to labels (Champions/Loyal/At-Risk/Lost) by sorting on R & M.
  Done: every customer gets a human-readable segment.
- **4.3 Segment profiling** · `core/diagnostic.py`
  Build: per-segment counts + revenue share → InsightJSON `segments` (incl. `at_risk_revenue_share`).
  Done: segment summary written to InsightJSON.
- **4.4 Anomaly detection** · `core/diagnostic.py`
  Build: IsolationForest on revenue-series features (value, rate-of-change, month) → flags → InsightJSON `anomalies`.
  Done: flagged dates returned with a score/type.
- **4.5 Diagnostic page UI** · `pages/3_Diagnostic.py`
  Build: segment scatter/table + per-segment revenue share; anomaly markers overlaid on the trend; feature-importance panel (reuse 3.7 SHAP).
  Done: renders; **gracefully hides RFM if `customer_id` absent**.
- **4.6 Tests** · `tests/test_rfm.py`
  Build: RFM math on a tiny fixture.
  Done: `pytest` green.

✅ **Checkpoint:** "these customers are at risk; these revenue moves are abnormal."

---

## PHASE 5 — Prescriptive + Ask-your-data  ·  3–4 days  ·  *the innovation*
**Goal:** LLM turns combined model output into **ranked, evidence-cited actions** + a working data chat.

- **5.1 InsightJSON assembler** · `core/insight_bus.py`
  Build: pydantic models for the schema (Arch §2); `insight_bus` collects outputs from Phases 2–4 and validates.
  Done: a fully-populated, schema-valid InsightJSON for the demo dataset.
- **5.2 LLM adapter** · `core/llm.py`
  Build: `complete(system, user) -> str`; env-selected provider (`groq`/`openai`/`ollama`); timeout + one retry; a **no-key path** that signals "use fallback."
  Done: returns text with a key; signals fallback without one.
- **5.3 Prescriptive prompt + parser** · `core/prescriptive.py`
  Build: prompt that **requires each action to cite a JSON field path** (e.g. `forecast.by_region.East`); parse LLM JSON → cards; rank by impact (revenue-share × |delta|).
  Done: 3–5 actions, each with `{title, why(evidence), impact_tag}`.
- **5.4 Rule-based fallback** · `core/prescriptive.py`
  Build: templated actions derived from the same InsightJSON when no key.
  Done: **app still gives real recommendations with no API key.**
- **5.5 Prescriptive page UI** · `pages/5_Prescriptive.py`
  Build: ranked action cards showing the evidence line + impact tag.
  Done: cards render; each cites a real model output (not generic advice).
- **5.6 Ask-your-data — answer mode** · `core/nlp.py`, `pages/6_Ask_Your_Data.py`
  Build: `st.chat_input`; send question + canonical schema + precomputed aggregates → LLM answer.
  Done: answers the canned demo questions correctly.
- **5.7 Ask-your-data — safe query mode (stretch)** · `core/nlp.py`
  Build: LLM generates a **whitelisted** pandas expression (no `eval` of raw text) → run → render a chart.
  Done: at least one live "chart from a question" works; **labeled stretch** (skip if short on time).
- **5.8 Tests** · `tests/test_insightjson.py`
  Build: schema validity; fallback path; **privacy** (LLM sees only aggregated JSON, never raw customer rows).
  Done: `pytest` green.

✅ **Checkpoint:** "AI says: focus on East (forecast −21%) and win back 128 at-risk customers (23% of revenue) — here's the evidence."

---

## PHASE 6 — Polish, hardening, deploy, demo  ·  2–3 days
**Goal:** submission-ready.

- **6.1 Theme & branding** · `.streamlit/config.toml`, `assets/`
  Build: finance color palette, logo, consistent layout.
  Done: app looks intentional, not default-Streamlit.
- **6.2 UX states** · all `pages/`
  Build: empty-states, spinners, try/except with friendly messages per page.
  Done: no raw traceback ever reaches the user.
- **6.3 Robustness pass** · —
  Build: run 2–3 deliberately messy CSVs through the whole pipeline.
  Done: graceful degradation everywhere (missing cols hide features, not crash).
- **6.4 Docs** · `README.md`
  Build: architecture diagram, screenshots, local-run + deploy steps.
  Done: a stranger can run it from the README.
- **6.5 Production deploy** · Streamlit Cloud
  Build: set LLM secrets in the platform; final deploy; end-to-end smoke test on the public URL.
  Done: full pipeline runs on the deployed URL with secrets.
- **6.6 Demo assets** · —
  Build: 2-min demo script (upload→descriptive→forecast→diagnose→prescribe→ask); record GIF/Loom; capture report screenshots per Arch §10 traceability.
  Done: demo script + recording + one screenshot per objective.

✅ **Checkpoint:** the final viva demo on the deployed URL.

---

## Critical path & parallelization
- **Strictly ordered:** 0 → 1 → 2 (all downstream needs canonical df + something to show).
- **3 and 4 can run in parallel** if two people (both consume canonical df); solo, do 3 then 4 (3 feeds SHAP to 4).
- **5 depends on 2+3+4** (needs InsightJSON populated). **6** is light-touch continuous, finalized last.

## Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| SARIMA unstable on weekly (52) points | Med | Monthly aggregation (period 12); XGBoost stays primary. |
| Free-tier RAM (1 GB) on Streamlit Cloud | Med | LLM off-box via API; weekly aggregation; cache; else HF Spaces. |
| LLM cost/latency/no key at grading | Med | Groq free tier + **rule-based fallback** (5.4). |
| `pmdarima` install issues | Low | Use `SARIMAX` fixed order; `pmdarima` optional. |
| Scope creep on 5.7 | Med | Ship 5.6 first; 5.7 is a labeled stretch. |
| Deadline shorter than ~3 wks | ? | Phases 0–3 + rule-based 5.4/5.5 = complete, defensible submission; drop 5.7, what-if, Path C. |

## Definition of done (prototype)
A deployed public URL where uploading a sales CSV produces, in one session: a cross-filterable revenue dashboard, a benchmarked explainable forecast, customer segments + anomaly flags, and a ranked list of AI recommendations that each cite their model evidence — usable by someone with no coding/ML background.

## Minimum viable submission (if time is short)
Phases **0, 1, 2, 3** + **5.1, 5.4, 5.5** (rule-based prescriptive, no LLM key needed) + **6.4, 6.5**. That's descriptive + predictive + evidence-cited recommendations, deployed — a complete four-layer story minus the LLM chat polish.
