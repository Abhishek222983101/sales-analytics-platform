# System Architecture — Intelligent Sales Analytics Platform

**Project:** Design and Development of an Intelligent Sales Analytics Platform Using Machine Learning for Predictive and Prescriptive Business Insights
**Owner:** Riddhi (VCET) · final-year prototype
**Dataset (single, fixed):** [Kaggle — Superstore Sales `rohitsahoo/sales-forecasting`](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
**Status:** planning → ready to build

> Read this with `BUILD-PLAN.md` (the phase-by-phase execution plan). This file is the *what and why*; the build plan is the *in what order*.

---

## 0. Locked scope decisions (read first)

These reconcile the synopsis with the revised WhatsApp instructions ("one dataset only, finance-specific, some innovation, prototype only"). **The synopsis must be edited to match these — see §9.**

| Decision | Value | Reason |
|---|---|---|
| Datasets | **ONE** (Superstore only) | Instruction: "sirf ek hi dataset". Drop all "tested on 2+ datasets" claims. |
| Generalization claim | **Design-level only** ("architected to generalize" via auto schema-mapper) — NOT empirically tested | Keeps the good part of the research-gap story without needing a 2nd dataset. |
| Domain framing | **Financial / revenue analytics** on retail sales | Instruction: "finance domain specific". This dataset is retail sales; "finance" = the revenue/money lens (revenue, growth %, revenue-at-risk, revenue concentration). |
| Headline innovation | **Explainable *recommendations*** — LLM reasons over the *combined* output of all 4 models and cites the evidence behind each action | This is Research Gap #3 in the synopsis and needs no 2nd dataset. |
| Product surface | **Streamlit multipage web app** (the "portal") | Fastest path to a demoable portal; one language (Python) for UI + ML. |
| Deployment | **Streamlit Community Cloud** (primary) + Groq/OpenAI hosted LLM | Free, redeploys on git push, no infra. See §7. |

### Hard data constraint (drives everything)
The `rohitsahoo/sales-forecasting` train.csv has **only `Sales` as a numeric measure** — **no Profit, Quantity, Discount, or review text.** Columns:

```
Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name,
Segment, Country, City, State, Postal Code, Region, Product ID, Category,
Sub-Category, Product Name, Sales
```

Consequences baked into this architecture:
- Forecasting target = **revenue** (Σ Sales), not profit/units.
- XGBoost features = date-parts + lag/rolling revenue + category/region (**no discount feature**).
- RFM fully supported (Recency=days-since-last-order, Frequency=order count, Monetary=Σ Sales).
- Prescriptive actions are **revenue-based** (concentration, timing, region/category reallocation, at-risk win-back) — **no restock/promotion/discount actions** (no data to justify them).
- VADER/sentiment model = **cut** (no text column).

*(Confirm exact rows/date-range with `pd.read_csv('data/sample_superstore.csv').info()` in Phase 0 — expected ~9,800 rows, 2015–2018, Country = United States.)*

---

## 1. Architecture at a glance

Single Python web app. Six pages (one per analytics layer + upload), a set of stateless service modules, an ML/stats core, and a hosted LLM accessed through one provider-agnostic adapter. All heavy compute is cached by dataframe hash.

```
┌──────────────────────────────────────────────────────────────────┐
│                     PRESENTATION  (Streamlit)                      │
│  Upload │ Descriptive │ Diagnostic │ Predictive │ Prescriptive │   │
│         │             │            │            │  Ask-your-data │  │
└───────────────┬──────────────────────────────────────────────────┘
                │ function calls (no HTTP; same process)
┌───────────────▼──────────────────────────────────────────────────┐
│                    APPLICATION / SERVICES  (core/)                 │
│  ingestion → schema_mapper → [descriptive] [diagnostic]            │
│                              [forecasting] → insight_bus →         │
│                              [prescriptive] [nlp]                   │
│  insight_bus.py assembles the unified InsightJSON (the spine)      │
└───────────┬───────────────────────────────────┬──────────────────┘
            │ uses                               │ uses
┌───────────▼──────────────────┐   ┌─────────────▼────────────────────┐
│      ML / STATS CORE          │   │        LLM INFERENCE              │
│  xgboost                      │   │  llm.py adapter →                 │
│  statsmodels (SARIMAX)        │   │   Groq (Llama-3.1) / OpenAI       │
│  scikit-learn (KMeans,        │   │   (gpt-4o-mini) / Ollama (local)  │
│    IsolationForest, scaling)  │   │  + rule-based FALLBACK if no key  │
│  shap                         │   │  secrets-managed, never committed │
└───────────┬──────────────────┘   └───────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────┐
│                          DATA LAYER                                │
│  uploaded CSV → validated → canonical schema → feature frames      │
│  (all cached via @st.cache_data, keyed by file/df hash)           │
└──────────────────────────────────────────────────────────────────┘
```

**Why not a Next.js + FastAPI split for the prototype?** It triples the work (API layer, CORS, two deploys, state sync) for a demo. Streamlit gives upload, caching, Plotly charts, and a chat widget out of the box in one language. The split is documented in §7 as the "stretch / resume" option.

---

## 2. The InsightJSON contract (the spine + the innovation)

Every analytical model writes into **one shared structured object**. The prescriptive LLM reasons over this whole object — not documents (no RAG) — so each recommendation can cite the exact field that justifies it. This is the project's novel contribution: *explainability of the recommendation, not just the prediction.*

```jsonc
{
  "meta": { "rows": 9800, "date_range": ["2015-01-03","2018-12-30"], "currency": "USD" },
  "forecast": {
    "model_used": "xgboost",           // winner of the comparison
    "horizon_weeks": 8,
    "point_forecast_total": 142300.0,
    "delta_pct_vs_prev_period": -14.2, // <-- drives "revenue at risk" actions
    "ci_low": 128900.0, "ci_high": 155700.0,
    "by_region": { "West": -3.1, "East": -21.4, "Central": 2.0, "South": -8.8 }
  },
  "anomalies": [ { "date": "2018-11-23", "z": 3.4, "type": "spike" } ],
  "segments": {
    "labels": ["Champions","Loyal","At-Risk","Lost"],
    "counts": { "Champions": 44, "Loyal": 210, "At-Risk": 128, "Lost": 96 },
    "at_risk_revenue_share": 0.23     // <-- drives "win-back" actions
  },
  "drivers": [                         // SHAP, global importance
    { "feature": "lag_4_revenue", "importance": 0.31 },
    { "feature": "month",         "importance": 0.22 }
  ],
  "revenue_concentration": {           // Pareto
    "top20pct_customers_revenue_share": 0.71,  // <-- drives "protect key accounts"
    "top_category": "Technology"
  }
}
```

`core/insight_bus.py` builds this from the outputs of descriptive/diagnostic/forecasting. `core/prescriptive.py` sends it to the LLM with a prompt that **requires each action to reference a field path** (e.g. `forecast.by_region.East`). If the LLM/API key is absent, a rule-based fallback produces templated actions from the same JSON so the demo never dies.

---

## 3. Canonical schema + the auto schema-mapper

The app never hard-codes Superstore column names. `schema_mapper.py` maps an arbitrary uploaded CSV onto this internal standard — this is what lets the report honestly say "architected to generalize to any sales database" (design-level claim, §0).

**Canonical schema**
```
date         datetime64   (required)
revenue      float        (required)
order_id     str          (optional)
customer_id  str          (optional; enables RFM)
category     str          (optional; enables breakdowns)
sub_category str          (optional)
region       str          (optional)
segment      str          (optional)
product_id   str          (optional)
```

**Mapping strategy (in order):**
1. **Name match** — fuzzy match header names against a synonym dictionary (`{"revenue": ["sales","amount","revenue","total","gmv"], "date": ["order date","date","order_date","invoice date"], ...}`).
2. **Dtype heuristic** — first parseable-as-date column → `date`; the numeric column with the widest range / highest sum → `revenue` if name match is ambiguous.
3. **User override** — the Upload page shows the detected mapping in `st.selectbox` dropdowns so the user can correct it before proceeding.

Output: a `canonical_df` + a `mapping_report` (shown to the user). Everything downstream reads only canonical column names.

---

## 4. Module responsibilities (`core/`)

| Module | Responsibility | Key libs | Writes to InsightJSON |
|---|---|---|---|
| `ingestion.py` | read CSV, validate (>1 date col, >1 numeric), basic clean (parse dates, drop dupes, coerce types, handle missing) | pandas | `meta` |
| `schema_mapper.py` | map uploaded columns → canonical schema (§3) | pandas, rapidfuzz | — |
| `descriptive.py` | KPI aggregates, trend series, breakdowns, Pareto | pandas | `revenue_concentration` |
| `diagnostic.py` | RFM build → StandardScaler → KMeans (k via elbow) → label segments; IsolationForest on daily revenue | scikit-learn | `segments`, `anomalies` |
| `forecasting.py` | feature-engineer, train XGBoost + SARIMAX, time-based split, metrics (RMSE/MAE/MAPE), pick winner, SHAP | xgboost, statsmodels, shap, scikit-learn | `forecast`, `drivers` |
| `insight_bus.py` | collect all module outputs → validated InsightJSON | pydantic | (assembles it) |
| `prescriptive.py` | InsightJSON → LLM prompt → ranked actions w/ evidence + impact tag; rule-based fallback | llm.py | — |
| `nlp.py` | "Ask your data": question + schema + aggregates → LLM answer (+ optional safe pandas expr over a column whitelist) | llm.py | — |
| `llm.py` | one `complete(system, user) -> str`; provider set by env (`groq`/`openai`/`ollama`); timeout, retry, graceful "no key" path | groq / openai sdk | — |

**Design rules:** service functions are pure (df in → result out), no Streamlit imports in `core/` (keeps it testable), all expensive calls wrapped in `@st.cache_data` at the page layer keyed by a stable df hash.

---

## 5. ML / stats design decisions

**Forecasting (Predictive layer)**
- Aggregate to a **weekly revenue series** (daily is too noisy for ~4 yrs; weekly gives ~208 points — enough for SARIMA seasonality and XGBoost lags).
- Features (XGBoost): `year, month, weekofyear, quarter, lag_1, lag_4, lag_8, rolling_mean_4, rolling_std_4`, plus optional one-hot `region`/`category` when forecasting a slice.
- **Time-based split** (last ~20% as test) — never random split (leakage). This is a graded correctness point.
- SARIMA: `statsmodels.SARIMAX`, seasonal period 52 (weekly) or start with monthly aggregation + period 12 if 52 is unstable on limited points. `pmdarima.auto_arima` optional for order search (note: can be install-flaky; SARIMAX with a fixed sensible order is the safe default).
- Metrics: RMSE, MAE, **MAPE** (headline, business-readable). Comparison table + overlaid actual/forecast chart with CI band.
- SHAP: `shap.TreeExplainer` on the XGBoost model → global importance for the `drivers` panel (doubles as diagnostic feature-importance).

**Diagnostic**
- RFM per `customer_id`; StandardScaler; **KMeans** with k chosen by elbow/silhouette (expect k≈4); map clusters → business labels by sorting on monetary/recency.
- **IsolationForest** on the daily/weekly revenue series features (value, rate-of-change, month) → anomaly flags overlaid on the trend.

**Prescriptive**
- Not a model — an LLM reasoning step over InsightJSON (§2). Deterministic-ish: low temperature, JSON-structured output parsed into cards. Always ranked by an `impact` heuristic (e.g. revenue share affected × |delta|).

---

## 6. Repository structure

```
sales-analytics-platform/
├── app.py                      # Streamlit entry: landing + nav
├── pages/                      # Streamlit auto-routes these (order by number)
│   ├── 1_Upload.py
│   ├── 2_Descriptive.py
│   ├── 3_Diagnostic.py
│   ├── 4_Predictive.py
│   ├── 5_Prescriptive.py
│   └── 6_Ask_Your_Data.py
├── core/                       # pure Python, NO streamlit imports (testable)
│   ├── ingestion.py
│   ├── schema_mapper.py
│   ├── descriptive.py
│   ├── diagnostic.py
│   ├── forecasting.py
│   ├── insight_bus.py
│   ├── prescriptive.py
│   ├── nlp.py
│   └── llm.py
├── data/
│   └── sample_superstore.csv   # bundled demo dataset (so app works with no upload)
├── assets/                     # logo, custom css
├── tests/                      # pytest: schema_mapper, rfm, metrics, insightjson
├── .streamlit/
│   ├── config.toml             # theme (finance palette)
│   └── secrets.toml            # GITIGNORED — LLM keys
├── DATA_PROFILE.md             # committed df.info() output (Phase 0)
├── requirements.txt
├── runtime.txt                 # python-3.11
├── Dockerfile                  # only if deploying to HF Spaces / Render
├── .gitignore
├── SYSTEM-ARCHITECTURE.md      # this file
├── BUILD-PLAN.md
└── README.md
```

**`requirements.txt` (pinned at build time):**
```
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
statsmodels
shap
rapidfuzz
pydantic
python-dotenv
groq            # or openai — the LLM provider sdk
pytest          # dev only
```

---

## 7. Deployment architecture

### Recommended — Path A: Streamlit Community Cloud (free, zero-infra) ✅
```
GitHub repo ──push──> Streamlit Community Cloud ──builds requirements.txt──> https://<app>.streamlit.app
                                     │
                            secrets (LLM_API_KEY) set in the Streamlit dashboard, not in git
                                     │
LLM inference ──HTTPS──> Groq API (Llama-3.1-8b/70b, free tier)  OR  OpenAI gpt-4o-mini
```
- **Steps:** push to GitHub → share.streamlit.io → "New app" → pick repo/branch/`app.py` → add secrets → deploy. Redeploys automatically on every `git push`.
- **Constraints:** ~1 GB RAM, sleeps when idle (wakes on visit). Fine because the LLM runs *off-box* via API — we never load an 8B model locally.
- **Why this wins for a prototype:** free, one repo, one command to redeploy, public URL for the viva.

### Alternative — Path B: Hugging Face Spaces (free, more RAM, ML-friendly)
Same code + a `Dockerfile`. Good if Streamlit Cloud RAM bites. Secrets via Space settings.

### Alternative — Path C: "real product" split (resume-grade, more work)
```
Vercel ──hosts──> Next.js frontend (the portal UI)
   │  fetch()
   ▼
Render / Railway ──hosts──> FastAPI backend  (/descriptive /forecast /diagnostic /prescriptive /ask)
                                   │ loads models in memory, calls LLM API
```
- **Important:** Vercel is for the **JS frontend only**. Do **not** try to run Streamlit/XGBoost/SARIMA/SHAP on Vercel — serverless functions have short execution limits, a ~250 MB bundle cap, cold starts, and no persistent in-memory model. The Python/ML **must** live on Render/Railway (or HF Spaces) as a normal always-on web service. Choose Path C only if a custom-looking portal is worth the extra ~1 week.

### LLM deployment note (applies to all paths)
For a *deployed* app the LLM must be a **hosted API** — a local Ollama/Llama-3-8B won't fit free-tier RAM. Use **Groq free tier (Llama-3.1)** or **OpenAI gpt-4o-mini** (both cost ≈ fractions of a cent per call, or free on Groq). Keep **Ollama for local dev only**. `llm.py` abstracts the provider so switching is a one-line env change, and the rule-based fallback keeps the demo alive with no key at all.

### Secrets & config
- `.streamlit/secrets.toml` (gitignored) locally; platform secret store when deployed.
- Keys: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`. Never commit; `.gitignore` covers `secrets.toml`, `.env`, `__pycache__`, `*.pyc`, data exports.

---

## 8. Non-functional requirements (prototype-appropriate)

| Concern | Approach |
|---|---|
| Performance | `@st.cache_data` on ingestion, features, model fits (keyed by df hash) so re-runs are instant. Weekly aggregation keeps model fits < a few seconds. |
| Robustness | Every layer degrades gracefully: no `customer_id` → hide RFM; no `category` → hide breakdown; no LLM key → rule-based recommendations. App must never crash on a "weird" CSV. |
| Testability | `core/` is Streamlit-free and unit-tested (schema mapping, RFM math, metric formulas, InsightJSON schema). |
| Security | No secrets in repo; uploaded data stays in session memory (not persisted server-side); LLM sees only aggregated JSON, never raw customer rows. |
| Reproducibility | Pinned `requirements.txt`, `runtime.txt`, bundled sample dataset, committed `DATA_PROFILE.md`. |

---

## 9. Required edits to the synopsis (`.docx`) so docs match the build

1. **Objective #6** ("test on more than one dataset") → **remove** or change to "designed for portability via an automatic schema-mapper (validated on the Superstore dataset)."
2. **Expected Outcome**, last paragraph ("tested on at least one dataset beyond the primary Kaggle dataset") → **remove**.
3. **Research Gap #1** wording → keep the *gap description*, but state the contribution as a **dataset-agnostic architecture**, not an empirically multi-dataset evaluation.
4. Replace every **profit / restock / promotion / discount** mention (Modules §5, Model #1 features, Prescriptive examples) with **revenue-based** equivalents.
5. **Model #6 (VADER)** → drop entirely (no text column), or keep one line noting it's out of scope for this dataset.
6. Reframe KPI/vocabulary as **financial** (revenue, growth %, revenue-at-risk, revenue concentration) to satisfy "finance domain specific."

---

## 10. Report ↔ feature traceability (for the viva/submission)

Every objective should map to a built, screenshot-able artifact:

| Synopsis objective | Built artifact | Phase |
|---|---|---|
| Ingestion + auto column mapping | Upload page + mapping report | 1 |
| Descriptive dashboard | Descriptive page (KPIs, trends, Pareto) | 2 |
| Predictive, multi-model + explainable | Predictive page (XGBoost vs SARIMA, SHAP) | 3 |
| Diagnostic (clustering + anomaly) | Diagnostic page (segments, anomalies) | 4 |
| Prescriptive (LLM ranked actions) | Prescriptive page (evidence-cited cards) | 5 |
| Non-technical usability | Whole Streamlit UX + Ask-your-data chat | 2–5 |

Keep one screenshot per row for the report's "Results" section.
