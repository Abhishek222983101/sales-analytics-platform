"""Sections 0-3: title, setup, data loading, web scraping, pre-processing."""


def add(md, code, fallback_csv):
    md(r'''
# 🛒 Sales Analytics using Data Warehousing & Data Mining

**Course:** Data Warehousing & Mining (DWM) · **Type:** End-to-end mini project
**Dataset:** Superstore Sales — [Kaggle · rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
**Team:** _<add names / roll numbers here>_

> **How to run:** upload this file to Google Colab → **Runtime ▸ Run all**. The data downloads itself, the web-scraping step fetches live data, and every model trains automatically (about 3–5 minutes). Nothing to upload or configure.

---

## 📑 Contents
| # | Section | DWM topic |
|---|---|---|
| 1 | Data collection | Dataset + data dictionary |
| 2 | Web scraping | Collecting external data |
| 3 | Data pre-processing | Cleaning, imputation, outliers, encoding, scaling |
| 4 | Star schema & OLAP | **Data warehousing** — fact/dimension tables, roll-up, drill-down, slice, dice |
| 5 | Exploratory data analysis | Visual analysis |
| 6 | Classification | Decision tree, Random Forest, Logistic Regression, KNN, Naive Bayes, XGBoost |
| 7 | Regression | Linear / Ridge / Tree / Forest / XGBoost + time-series forecast |
| 8 | Clustering | K-Means, Hierarchical, RFM customer segmentation |
| 9 | Association rules | **Apriori** (library + from scratch) |
| 10 | Bonus | Anomaly detection + live web app |
| 11 | Conclusions | Results, insights, limitations |

## 🎯 Problem statement
A retail business records every sale but cannot easily see *what happened, why, what will happen, and what to do about it*.
We build a small **data warehouse (star schema)** from the raw sales records and apply the standard **data-mining techniques**
(classification, regression, clustering, association rules) to turn that data into business decisions.

## 🧭 Pipeline
`Raw CSV` → **scrape** extra data → **pre-process** → load into **star schema (SQL)** → **EDA** → **mine** (classify · regress · cluster · associate) → **insights**
''')

    md(r'''
## ⚙️ 0. Setup
We use the standard Python data stack. `mlxtend` provides the Apriori algorithm (not pre-installed on Colab).
''')
    code(r'''
%pip install -q mlxtend
''')
    code(r'''
import warnings, io, re, sqlite3, time
from itertools import combinations
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import requests

PALETTE = ["#E8497B", "#8E7DBE", "#F7C948", "#5AA7A7", "#C64B8C", "#E9A178", "#2B2B33"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_columns = 40
RANDOM_STATE = 42
money = mtick.FuncFormatter(lambda x, _: f"${x:,.0f}")
print(f"Setup complete ✅  (pandas {pd.__version__} · numpy {np.__version__} · matplotlib {plt.matplotlib.__version__})")
''')

    # ------------------------------------------------------------------ 1
    md(r'''
## 📥 1. Data collection

The dataset is the **Superstore Sales** data: one row per *product line on an order* for a US retailer, 2015–2018.
We download the CSV directly from a public mirror of the Kaggle dataset. If the download is blocked, Colab will ask you to upload `train.csv` (from the Kaggle page) instead.
''')
    code(r'''
DATA_URL = "https://raw.githubusercontent.com/khairalanam/sales-forecast-excel-project/main/train.csv"
LOCAL_PATH = None            # e.g. "train.csv" if you already have the file next to the notebook

def load_data():
    if LOCAL_PATH:
        return pd.read_csv(LOCAL_PATH)
    try:
        return pd.read_csv(DATA_URL)
    except Exception as exc:
        print(f"Could not download the dataset ({type(exc).__name__}). Falling back to a manual upload...")
    from google.colab import files            # only available on Colab
    uploaded = files.upload()
    return pd.read_csv(io.BytesIO(next(iter(uploaded.values()))))

raw = load_data()
print(f"Loaded {raw.shape[0]:,} rows × {raw.shape[1]} columns")
raw.head()
''')
    code(r'''
data_dictionary = pd.DataFrame([
    ("Row ID",        "int",    "Row counter — carries no business meaning"),
    ("Order ID",      "text",   "Order identifier; one order has several rows (one per product)"),
    ("Order Date",    "date",   "When the order was placed (dd/mm/yyyy)"),
    ("Ship Date",     "date",   "When the order was shipped"),
    ("Ship Mode",     "text",   "Shipping class: Same Day / First / Second / Standard"),
    ("Customer ID",   "text",   "Customer identifier"),
    ("Customer Name", "text",   "Customer name"),
    ("Segment",       "text",   "Consumer / Corporate / Home Office"),
    ("Country",       "text",   "Always United States"),
    ("City",          "text",   "Delivery city"),
    ("State",         "text",   "Delivery state"),
    ("Postal Code",   "number", "Delivery ZIP code"),
    ("Region",        "text",   "Central / East / South / West"),
    ("Product ID",    "text",   "Product identifier"),
    ("Category",      "text",   "Furniture / Office Supplies / Technology"),
    ("Sub-Category",  "text",   "17 product types (Chairs, Phones, Binders, …)"),
    ("Product Name",  "text",   "Product description"),
    ("Sales",         "money",  "★ The only numeric measure: sales value of the line (USD)"),
], columns=["column", "type", "meaning"])
data_dictionary
''')
    code(r'''
raw.info()
''')
    code(r'''
raw[["Sales"]].describe().T
''')
    md(r'''
**Reading the summary:** there is *only one* numeric measure (`Sales`) — no quantity, discount or profit. That shapes every model below:
we can analyse **revenue**, not margin. Also note the huge gap between the median and the maximum sale: the distribution is extremely skewed (we deal with that in pre-processing).
''')

    # ------------------------------------------------------------------ 2
    md(r'''
## 🌐 2. Web scraping — enriching the data with US state populations

Sales totals alone favour big states. To judge a state fairly we need its **population**, which is not in the dataset — so we **scrape** it from Wikipedia
(*List of U.S. states and territories by population*, 2020 Census column) and use it later as a new attribute in the warehouse (`dim_geography`) and in the models.

**How the scraper works**
1. `requests.get` downloads the page (with a polite `User-Agent`; one request only).
2. `pandas.read_html` parses every HTML `<table>`; we pick the population table.
3. Clean-up: flatten the two-row header, strip footnote markers like `[a]`, remove commas, convert to integers.
4. Keep only the 50 states + D.C. (drop territories and total rows).

> Wikipedia text is CC BY-SA licensed; we fetch a single page for coursework purposes. If the live request ever fails (no internet, layout change), the notebook falls back to an embedded copy of a previous scrape so it never breaks.
''')
    code(
        r'''
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population"
US_STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia",
    "Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
    "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey",
    "New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia",
    "Wisconsin","Wyoming","District of Columbia",
]

FALLBACK_CSV = """__FALLBACK_CSV__"""

def scrape_state_population():
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (DWM college project)"}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))          # every <table> on the page
    t = tables[0]
    t.columns = [" | ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    state_col = next(c for c in t.columns if c.startswith("State or territory"))
    pop_col   = next(c for c in t.columns if "April 1, 2020" in c)
    out = t[[state_col, pop_col]].copy()
    out.columns = ["state", "population_2020"]
    out["state"] = out["state"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    out["population_2020"] = pd.to_numeric(
        out["population_2020"].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
    return out[out["state"].isin(US_STATES)].dropna().reset_index(drop=True)

try:
    state_pop = scrape_state_population()
    assert len(state_pop) >= 50, "unexpected table layout"
    scrape_source = "live scrape of Wikipedia"
except Exception as exc:
    print(f"Live scrape failed ({type(exc).__name__}: {exc}) → using the embedded copy.")
    state_pop = pd.read_csv(io.StringIO(FALLBACK_CSV))
    scrape_source = "embedded copy of an earlier scrape"

state_pop["population_2020"] = state_pop["population_2020"].astype(int)
print(f"Source: {scrape_source} · {len(state_pop)} rows · total population {state_pop['population_2020'].sum():,}")
state_pop.head(8)
'''.replace("__FALLBACK_CSV__", fallback_csv)
    )
    code(r'''
# Sanity checks on the scraped data + a quick visual
sales_states = set(raw["State"])
missing_in_scrape = sorted(sales_states - set(state_pop["state"]))
print(f"States in the sales data: {len(sales_states)} · matched by the scrape: {len(sales_states) - len(missing_in_scrape)} · unmatched: {missing_in_scrape or 'none'}")

top = state_pop.sort_values("population_2020", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=top, y="state", x="population_2020", color=PALETTE[1], ax=ax)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.set(title="Scraped data: 10 most populous states (2020 Census)", xlabel="Population", ylabel="")
plt.tight_layout(); plt.show()
''')

    # ------------------------------------------------------------------ 3
    md(r'''
## 🧹 3. Data pre-processing

Raw data is never model-ready. We follow a fixed checklist and **show evidence at every step**:

| Step | Question |
|---|---|
| 3.1 Audit | What is wrong with the raw data? |
| 3.2 Data types | Are dates / ZIP codes stored correctly? |
| 3.3 Missing values | Which values are missing and how do we fill them? |
| 3.4 Duplicates | Are any records repeated? |
| 3.5 Consistency | Do the business rules hold? |
| 3.6 Feature engineering | What useful columns can we derive? |
| 3.7 Outliers & skew | Are extreme values errors or real? |
| 3.8 Encoding & scaling | How do we make data digestible for algorithms? |
| 3.9 Before vs after | Did we improve quality? |
''')

    md(r'''
### 3.1 Audit of the raw data
''')
    code(r'''
def quality_report(d):
    return pd.DataFrame({
        "dtype":  d.dtypes.astype(str),
        "nulls":  d.isna().sum(),
        "null_%": (d.isna().mean() * 100).round(2),
        "unique": d.nunique(),
    })

audit_before = quality_report(raw)
n_dupes_raw = int(raw.drop(columns="Row ID").duplicated().sum())
print(f"Total missing cells: {int(raw.isna().sum().sum())} · duplicate records (ignoring Row ID): {n_dupes_raw}")
audit_before
''')
    code(r'''
nulls = raw.isna().sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
colors = [PALETTE[0] if v > 0 else "#D9D9E3" for v in nulls.values]
bars = ax.barh(nulls.index, nulls.values, color=colors)
ax.bar_label(bars, padding=3)
ax.invert_yaxis()
ax.set(title="Missing values per column (raw data)", xlabel="Number of missing cells")
plt.tight_layout(); plt.show()
''')
    md(r'''
**Finding:** the data is mostly complete — only `Postal Code` has gaps — but a "mostly complete" file still hides problems (wrong types, a duplicate, extreme skew). We go step by step.
''')

    md(r'''
### 3.2 Fixing data types
Dates arrive as text. Before parsing we must know the format: is `08/11/2017` the 8th of November or the 11th of August? We look for evidence — any date whose first number is greater than 12 *must* be a day.
''')
    code(r'''
df = raw.copy()

first_part = df["Order Date"].str.split("/").str[0].astype(int)
print(f"Order dates whose first number is > 12: {(first_part > 12).sum():,}  →  the first number is the DAY (format dd/mm/yyyy)")

df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  format="%d/%m/%Y")
print(f"Order dates now run from {df['Order Date'].min():%d %b %Y} to {df['Order Date'].max():%d %b %Y}")
df[["Order Date", "Ship Date"]].dtypes
''')

    md(r'''
### 3.3 Missing values — `Postal Code`
First: *where* are the gaps?
''')
    code(r'''
missing_mask = df["Postal Code"].isna()
print(f"{missing_mask.sum()} rows have no postal code. Their city/state:")
df.loc[missing_mask, ["City", "State"]].value_counts().to_frame("rows")
''')
    md(r'''
All the gaps belong to **one place — Burlington, Vermont**. That makes the choice easy:

| Option | Verdict |
|---|---|
| Drop the 11 rows | Loses real sales (information loss) |
| Fill with the most common ZIP | Wrong — would put Vermont sales in another state |
| Fill with the **known ZIP of Burlington, VT (05401)** — domain knowledge | ✅ Correct and transparent |

We also add a flag column `postal_imputed` so the change is traceable.

**Bonus catch:** ZIP codes were read as numbers, which silently drops leading zeros (Vermont's `05401` becomes `5401`). ZIP codes are *identifiers, not quantities*, so we store them as 5-character text.
''')
    code(r'''
df["postal_imputed"] = missing_mask
df.loc[missing_mask, "Postal Code"] = 5401                       # Burlington, VT
df["Postal Code"] = df["Postal Code"].astype(int).astype(str).str.zfill(5)

print("Missing postal codes after imputation:", int(df["Postal Code"].isna().sum()))
print("Example (leading zero restored):", df.loc[missing_mask, "Postal Code"].iloc[0])
''')

    md(r'''
### 3.4 Duplicate records
A record that is identical in *every* business field (order, product, customer, place, sales) is almost certainly a double entry. `Row ID` is just a counter, so we ignore it when comparing.
''')
    code(r'''
biz_cols = [c for c in df.columns if c != "Row ID"]
dupe_any = df.duplicated(subset=biz_cols, keep=False)
display(df.loc[dupe_any, ["Row ID", "Order ID", "Product ID", "Customer ID", "Sales"]])

before = len(df)
df = df.drop_duplicates(subset=biz_cols, keep="first").reset_index(drop=True)
print(f"Rows: {before:,} → {len(df):,}  ({before - len(df)} duplicate removed)")
''')

    md(r'''
### 3.5 Consistency checks (business rules)
Before trusting the data we test rules that *must* be true. Each check below should read `True`; anything `False` needs a decision.
''')
    code(r'''
for c in ["Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment", "Country", "City", "State",
          "Region", "Product ID", "Category", "Sub-Category", "Product Name"]:
    df[c] = df[c].astype(str).str.strip()                         # remove stray whitespace

one_to_one = lambda a, b: bool((df.groupby(a)[b].nunique() == 1).all())
checks = pd.Series({
    "Each Customer ID has exactly one name":           one_to_one("Customer ID", "Customer Name"),
    "Each Customer ID has exactly one segment":        one_to_one("Customer ID", "Segment"),
    "Each State belongs to exactly one Region":        one_to_one("State", "Region"),
    "Each Sub-Category belongs to one Category":       one_to_one("Sub-Category", "Category"),
    "Each Product ID has exactly one Category":        one_to_one("Product ID", "Category"),
    "Each Product ID has exactly one Product Name":    one_to_one("Product ID", "Product Name"),
    "Ship Date is never before Order Date":            bool((df["Ship Date"] >= df["Order Date"]).all()),
    "Sales is always positive":                        bool((df["Sales"] > 0).all()),
}, name="passes?")
checks.to_frame()
''')
    code(r'''
multi_name = df.groupby("Product ID")["Product Name"].nunique()
multi_name = multi_name[multi_name > 1]
print(f"{len(multi_name)} Product IDs appear under more than one product name. Example:")
ex = multi_name.index[0]
display(df.loc[df["Product ID"] == ex, ["Product ID", "Product Name"]].value_counts().to_frame("rows"))
''')
    md(r'''
**Decision:** a small share of product IDs carry more than one name (the same product renamed over time). The **ID is the reliable key**, so in the warehouse (section 4) each product ID gets *one* name — its most frequent one. Everything else passes.
''')

    md(r'''
### 3.6 Feature engineering
New columns derived from existing ones give the models and charts something to work with.
''')
    code(r'''
df["order_year"]       = df["Order Date"].dt.year
df["order_quarter"]    = df["Order Date"].dt.quarter
df["order_month"]      = df["Order Date"].dt.month
df["order_month_name"] = df["Order Date"].dt.month_name()
df["order_weekday"]    = df["Order Date"].dt.day_name()
df["ship_days"]        = (df["Ship Date"] - df["Order Date"]).dt.days       # delivery lead time
df["log_sales"]        = np.log1p(df["Sales"])                              # tames the skew (see 3.7)

print("Ship lead time (days):", df["ship_days"].min(), "to", df["ship_days"].max())
df[["Order Date", "order_year", "order_quarter", "order_month_name", "order_weekday", "ship_days", "Sales", "log_sales"]].head()
''')

    md(r'''
### 3.7 Outliers and skewness
`Sales` is the only measure, so its shape matters. We inspect it with a box-plot and the **IQR rule** (a value is an outlier if it lies above Q3 + 1.5 × IQR).
''')
    code(r'''
q1, q3 = df["Sales"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
df["is_outlier"] = df["Sales"] > upper_fence

print(f"Q1 = ${q1:,.2f} · Q3 = ${q3:,.2f} · IQR = ${iqr:,.2f} · upper fence = ${upper_fence:,.2f}")
print(f"Outliers: {df['is_outlier'].sum():,} rows ({df['is_outlier'].mean():.1%}) · skewness before = {df['Sales'].skew():.2f} · after log transform = {df['log_sales'].skew():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
sns.boxplot(x=df["Sales"], ax=axes[0, 0], color=PALETTE[0], fliersize=2)
axes[0, 0].set(title="Sales — box-plot (raw)", xlabel="Sales ($)")
sns.histplot(df["Sales"], bins=60, ax=axes[0, 1], color=PALETTE[0])
axes[0, 1].set(title=f"Sales — histogram (raw, skew = {df['Sales'].skew():.1f})", xlabel="Sales ($)")
sns.boxplot(x=df["log_sales"], ax=axes[1, 0], color=PALETTE[1], fliersize=2)
axes[1, 0].set(title="log(1 + Sales) — box-plot", xlabel="log(1 + Sales)")
sns.histplot(df["log_sales"], bins=40, ax=axes[1, 1], color=PALETTE[1])
axes[1, 1].set(title=f"log(1 + Sales) — histogram (skew = {df['log_sales'].skew():.2f})", xlabel="log(1 + Sales)")
plt.tight_layout(); plt.show()
''')
    code(r'''
# Are the outliers errors, or genuine big-ticket sales?
print("Largest 5 sales:")
display(df.nlargest(5, "Sales")[["Order ID", "Sub-Category", "Product Name", "Sales"]])

by_sub = (df.groupby("Sub-Category")["is_outlier"].mean().sort_values(ascending=False) * 100).round(1)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=by_sub.values, y=by_sub.index, color=PALETTE[2], ax=ax)
ax.set(title="Share of rows that are outliers, by sub-category", xlabel="% of rows flagged", ylabel="")
plt.tight_layout(); plt.show()
''')
    md(r'''
**Decision — keep, don't delete.** The outliers are not typing errors: they are expensive products (copiers, machines, conference-room tech) and they concentrate in a few sub-categories. Deleting them would remove exactly the sales a business cares about most.
Instead we **flag** them (`is_outlier`) and **log-transform** the target for modelling, which reduces skewness from the value shown above to near zero.
''')

    md(r'''
### 3.8 Encoding and scaling
Algorithms work on numbers, so categories must be **encoded**; and features on different scales (a ZIP-like count vs a dollar amount) must be **scaled**.
''')
    code(r'''
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

demo = df[["Ship Mode", "Sales"]].head(6).copy()
demo["label_encoded"] = LabelEncoder().fit_transform(demo["Ship Mode"])
onehot = pd.get_dummies(demo["Ship Mode"], prefix="mode").astype(int)
print("Label encoding invents an order (0 < 1 < 2 …) that does not exist; one-hot encoding does not.")
pd.concat([demo, onehot], axis=1)
''')
    code(r'''
sales = df[["Sales"]]
minmax_raw   = MinMaxScaler().fit_transform(sales)                       # squeezed by outliers
std_log      = StandardScaler().fit_transform(df[["log_sales"]])         # log first, then standardise

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
sns.histplot(sales["Sales"], bins=50, ax=axes[0], color=PALETTE[0]);  axes[0].set(title="Raw Sales", xlabel="$")
sns.histplot(minmax_raw.ravel(), bins=50, ax=axes[1], color=PALETTE[2]); axes[1].set(title="Min-Max scaled (0–1)\nalmost everything piles up near 0", xlabel="scaled value")
sns.histplot(std_log.ravel(), bins=50, ax=axes[2], color=PALETTE[1]);    axes[2].set(title="log → Standard scaled (mean 0, std 1)\nusable shape", xlabel="z-score")
plt.tight_layout(); plt.show()

summary = pd.DataFrame({
    "mean": [sales["Sales"].mean(), minmax_raw.mean(), std_log.mean()],
    "std":  [sales["Sales"].std(),  minmax_raw.std(),  std_log.std()],
    "min":  [sales["Sales"].min(),  minmax_raw.min(),  std_log.min()],
    "max":  [sales["Sales"].max(),  minmax_raw.max(),  std_log.max()],
}, index=["Raw Sales", "Min-Max scaled", "log + Standard scaled"])
summary
''')
    md(r'''
**Takeaway:** with extreme outliers, Min-Max scaling squashes 90% of the data into a sliver. *Log first, then standardise* gives a well-behaved feature. In the modelling sections we use **one-hot encoding** for categories and **standard scaling** for numbers, wrapped in a scikit-learn `ColumnTransformer` so the exact same steps are applied to training and test data (no leakage).
''')

    md(r'''
### 3.9 Before vs after
''')
    code(r'''
clean = df.copy()

comparison = pd.DataFrame({
    "Raw data": [len(raw), raw.shape[1], int(raw.isna().sum().sum()), n_dupes_raw,
                 raw["Order Date"].dtype.name, raw["Postal Code"].dtype.name, 0],
    "Clean data": [len(clean), clean.shape[1], int(clean.isna().sum().sum()),
                   int(clean.duplicated(subset=biz_cols).sum()),
                   clean["Order Date"].dtype.name, clean["Postal Code"].dtype.name, int(clean["is_outlier"].sum())],
}, index=["Rows", "Columns", "Missing cells", "Duplicate rows", "Order Date type", "Postal Code type",
          "Outliers flagged (kept)"])
display(comparison.astype(str))
print(f"\nSales total is preserved apart from the one duplicate: raw ${raw['Sales'].sum():,.2f} → clean ${clean['Sales'].sum():,.2f}")
''')
