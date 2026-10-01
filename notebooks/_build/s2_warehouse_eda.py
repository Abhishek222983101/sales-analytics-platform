"""Sections 4-5: star schema + OLAP, exploratory data analysis."""


def add(md, code):
    # ------------------------------------------------------------------ 4
    md(r'''
## 🏛️ 4. Data warehouse — star schema & OLAP

A flat CSV is fine for one-off analysis, but a business needs a **data warehouse**: a structure designed for *answering analytical questions fast and consistently*.
The classic design is the **star schema**:

* one central **fact table** holding the numbers we measure (here: `sales`), one row per event;
* several surrounding **dimension tables** describing *who / what / where / when* (customer, product, geography, date, shipping).

The fact table points to each dimension through a **foreign key**, so the picture looks like a star.

### Design decisions
| Decision | Our choice | Why |
|---|---|---|
| **Grain** (what one fact row means) | One *product line on an order* | Finest detail available → can always roll up, never drill below |
| **Measures** | `sales` (additive), `ship_days` | Numbers we aggregate |
| **Dimensions** | Date, Customer, Product, Geography, Ship Mode | Every question is "sales by ___" |
| **Degenerate dimension** | `order_id` stays in the fact table | It identifies the order but has no attributes of its own |
| **Role-playing dimension** | `dim_date` is used twice: `order_date_key` and `ship_date_key` | One calendar table, two meanings |
| **Surrogate keys** | Integer keys (`customer_key`, …) | Stable, small, independent of business IDs |
| **Enrichment** | `dim_geography.state_population_2020` (from our web scrape) | Lets us compute *revenue per resident* |

**Star vs snowflake:** a snowflake schema further normalises dimensions (e.g. a separate `category` table). It saves a little space but needs more joins; with small dimensions like ours, the star's simplicity and speed win.
''')

    md(r'''
### 4.1 The schema diagram
''')
    code(r'''
def draw_star_schema():
    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec=color, lw=2.2, zorder=3))
        ax.add_patch(plt.Rectangle((x, y + h - 0.55), w, 0.55, fc=color, ec=color, zorder=4))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", color="white",
                weight="bold", fontsize=11, zorder=5)
        ax.text(x + 0.18, y + h - 0.78, "\n".join(lines), ha="left", va="top", fontsize=8.6,
                family="monospace", linespacing=1.5, zorder=5)

    fact = (4.75, 2.55, 3.5, 3.55)
    dims = {
        "dim_date":      (0.25, 5.1, 3.6, 3.2, PALETTE[1],
                          ["PK date_key", "   date", "   year", "   quarter", "   month", "   month_name", "   day",
                           "   day_of_week", "   is_weekend"]),
        "dim_customer":  (9.15, 6.1, 3.6, 2.2, PALETTE[3],
                          ["PK customer_key", "   customer_id", "   customer_name", "   segment"]),
        "dim_product":   (0.25, 0.7, 3.6, 2.3, PALETTE[2],
                          ["PK product_key", "   product_id", "   product_name", "   category", "   sub_category"]),
        "dim_geography": (9.15, 0.7, 3.6, 3.15, PALETTE[5],
                          ["PK geo_key", "   country", "   region", "   state", "   city", "   postal_code",
                           "   state_population_2020", "   (from web scraping)"]),
        "dim_ship_mode": (4.75, 0.45, 3.5, 1.3, PALETTE[4],
                          ["PK ship_mode_key", "   ship_mode"]),
    }
    # connectors first (drawn behind the boxes)
    fx, fy, fw, fh = fact
    centre = (fx + fw / 2, fy + fh / 2)
    for name, (x, y, w, h, color, _) in dims.items():
        ax.plot([centre[0], x + w / 2], [centre[1], y + h / 2], color="#9A9AAE", lw=1.6, zorder=1)
    box(*fact, "fact_sales", ["PK sales_key", "FK order_date_key  ─▶ dim_date", "FK ship_date_key   ─▶ dim_date",
                              "FK customer_key    ─▶ dim_customer", "FK product_key     ─▶ dim_product",
                              "FK geo_key         ─▶ dim_geography", "FK ship_mode_key   ─▶ dim_ship_mode",
                              "   order_id  (degenerate dim)", "   sales      ★ measure", "   ship_days  ★ measure"],
        PALETTE[0])
    for name, (x, y, w, h, color, lines) in dims.items():
        box(x, y, w, h, name, lines, color)
    ax.text(4.0, 6.45, "dim_date is role-playing:\nused for order date AND ship date", fontsize=8.5, color="#6A6A80",
            style="italic", ha="left", va="center")
    ax.set_title("Star schema of the Sales data warehouse", fontsize=14, weight="bold", pad=6)
    plt.tight_layout(); plt.show()

draw_star_schema()
''')

    md(r'''
### 4.2 Building the dimension tables (ETL)
This is the **ETL** step — *Extract* from the clean table, *Transform* into dimensions with surrogate keys, *Load* into a database.
''')
    code(r'''
# ---- dim_date : a full calendar covering every order AND ship date --------------------
cal = pd.date_range(clean["Order Date"].min(), clean["Ship Date"].max(), freq="D")
dim_date = pd.DataFrame({"date": cal})
dim_date["date_key"]    = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
dim_date["year"]        = dim_date["date"].dt.year
dim_date["quarter"]     = dim_date["date"].dt.quarter
dim_date["month"]       = dim_date["date"].dt.month
dim_date["month_name"]  = dim_date["date"].dt.month_name()
dim_date["day"]         = dim_date["date"].dt.day
dim_date["day_of_week"] = dim_date["date"].dt.day_name()
dim_date["is_weekend"]  = (dim_date["date"].dt.dayofweek >= 5).astype(int)
dim_date = dim_date[["date_key", "date", "year", "quarter", "month", "month_name", "day", "day_of_week", "is_weekend"]]

# ---- dim_customer ----------------------------------------------------------------------
dim_customer = (clean.drop_duplicates("Customer ID")[["Customer ID", "Customer Name", "Segment"]]
                .sort_values("Customer ID").reset_index(drop=True)
                .rename(columns={"Customer ID": "customer_id", "Customer Name": "customer_name", "Segment": "segment"}))
dim_customer.insert(0, "customer_key", np.arange(1, len(dim_customer) + 1))

# ---- dim_product (one name per product ID = its most frequent name) --------------------
best_name = clean.groupby("Product ID")["Product Name"].agg(lambda s: s.value_counts().idxmax())
dim_product = (clean.drop_duplicates("Product ID")[["Product ID", "Category", "Sub-Category"]]
               .assign(product_name=lambda d: d["Product ID"].map(best_name))
               .sort_values("Product ID").reset_index(drop=True)
               .rename(columns={"Product ID": "product_id", "Category": "category", "Sub-Category": "sub_category"}))
dim_product.insert(0, "product_key", np.arange(1, len(dim_product) + 1))
dim_product = dim_product[["product_key", "product_id", "product_name", "category", "sub_category"]]

# ---- dim_geography (+ scraped population) ----------------------------------------------
geo_cols = ["Country", "Region", "State", "City", "Postal Code"]
dim_geography = (clean[geo_cols].drop_duplicates().sort_values(geo_cols).reset_index(drop=True)
                 .merge(state_pop.rename(columns={"state": "State", "population_2020": "state_population_2020"}),
                        on="State", how="left")
                 .rename(columns={"Country": "country", "Region": "region", "State": "state",
                                  "City": "city", "Postal Code": "postal_code"}))
dim_geography.insert(0, "geo_key", np.arange(1, len(dim_geography) + 1))

# ---- dim_ship_mode ---------------------------------------------------------------------
dim_ship_mode = pd.DataFrame({"ship_mode": sorted(clean["Ship Mode"].unique())})
dim_ship_mode.insert(0, "ship_mode_key", np.arange(1, len(dim_ship_mode) + 1))

for name, d in {"dim_date": dim_date, "dim_customer": dim_customer, "dim_product": dim_product,
                "dim_geography": dim_geography, "dim_ship_mode": dim_ship_mode}.items():
    print(f"{name:<14} {len(d):>6,} rows")
dim_geography.head(3)
''')

    md(r'''
### 4.3 Building the fact table
Each clean row is matched to its dimension keys. We use `validate="m:1"` so pandas **raises an error if any join would duplicate rows** — a cheap guard against the classic "fan-out" bug that inflates totals.
''')
    code(r'''
f = clean.sort_values(["Order Date", "Order ID", "Product ID"]).reset_index(drop=True).copy()
f["order_date_key"] = f["Order Date"].dt.strftime("%Y%m%d").astype(int)
f["ship_date_key"]  = f["Ship Date"].dt.strftime("%Y%m%d").astype(int)

f = (f.merge(dim_customer[["customer_key", "customer_id"]], left_on="Customer ID", right_on="customer_id", validate="m:1")
       .merge(dim_product[["product_key", "product_id"]], left_on="Product ID", right_on="product_id", validate="m:1")
       .merge(dim_geography[["geo_key", "country", "state", "city", "postal_code"]],
              left_on=["Country", "State", "City", "Postal Code"],
              right_on=["country", "state", "city", "postal_code"], validate="m:1")
       .merge(dim_ship_mode, left_on="Ship Mode", right_on="ship_mode", validate="m:1"))

fact_sales = pd.DataFrame({
    "sales_key":      np.arange(1, len(f) + 1),
    "order_id":       f["Order ID"].values,
    "order_date_key": f["order_date_key"].values,
    "ship_date_key":  f["ship_date_key"].values,
    "customer_key":   f["customer_key"].values,
    "product_key":    f["product_key"].values,
    "geo_key":        f["geo_key"].values,
    "ship_mode_key":  f["ship_mode_key"].values,
    "sales":          f["Sales"].values,
    "ship_days":      f["ship_days"].values,
})

# Reconciliation: the warehouse must agree exactly with the clean table
assert len(fact_sales) == len(clean), "row count mismatch"
assert abs(fact_sales["sales"].sum() - clean["Sales"].sum()) < 1e-6, "sales total mismatch"
print(f"✅ fact_sales: {len(fact_sales):,} rows · total sales ${fact_sales['sales'].sum():,.2f} (matches the clean table exactly)")
fact_sales.head()
''')

    md(r'''
### 4.4 Loading into a SQL database
We create real tables with **primary keys, foreign keys and indexes** in SQLite (built into Python — nothing to install) and switch foreign-key enforcement **on**, so the database itself rejects any row that points to a non-existent dimension member.
''')
    code(r'''
DDL = """
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY, date TEXT NOT NULL, year INTEGER, quarter INTEGER, month INTEGER,
    month_name TEXT, day INTEGER, day_of_week TEXT, is_weekend INTEGER);
CREATE TABLE dim_customer (
    customer_key INTEGER PRIMARY KEY, customer_id TEXT NOT NULL UNIQUE, customer_name TEXT, segment TEXT);
CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY, product_id TEXT NOT NULL UNIQUE, product_name TEXT, category TEXT, sub_category TEXT);
CREATE TABLE dim_geography (
    geo_key INTEGER PRIMARY KEY, country TEXT, region TEXT, state TEXT, city TEXT, postal_code TEXT,
    state_population_2020 INTEGER);
CREATE TABLE dim_ship_mode (
    ship_mode_key INTEGER PRIMARY KEY, ship_mode TEXT NOT NULL UNIQUE);
CREATE TABLE fact_sales (
    sales_key      INTEGER PRIMARY KEY,
    order_id       TEXT    NOT NULL,
    order_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    ship_date_key  INTEGER NOT NULL REFERENCES dim_date(date_key),
    customer_key   INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    product_key    INTEGER NOT NULL REFERENCES dim_product(product_key),
    geo_key        INTEGER NOT NULL REFERENCES dim_geography(geo_key),
    ship_mode_key  INTEGER NOT NULL REFERENCES dim_ship_mode(ship_mode_key),
    sales          REAL    NOT NULL,
    ship_days      INTEGER);
CREATE INDEX ix_fact_order_date ON fact_sales(order_date_key);
CREATE INDEX ix_fact_customer   ON fact_sales(customer_key);
CREATE INDEX ix_fact_product    ON fact_sales(product_key);
CREATE INDEX ix_fact_geo        ON fact_sales(geo_key);
"""
con = sqlite3.connect(":memory:")
con.execute("PRAGMA foreign_keys = ON")
con.executescript(DDL)

dim_date_sql = dim_date.assign(date=dim_date["date"].dt.strftime("%Y-%m-%d"))
for name, d in [("dim_date", dim_date_sql), ("dim_customer", dim_customer), ("dim_product", dim_product),
                ("dim_geography", dim_geography), ("dim_ship_mode", dim_ship_mode), ("fact_sales", fact_sales)]:
    d.to_sql(name, con, if_exists="append", index=False)
con.commit()

q = lambda sql, params=(): pd.read_sql_query(sql, con, params=params)

violations = con.execute("PRAGMA foreign_key_check").fetchall()
print("Foreign-key violations:", violations if violations else "none ✅")
q("""SELECT 'fact_sales' AS "table", COUNT(*) AS rows FROM fact_sales
     UNION ALL SELECT 'dim_date', COUNT(*) FROM dim_date
     UNION ALL SELECT 'dim_customer', COUNT(*) FROM dim_customer
     UNION ALL SELECT 'dim_product', COUNT(*) FROM dim_product
     UNION ALL SELECT 'dim_geography', COUNT(*) FROM dim_geography
     UNION ALL SELECT 'dim_ship_mode', COUNT(*) FROM dim_ship_mode""")
''')

    md(r'''
### 4.5 OLAP operations
**OLAP** (Online Analytical Processing) means slicing the *cube* of Date × Product × Geography × Customer in different ways. The five classic operations:

| Operation | Meaning | Example below |
|---|---|---|
| **Roll-up** | Aggregate to a *coarser* level | month → quarter → year → grand total |
| **Drill-down** | Go to a *finer* level | Category → Sub-Category |
| **Slice** | Fix **one** dimension to one value | only year 2018 |
| **Dice** | Fix **several** dimensions to sets of values | West/East × Technology/Furniture × 2017–18 |
| **Pivot** | Rotate axes to cross-tabulate | Category × Region |
''')
    md(r'''
**① Roll-up** — revenue by year and quarter, with year sub-totals and a grand total.
*(Databases like MySQL/Oracle have `GROUP BY ROLLUP`; SQLite does not, so we build the same result with `UNION ALL`.)*
''')
    code(r'''
rollup = q("""
SELECT d.year AS year, d.quarter AS quarter, ROUND(SUM(f.sales), 0) AS revenue
FROM fact_sales f JOIN dim_date d ON f.order_date_key = d.date_key
GROUP BY d.year, d.quarter
UNION ALL
SELECT d.year, NULL, ROUND(SUM(f.sales), 0)
FROM fact_sales f JOIN dim_date d ON f.order_date_key = d.date_key
GROUP BY d.year
UNION ALL
SELECT NULL, NULL, ROUND(SUM(f.sales), 0) FROM fact_sales f
ORDER BY year, quarter
""")
rollup["level"] = np.where(rollup["year"].isna(), "Grand total", np.where(rollup["quarter"].isna(), "Year total", "Quarter"))
rollup["year"] = rollup["year"].astype("Int64").astype(str).replace("<NA>", "ALL")
rollup["quarter"] = rollup["quarter"].astype("Int64").astype(str).replace("<NA>", "ALL")
rollup
''')
    md(r'''
**② Drill-down** — start at Category, then drill into the biggest one.
''')
    code(r'''
cat_totals = q("""SELECT p.category, ROUND(SUM(f.sales),0) AS revenue
                  FROM fact_sales f JOIN dim_product p ON f.product_key = p.product_key
                  GROUP BY p.category ORDER BY revenue DESC""")
top_cat = cat_totals.iloc[0]["category"]
sub_totals = q("""SELECT p.sub_category, ROUND(SUM(f.sales),0) AS revenue
                  FROM fact_sales f JOIN dim_product p ON f.product_key = p.product_key
                  WHERE p.category = ? GROUP BY p.sub_category ORDER BY revenue DESC""", (top_cat,))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
sns.barplot(data=cat_totals, y="category", x="revenue", color=PALETTE[0], ax=axes[0])
axes[0].set(title="Level 1 — revenue by Category", ylabel="", xlabel="Revenue"); axes[0].xaxis.set_major_formatter(money)
sns.barplot(data=sub_totals, y="sub_category", x="revenue", color=PALETTE[1], ax=axes[1])
axes[1].set(title=f"Level 2 — drill-down into '{top_cat}'", ylabel="", xlabel="Revenue"); axes[1].xaxis.set_major_formatter(money)
plt.tight_layout(); plt.show()
''')
    md(r'''
**③ Slice** — fix one dimension (year = 2018) and look at the rest.  **④ Dice** — fix several at once.
''')
    code(r'''
slice_2018 = q("""SELECT g.region, ROUND(SUM(f.sales),0) AS revenue
                  FROM fact_sales f JOIN dim_date d ON f.order_date_key = d.date_key
                  JOIN dim_geography g ON f.geo_key = g.geo_key
                  WHERE d.year = 2018 GROUP BY g.region ORDER BY revenue DESC""")
print("SLICE — year = 2018, revenue by region")
display(slice_2018)

dice = q("""SELECT d.year, g.region, p.category, ROUND(SUM(f.sales),0) AS revenue
            FROM fact_sales f JOIN dim_date d ON f.order_date_key = d.date_key
            JOIN dim_geography g ON f.geo_key = g.geo_key
            JOIN dim_product p ON f.product_key = p.product_key
            WHERE g.region IN ('West','East') AND p.category IN ('Technology','Furniture') AND d.year IN (2017, 2018)
            GROUP BY d.year, g.region, p.category ORDER BY d.year, g.region, p.category""")
print("DICE — regions {West, East} × categories {Technology, Furniture} × years {2017, 2018}")
display(dice)
''')
    md(r'''
**⑤ Pivot** — rotate Region into columns to make a cross-tab, with totals.
''')
    code(r'''
cr = q("""SELECT p.category, g.region, SUM(f.sales) AS revenue
          FROM fact_sales f JOIN dim_product p ON f.product_key = p.product_key
          JOIN dim_geography g ON f.geo_key = g.geo_key GROUP BY p.category, g.region""")
pivot = cr.pivot_table(index="category", columns="region", values="revenue", aggfunc="sum",
                       margins=True, margins_name="TOTAL").round(0)
display(pivot.style.format("${:,.0f}"))

fig, ax = plt.subplots(figsize=(7.5, 3.6))
sns.heatmap(pivot.drop(index="TOTAL", columns="TOTAL"), annot=True, fmt=",.0f",
            cmap=sns.light_palette(PALETTE[0], as_cmap=True), cbar_kws={"label": "Revenue ($)"}, ax=ax)
ax.set(title="Pivot: revenue by Category × Region", xlabel="", ylabel="")
plt.tight_layout(); plt.show()
''')
    md(r'''
**⑥ Using the scraped data** — because population lives in `dim_geography`, one SQL join answers a question the original CSV never could: *which states buy the most **per resident**?*
''')
    code(r'''
percap = q("""SELECT g.state, ROUND(SUM(f.sales),0) AS revenue, MAX(g.state_population_2020) AS population,
                     ROUND(SUM(f.sales) * 1000.0 / MAX(g.state_population_2020), 2) AS revenue_per_1000_residents
              FROM fact_sales f JOIN dim_geography g ON f.geo_key = g.geo_key
              GROUP BY g.state ORDER BY revenue_per_1000_residents DESC""")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
t1 = percap.sort_values("revenue", ascending=False).head(10)
sns.barplot(data=t1, y="state", x="revenue", color=PALETTE[0], ax=axes[0])
axes[0].set(title="Top 10 states by TOTAL revenue", ylabel="", xlabel="Revenue"); axes[0].xaxis.set_major_formatter(money)
t2 = percap.head(10)
sns.barplot(data=t2, y="state", x="revenue_per_1000_residents", color=PALETTE[3], ax=axes[1])
axes[1].set(title="Top 10 states by revenue PER 1,000 RESIDENTS", ylabel="", xlabel="$ per 1,000 residents")
plt.tight_layout(); plt.show()

hidden = [s for s in t2["state"] if s not in set(t1["state"])]
print(f"{len(hidden)} of the 10 best states per resident are NOT in the top-10 by total revenue: {', '.join(hidden)}.")
print("Total revenue favours big states; the per-resident view surfaces smaller states that punch above their weight.")
''')
    md(r'''
**Why this matters:** the warehouse answered six different questions with simple `JOIN`s and `GROUP BY`s, never touching the raw CSV again, and the integrity checks guarantee the numbers agree with the source. That is the whole point of a star schema.
''')

    # ------------------------------------------------------------------ 5
    md(r'''
## 📊 5. Exploratory data analysis (EDA)

EDA means *looking before modelling*: understand each variable alone (univariate), then in pairs and groups (bivariate / multivariate), and form hypotheses for the mining steps.
''')
    md(r'''
### 5.1 Univariate — what does the data look like?
''')
    code(r'''
import matplotlib.dates as mdates
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat[:4], ["Category", "Segment", "Region", "Ship Mode"]):
    order = clean[col].value_counts().index
    sns.countplot(data=clean, y=col, order=order, color=PALETTE[0], ax=ax)
    ax.set(title=f"Rows by {col}", xlabel="Number of order lines", ylabel="")
sns.histplot(clean["Sales"], bins=60, log_scale=(True, False), ax=axes[1, 1], color=PALETTE[1])
axes[1, 1].set(title="Sales per line (log x-axis)", xlabel="Sales ($, log scale)")
sns.countplot(data=clean, x="ship_days", color=PALETTE[2], ax=axes[1, 2])
axes[1, 2].set(title="Delivery time", xlabel="Days from order to ship", ylabel="Order lines")
plt.tight_layout(); plt.show()
''')
    md(r'''
### 5.2 What sells? — revenue by product
''')
    code(r'''
rev_sub = clean.groupby("Sub-Category")["Sales"].sum().sort_values()
cat_of = clean.drop_duplicates("Sub-Category").set_index("Sub-Category")["Category"]
cat_color = dict(zip(sorted(clean["Category"].unique()), [PALETTE[0], PALETTE[1], PALETTE[3]]))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(rev_sub.index, rev_sub.values, color=[cat_color[cat_of[s]] for s in rev_sub.index])
ax.bar_label(bars, labels=[f"${v/1000:,.0f}k  ({v/rev_sub.sum():.0%})" for v in rev_sub.values], padding=4, fontsize=8.5)
ax.xaxis.set_major_formatter(money); ax.set_xlim(0, rev_sub.max() * 1.22)
ax.set(title="Revenue by sub-category (colour = category)", xlabel="Revenue", ylabel="")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_color.values()], labels=list(cat_color), loc="lower right")
plt.tight_layout(); plt.show()
''')
    md(r'''
### 5.3 When does it sell? — trends and seasonality
''')
    code(r'''
monthly = clean.set_index("Order Date")["Sales"].resample("MS").sum()
yearly = clean.groupby("order_year")["Sales"].sum()
piv = clean.pivot_table(index="order_year", columns="order_month", values="Sales", aggfunc="sum")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
wd = clean.groupby("order_weekday")["Sales"].sum().reindex(weekday_order)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(monthly.index, monthly.values, marker="o", ms=3.5, color=PALETTE[0])
axes[0, 0].set(title="Monthly revenue", ylabel="Revenue"); axes[0, 0].yaxis.set_major_formatter(money)
axes[0, 0].xaxis.set_major_locator(mdates.YearLocator()); axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

bars = axes[0, 1].bar(yearly.index.astype(str), yearly.values, color=PALETTE[1])
growth = yearly.pct_change() * 100
axes[0, 1].bar_label(bars, labels=[f"${v/1000:,.0f}k" + ("" if np.isnan(g) else f"\n{g:+.0f}% YoY") for v, g in zip(yearly.values, growth)], padding=3)
axes[0, 1].set(title="Yearly revenue and growth", ylabel="Revenue"); axes[0, 1].yaxis.set_major_formatter(money)
axes[0, 1].set_ylim(0, yearly.max() * 1.25)

sns.heatmap(piv / 1000, annot=True, fmt=".0f", cmap=sns.light_palette(PALETTE[0], as_cmap=True), ax=axes[1, 0],
            cbar_kws={"label": "Revenue ($ thousands)"})
axes[1, 0].set(title="Seasonality heat-map: year × month", xlabel="Month", ylabel="")

axes[1, 1].bar(wd.index.str[:3], wd.values, color=PALETTE[3])
axes[1, 1].set(title="Revenue by weekday", ylabel="Revenue"); axes[1, 1].yaxis.set_major_formatter(money)
plt.tight_layout(); plt.show()
''')
    md(r'''
### 5.4 Who buys? — customers and the 80/20 rule
''')
    code(r'''
cust = clean.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False)
cum = cust.cumsum() / cust.sum()
pct_for_80 = (cum <= 0.80).sum() / len(cust)
top20_share = cust.head(int(round(0.2 * len(cust)))).sum() / cust.sum()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
top10 = cust.head(10).sort_values()
axes[0].barh(top10.index, top10.values, color=PALETTE[5])
axes[0].xaxis.set_major_formatter(money); axes[0].set(title="Top 10 customers by revenue", xlabel="Revenue")
x = np.arange(1, len(cum) + 1) / len(cum)
axes[1].plot(x, cum.values, color=PALETTE[0], lw=2.5)
axes[1].plot([0, 1], [0, 1], "--", color="#9A9AAE", label="perfect equality")
axes[1].axhline(0.8, color=PALETTE[2], lw=1.2); axes[1].axvline(pct_for_80, color=PALETTE[2], lw=1.2)
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1)); axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
axes[1].set(title=f"Pareto curve — {pct_for_80:.0%} of customers bring 80% of revenue",
            xlabel="Share of customers (largest first)", ylabel="Share of revenue"); axes[1].legend()
plt.tight_layout(); plt.show()
print(f"The top 20% of customers account for {top20_share:.0%} of revenue.")
''')
    md(r'''
### 5.5 Relationships — what drives sales?
''')
    code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
order = clean.groupby("Sub-Category")["log_sales"].median().sort_values(ascending=False).index
sns.boxplot(data=clean, y="Sub-Category", x="log_sales", order=order, color=PALETTE[1], fliersize=1.5, ax=axes[0])
axes[0].set(title="Sale size by sub-category (log scale)", xlabel="log(1 + Sales)", ylabel="")

avg = clean.pivot_table(index="Category", columns="Region", values="Sales", aggfunc="mean")
sns.heatmap(avg, annot=True, fmt=".0f", cmap=sns.light_palette(PALETTE[0], as_cmap=True), ax=axes[1],
            cbar_kws={"label": "Average sale ($)"})
axes[1].set(title="Average sale per line: Category × Region", xlabel="", ylabel="")
plt.tight_layout(); plt.show()
''')
    code(r'''
num_cols = ["Sales", "log_sales", "ship_days", "order_month", "order_quarter", "order_year"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
sns.heatmap(clean[num_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0])
axes[0].set(title="Correlation of numeric features")
sns.boxplot(data=clean, x="Ship Mode", y="ship_days", order=["Same Day", "First Class", "Second Class", "Standard Class"],
            color=PALETTE[2], ax=axes[1])
axes[1].set(title="Ship mode vs delivery time", xlabel="", ylabel="Days to ship")
plt.tight_layout(); plt.show()
''')
    md(r'''
**Reading the relationships:** correlation among the numeric columns is weak — *when* or *how fast* something ships barely changes the sale size. What really separates a $10 sale from a $1,000 sale is **what** was bought (the sub-category box-plots barely overlap). This is a key hypothesis for the models: **product type will dominate the predictions.**
''')
    md(r'''
### 5.6 EDA summary — key findings
''')
    code(r'''
best_month = monthly.idxmax()
best_cat = clean.groupby("Category")["Sales"].sum().idxmax()
best_sub = rev_sub.idxmax()
best_region = clean.groupby("Region")["Sales"].sum().idxmax()
cagr = (yearly.iloc[-1] / yearly.iloc[0]) ** (1 / (len(yearly) - 1)) - 1
nov_dec = monthly[monthly.index.month.isin([11, 12])].sum() / monthly.sum()

print(f"1. Total revenue ${clean['Sales'].sum():,.0f} from {clean['Order ID'].nunique():,} orders and {clean['Customer ID'].nunique()} customers.")
print(f"2. Revenue grew from ${yearly.iloc[0]:,.0f} ({yearly.index[0]}) to ${yearly.iloc[-1]:,.0f} ({yearly.index[-1]}) — about {cagr:.0%} per year.")
print(f"3. Strong seasonality: November + December alone bring {nov_dec:.0%} of all revenue; best month ever = {best_month:%B %Y}.")
print(f"4. {best_cat} is the top category; {best_sub} the top sub-category; {best_region} the top region.")
print(f"5. Customer revenue is concentrated: top 20% of customers = {top20_share:.0%} of revenue.")
print(f"6. Sale size depends on the product, not on timing or shipping (weak numeric correlations).")
''')
