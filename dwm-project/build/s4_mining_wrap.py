"""Sections 8-11: clustering, association rules (Apriori), bonus, conclusions."""


def add(md, code):
    # ------------------------------------------------------------------ 8
    md(r'''
## 🧩 8. Clustering

**Clustering** is *unsupervised*: there is no label to predict — the algorithm groups similar records on its own. Business question:

> *Which kinds of customers do we have, and how should we treat each group?*

We use the classic **RFM** representation of a customer:

| Letter | Meaning | Good customer = |
|---|---|---|
| **R**ecency | days since the last order | low |
| **F**requency | number of orders | high |
| **M**onetary | total money spent | high |
''')
    code(r'''
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

snapshot = clean["Order Date"].max() + pd.Timedelta(days=1)
rfm = clean.groupby("Customer ID").agg(
    recency=("Order Date", lambda s: (snapshot - s.max()).days),
    frequency=("Order ID", "nunique"),
    monetary=("Sales", "sum"),
)
print(f"{len(rfm)} customers · analysis date = {snapshot:%d %b %Y}")
rfm.describe().round(1)
''')
    md(r'''
### 8.1 Preparing the features
Distance-based algorithms are sensitive to scale and skew (monetary is in thousands, frequency in single digits, and all three are right-skewed). So: **log-transform, then standardise** — the same lesson as in pre-processing.
''')
    code(r'''
X_rfm = StandardScaler().fit_transform(np.log1p(rfm))

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for j, col in enumerate(["recency", "frequency", "monetary"]):
    sns.histplot(rfm[col], bins=30, ax=axes[0, j], color=PALETTE[j]); axes[0, j].set(title=f"{col} (raw)", xlabel="")
    sns.histplot(X_rfm[:, j], bins=30, ax=axes[1, j], color=PALETTE[j]); axes[1, j].set(title=f"{col} (log + scaled)", xlabel="")
plt.tight_layout(); plt.show()
''')
    md(r'''
### 8.2 Choosing the number of clusters (k)
K-Means needs k in advance. We do not guess — we measure three criteria for k = 2…8:

* **Elbow (inertia)** — total squared distance to the centroids; look for the "bend" where adding clusters stops helping much.
* **Silhouette score** — how well each point fits its own cluster vs the next-nearest (−1…1, **higher** is better).
* **Davies–Bouldin index** — average cluster overlap (**lower** is better).
''')
    code(r'''
ks = list(range(2, 9))
inertia, sil, dbi = [], [], []
for k in ks:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_rfm)
    inertia.append(km_k.inertia_)
    sil.append(silhouette_score(X_rfm, km_k.labels_))
    dbi.append(davies_bouldin_score(X_rfm, km_k.labels_))

# Marketing needs at least 3 segments (k=2 only splits "active" vs "inactive"), so we pick the best silhouette among 3..6
candidates = [k for k in ks if 3 <= k <= 6]
best_k = max(candidates, key=lambda k: sil[ks.index(k)])

fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
axes[0].plot(ks, inertia, "-o", color=PALETTE[0]); axes[0].set(title="Elbow method", xlabel="k", ylabel="inertia")
axes[1].plot(ks, sil, "-o", color=PALETTE[1]); axes[1].set(title="Silhouette score (higher = better)", xlabel="k")
axes[2].plot(ks, dbi, "-o", color=PALETTE[3]); axes[2].set(title="Davies–Bouldin (lower = better)", xlabel="k")
for ax in axes:
    ax.axvline(best_k, color=PALETTE[2], ls="--", lw=1.5); ax.set_xticks(ks)
plt.tight_layout(); plt.show()
print(f"Chosen k = {best_k} (best silhouette = {sil[ks.index(best_k)]:.2f} among k = 3…6).")
pd.DataFrame({"k": ks, "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi}).round(3)
''')
    md(r'''
### 8.3 K-Means result and segment profiles
''')
    code(r'''
km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_rfm)
rfm["cluster"] = km.labels_

# name the clusters by ranking them: high monetary + high frequency + LOW recency = best
prof = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
score = prof["monetary"].rank() + prof["frequency"].rank() - prof["recency"].rank()
seg_names = ["Champions", "Loyal", "Potential", "At-Risk", "Lost", "Dormant"]
name_map = {c: seg_names[i] for i, c in enumerate(score.sort_values(ascending=False).index)}
rfm["segment"] = rfm["cluster"].map(name_map)

profile = (rfm.groupby("segment").agg(customers=("monetary", "size"), avg_recency_days=("recency", "mean"),
                                      avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean"),
                                      total_revenue=("monetary", "sum")))
profile["revenue_share"] = profile["total_revenue"] / profile["total_revenue"].sum()
profile = profile.loc[[n for n in seg_names if n in profile.index]]
profile.round(2)
''')
    code(r'''
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pts = pca.fit_transform(X_rfm)
seg_order = list(profile.index)
seg_color = dict(zip(seg_order, [PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[2], PALETTE[5], "#9A9AAE"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for s in seg_order:
    m = (rfm["segment"] == s).values
    axes[0].scatter(pts[m, 0], pts[m, 1], s=22, alpha=0.75, color=seg_color[s], label=f"{s} ({m.sum()})")
axes[0].set(title=f"Customer segments in 2-D (PCA, {pca.explained_variance_ratio_.sum():.0%} of variance)",
            xlabel="principal component 1", ylabel="principal component 2"); axes[0].legend()

centroids = pd.DataFrame(X_rfm, columns=["recency", "frequency", "monetary"]).groupby(rfm["segment"].values).mean().loc[seg_order]
sns.heatmap(centroids, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1], cbar_kws={"label": "standardised average"})
axes[1].set(title="Segment profile (red = above average, blue = below)", ylabel="")
plt.tight_layout(); plt.show()
''')
    code(r'''
for seg, row in profile.iterrows():
    print(f"• {seg:<10} {int(row['customers']):>4} customers · last order ~{row['avg_recency_days']:.0f} days ago · "
          f"{row['avg_orders']:.1f} orders · ${row['avg_spend']:,.0f} each · {row['revenue_share']:.0%} of revenue")
''')
    md(r'''
**Marketing actions per segment** — *Champions*: reward and protect (early access, loyalty perks) · *Loyal*: upsell and cross-sell · *Potential*: nudge towards a second/third order · *At-Risk / Lost*: win-back campaigns with a personal touch.
''')
    md(r'''
### 8.4 Hierarchical clustering (comparison)
**Agglomerative (Ward) clustering** starts with every customer alone and repeatedly merges the two closest groups, producing a tree (**dendrogram**). It needs no random start, and we can compare it with K-Means: if two very different algorithms find similar groups, the structure is real.
''')
    code(r'''
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X_rfm)
comparison_cluster = pd.DataFrame({
    "silhouette": [silhouette_score(X_rfm, km.labels_), silhouette_score(X_rfm, agg.labels_)],
    "Davies–Bouldin": [davies_bouldin_score(X_rfm, km.labels_), davies_bouldin_score(X_rfm, agg.labels_)],
}, index=["K-Means", "Agglomerative (Ward)"]).round(3)
ari = adjusted_rand_score(km.labels_, agg.labels_)
display(comparison_cluster)
print(f"Agreement between the two algorithms (Adjusted Rand Index): {ari:.2f}  (1 = identical groupings, 0 = unrelated)")
sil_best = sil[ks.index(best_k)]
if ari >= 0.6:
    print("Reading: strong agreement → the customer groups are real, well-separated structure.")
else:
    print(f"Reading: only {'moderate' if ari >= 0.3 else 'weak'} agreement (and a silhouette of {sil_best:.2f}) → customers form a CONTINUUM rather than "
          "sharply separated islands. The segments are still useful as practical buckets, but their borders are soft.")

rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X_rfm), size=120, replace=False)
Z = linkage(X_rfm[sample_idx], method="ward")
fig, ax = plt.subplots(figsize=(13, 4.5))
dendrogram(Z, truncate_mode="lastp", p=24, leaf_rotation=90, leaf_font_size=9, color_threshold=0.7 * max(Z[:, 2]), ax=ax)
ax.set(title="Dendrogram (120 random customers, last 24 merges)", xlabel="cluster size / customer", ylabel="merge distance")
plt.tight_layout(); plt.show()
''')

    # ------------------------------------------------------------------ 9
    md(r'''
## 🛍️ 9. Association rules — the Apriori algorithm

**Association-rule mining** finds items that occur *together*. The classic example is market-basket analysis: *"customers who buy A also buy B."* Here a **basket = one order**, and the **items = the sub-categories** in it (a 47% share of orders contain more than one sub-category, so there is something to find).

### Key measures
For a rule **A ⇒ B** (N = number of orders):

| Measure | Formula | Meaning |
|---|---|---|
| **Support** | `count(A and B) / N` | How common is the combination? |
| **Confidence** | `count(A and B) / count(A)` | Of orders with A, how many also have B? |
| **Lift** | `confidence / support(B)` | How much more often B appears *given A* than by chance. **Lift > 1** = positive association, **= 1** = independent, **< 1** = they avoid each other |

### The Apriori principle
> *If an itemset is frequent, all of its subsets are frequent* — equivalently, **if a set is infrequent, no larger set containing it can be frequent.**

That lets the algorithm **prune** huge numbers of candidates: build frequent 1-item sets → combine them into candidate 2-item sets → keep the frequent ones → build 3-item candidates, and so on.
''')
    code(r'''
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

baskets = clean.groupby("Order ID")["Sub-Category"].apply(lambda s: sorted(set(s))).tolist()
N = len(baskets)
sizes = pd.Series([len(b) for b in baskets])
print(f"{N:,} baskets · average {sizes.mean():.2f} distinct sub-categories per order · {(sizes > 1).mean():.0%} of orders have 2 or more")

te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)   # True/False matrix: order × item
display(onehot.head(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
vc = sizes.value_counts().sort_index()
axes[0].bar(vc.index.astype(str), vc.values, color=PALETTE[1]); axes[0].set(title="Basket size (distinct sub-categories per order)", xlabel="items in basket", ylabel="orders")
supp = onehot.mean().sort_values()
axes[1].barh(supp.index, supp.values, color=PALETTE[0]); axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1))
axes[1].set(title="Support of single items", xlabel="share of orders containing the item")
plt.tight_layout(); plt.show()
''')
    md(r'''
### 9.1 Choosing minimum support
Too high → nothing interesting survives; too low → thousands of rules by chance. We look at how many frequent itemsets exist at different thresholds.
''')
    code(r'''
sens = []
for s in [0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
    fi = apriori(onehot, min_support=s, use_colnames=True)
    lens = fi["itemsets"].apply(len)
    sens.append({"min_support": s, "≈ orders": int(round(s * N)), "1-itemsets": int((lens == 1).sum()),
                 "2-itemsets": int((lens == 2).sum()), "3-itemsets": int((lens == 3).sum()), "total": len(fi)})
pd.DataFrame(sens).set_index("min_support")
''')
    code(r'''
MIN_SUPPORT = 0.01          # an itemset must appear in at least ~1% of orders (≈ 49 orders)
freq = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
try:
    rules = association_rules(freq, num_itemsets=N, metric="confidence", min_threshold=0.10)
except TypeError:                                              # older mlxtend versions have no num_itemsets argument
    rules = association_rules(freq, metric="confidence", min_threshold=0.10)

rules = rules[rules["lift"] > 1].copy()                        # keep only positive associations
rules["rule"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s))) + "  ⇒  " + rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
rules["orders"] = (rules["support"] * N).round().astype(int)
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
print(f"min_support = {MIN_SUPPORT:.0%} → {len(freq)} frequent itemsets → {len(rules)} rules with lift > 1")
rules[["rule", "orders", "support", "confidence", "lift"]].head(15).round(3)
''')
    code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
sc = axes[0].scatter(rules["support"], rules["confidence"], c=rules["lift"], s=60, cmap="magma_r", edgecolor="white")
plt.colorbar(sc, ax=axes[0], label="lift")
axes[0].set(title="All rules: support vs confidence (colour = lift)", xlabel="support", ylabel="confidence")

A = onehot.values.astype(float)
co = (A.T @ A) / N                                           # P(A and B)
p = A.mean(axis=0)
lift_arr = co / np.outer(p, p)
np.fill_diagonal(lift_arr, np.nan)                             # an item with itself is not interesting
lift_mat = pd.DataFrame(lift_arr, index=onehot.columns, columns=onehot.columns)
sns.heatmap(lift_mat, cmap="RdBu_r", center=1, annot=True, fmt=".2f", annot_kws={"size": 6.5}, linewidths=0.3,
            cbar_kws={"label": "lift (1 = independent)"}, ax=axes[1])
axes[1].set(title="Pairwise lift between sub-categories", xlabel="", ylabel="")
plt.setp(axes[1].get_xticklabels(), rotation=60, ha="right", fontsize=8); plt.setp(axes[1].get_yticklabels(), fontsize=8)
plt.tight_layout(); plt.show()
''')
    md(r'''
### 9.2 A caution about rare pairs — and checking the formulas by hand
The heat-map shows some pairs with a *higher* lift than any rule in the table above. Why are they missing from the rules? Because they fall **below the minimum-support threshold**: they co-occur in very few orders, so the lift is easily a fluke. This is exactly why Apriori needs a support cut-off.
We take the top single-item rule and recompute its measures directly from the basket counts — if our understanding of support / confidence / lift is right, the numbers must match the library's.
''')
    code(r'''
pairs = [(lift_mat.index[i], lift_mat.columns[j], lift_mat.iloc[i, j], int(round(co[i, j] * N)))
         for i in range(len(lift_mat)) for j in range(i + 1, len(lift_mat))]
rare = pd.DataFrame(pairs, columns=["item A", "item B", "lift", "orders together"]).sort_values("lift", ascending=False).head(5)
print("Highest pairwise lifts overall — note how few orders support them:")
display(rare.reset_index(drop=True))
print("Compare with the rules table: those pairs have lift > 1.5 but appear in only a handful of orders → unreliable, which is why we require support ≥ 1%.\n")

single = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].iloc[0]
a, b = next(iter(single["antecedents"])), next(iter(single["consequents"]))
n_a, n_b, n_ab = int(onehot[a].sum()), int(onehot[b].sum()), int((onehot[a] & onehot[b]).sum())

support_hand    = n_ab / N
confidence_hand = n_ab / n_a
lift_hand       = confidence_hand / (n_b / N)
print(f"Rule: {a} ⇒ {b}")
print(f"  orders with {a}: {n_a} · orders with {b}: {n_b} · orders with both: {n_ab} · total orders: {N}")
print(f"  support    = {n_ab}/{N}      = {support_hand:.4f}   (library: {single['support']:.4f})")
print(f"  confidence = {n_ab}/{n_a}  = {confidence_hand:.4f}   (library: {single['confidence']:.4f})")
print(f"  lift       = {confidence_hand:.4f} / ({n_b}/{N}) = {lift_hand:.4f}   (library: {single['lift']:.4f})")
assert np.isclose(support_hand, single["support"]) and np.isclose(confidence_hand, single["confidence"]) and np.isclose(lift_hand, single["lift"])
print("✅ Hand calculation matches the library.")
''')
    md(r'''
### 9.3 Apriori from scratch
To show we understand the algorithm and are not just calling a library, here is a compact implementation of the level-wise **generate → prune → count** loop. We then verify that it finds *exactly the same* frequent itemsets as `mlxtend`, and print how many candidates each level generated vs how many survived (the Apriori pruning in action).
''')
    code(r'''
def apriori_scratch(transactions, min_support):
    n = len(transactions)
    tx = [frozenset(t) for t in transactions]
    support = lambda itemset: sum(itemset <= t for t in tx) / n

    items = sorted({i for t in tx for i in t})
    level = {frozenset([i]): support(frozenset([i])) for i in items}
    stats = [(1, len(items), sum(v >= min_support for v in level.values()))]
    level = {k: v for k, v in level.items() if v >= min_support}
    frequent, k = dict(level), 2
    while level:
        prev = list(level)
        # JOIN: merge two frequent (k-1)-itemsets into a k-itemset
        # PRUNE: discard it unless ALL of its (k-1)-subsets are frequent (the Apriori principle)
        candidates = {a | b for a in prev for b in prev
                      if len(a | b) == k and all(frozenset(s) in level for s in combinations(a | b, k - 1))}
        level = {c: support(c) for c in candidates}
        stats.append((k, len(candidates), sum(v >= min_support for v in level.values())))
        level = {c: v for c, v in level.items() if v >= min_support}      # COUNT: keep only the frequent ones
        frequent.update(level)
        k += 1
    return frequent, stats

scratch, stats = apriori_scratch(baskets, MIN_SUPPORT)
library = {frozenset(i): s for i, s in zip(freq["itemsets"], freq["support"])}

same = set(scratch) == set(library)
max_diff = max(abs(scratch[k] - library[k]) for k in library) if same else float("nan")
print(f"From-scratch Apriori found {len(scratch)} frequent itemsets · mlxtend found {len(library)} · identical sets: {same} · max support difference: {max_diff:.1e}")
assert same, "implementations disagree"
pd.DataFrame(stats, columns=["level k", "candidates generated (after pruning)", "frequent itemsets kept"]).set_index("level k")
''')
    code(r'''
top_rule = rules.iloc[0]
print(f"Strongest rule (highest lift): {top_rule['rule']}  → lift {top_rule['lift']:.2f}, confidence {top_rule['confidence']:.0%}, in {top_rule['orders']} orders.")
print(f"Median lift across all {len(rules)} rules: {rules['lift'].median():.2f}.")
if rules["lift"].max() < 1.5:
    print("Reading: every lift is close to 1 → customers choose sub-categories almost independently. "
          "Bundles should be tested cautiously (A/B test) rather than assumed.")
else:
    print("Reading: some pairs are bought together clearly more often than chance → good candidates for bundles / cross-sell offers.")
''')

    # ------------------------------------------------------------------ 10
    md(r'''
## 🎁 10. Bonus

### 10.1 Anomaly detection (Isolation Forest)
Which months were *unusual*? **Isolation Forest** isolates points with random splits; points that are isolated quickly (few splits) are anomalies. No labels needed.
''')
    code(r'''
from sklearn.ensemble import IsolationForest

m = ts.to_frame("revenue")
m["pct_change"] = m["revenue"].pct_change().fillna(0)
Zs = StandardScaler().fit_transform(m[["revenue", "pct_change"]])
iso = IsolationForest(contamination=0.08, random_state=RANDOM_STATE).fit(Zs)
m["anomaly"] = iso.predict(Zs) == -1
flagged = m[m["anomaly"]].copy()
flagged["type"] = np.where(flagged["pct_change"] >= 0, "spike", "drop")

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.plot(m.index, m["revenue"], color="#2B2B33", lw=1.6, label="Monthly revenue")
sp, dr = flagged[flagged["type"] == "spike"], flagged[flagged["type"] == "drop"]
ax.scatter(sp.index, sp["revenue"], s=110, marker="^", color="#2E9E7B", zorder=3, label="Spike")
ax.scatter(dr.index, dr["revenue"], s=110, marker="v", color=PALETTE[0], zorder=3, label="Drop")
ax.yaxis.set_major_formatter(money); ax.set(title="Unusual months flagged by Isolation Forest", ylabel="Revenue"); ax.legend()
plt.tight_layout(); plt.show()
flagged.assign(month=flagged.index.strftime("%b %Y"), change=flagged["pct_change"].map("{:+.0%}".format))[["month", "revenue", "change", "type"]].reset_index(drop=True)
''')
    md(r'''
### 10.2 A live interactive version (optional extra)
Everything above is also packaged as a small **interactive web application** (Streamlit): upload any sales CSV and get the dashboard, customer segments, forecast and AI-written recommendations.
It is only an *extra* — the notebook is the complete, self-contained submission.

🌐 **Live demo:** https://lumi-sales-analytics.onrender.com  *(free hosting: the first load after idle time can take up to a minute)*
''')

    # ------------------------------------------------------------------ 11
    md(r'''
## ✅ 11. Conclusions
''')
    code(r'''
summary = pd.DataFrame([
    ("Web scraping",       f"{len(state_pop)} state populations scraped ({scrape_source}); joined to all {clean['State'].nunique()} sales states"),
    ("Pre-processing",     f"{len(raw):,} → {len(clean):,} rows; {int(clean['postal_imputed'].sum())} missing ZIPs imputed, {n_dupes_raw} duplicate removed, dates & ZIP types fixed, "
                           f"{int(clean['is_outlier'].sum()):,} outliers flagged and log-transformed (skew {clean['Sales'].skew():.1f} → {clean['log_sales'].skew():.1f})"),
    ("Star schema / OLAP", f"1 fact table ({len(fact_sales):,} rows) + 5 dimensions; 0 FK violations; roll-up, drill-down, slice, dice, pivot queries"),
    ("Classification",     f"6 algorithms compared; top F1 {clf_results.loc[best_clf, 'F1']:.2f} ({best_clf}), ROC-AUC {clf_results.loc[best_clf, 'ROC-AUC']:.2f}; the best {len(tied)} are a statistical tie"),
    ("Regression",         f"R² {reg_results.loc[best_reg, 'R²']:.2f} on log(Sales) ({best_reg}); monthly revenue forecast: {best_ts} wins the backtest with MAPE {ts_results.loc[best_ts, 'MAPE %']:.1f}%"),
    ("Clustering",         f"K-Means k = {best_k} chosen by elbow + silhouette ({sil[ks.index(best_k)]:.2f}); agreement with hierarchical ARI {ari:.2f}"),
    ("Association rules",  f"Apriori at {MIN_SUPPORT:.0%} support → {len(rules)} rules; top lift {rules['lift'].max():.2f}; custom implementation verified against mlxtend"),
], columns=["Technique", "Result"])
pd.set_option("display.max_colwidth", 200)
summary
''')
    md(r'''
### Business insights
1. **Plan for the year-end peak.** November–December carry a disproportionate share of revenue (see EDA 5.6) — inventory, staffing and marketing should be front-loaded for them.
2. **Protect the top customers, win back the rest.** Revenue is concentrated in a small share of customers; the *Champions* segment needs retention care, while *At-Risk / Lost* customers are an inexpensive win-back target.
3. **Product mix decides the sale size.** Both the EDA and the classifier agree that sub-category is the dominant driver of high-value sales — promote the high-ticket lines.
4. **Per-resident view reshuffles the geography.** The warehouse + scraped population surface smaller states that out-perform big ones per head (section 4.5 ⑥) — information total-revenue rankings hide.
5. **Bundle with care.** Basket analysis finds only weak co-purchase patterns; test bundles before rolling them out.

### Limitations
* **One measure only** (`Sales`): no quantity, discount, cost or profit, so we model revenue not margin, and regression R² is moderate.
* **One retailer, 2015–2018** — patterns may not transfer to other businesses or to today.
* **Population is from the 2020 Census** while sales are 2015–2018 — a reasonable proxy, not an exact match.
* Association rules are based on 17 broad sub-categories rather than individual products.

### Future work
Add quantity / discount / cost to the fact table; connect a BI tool (Power BI / Tableau) directly to the star schema; automate the scrape and ETL on a schedule; add customer-churn prediction and next-basket recommendation.

### References
* Dataset — Superstore Sales, Kaggle (rohitsahoo/sales-forecasting).
* Population data — Wikipedia, *List of U.S. states and territories by population* (CC BY-SA).
* R. Kimball & M. Ross, *The Data Warehouse Toolkit* (star schema, fact/dimension design).
* J. Han, M. Kamber & J. Pei, *Data Mining: Concepts and Techniques*.
* R. Agrawal & R. Srikant, *Fast Algorithms for Mining Association Rules* (Apriori, 1994).
* scikit-learn, XGBoost, statsmodels, mlxtend, pandas, seaborn documentation.
''')
