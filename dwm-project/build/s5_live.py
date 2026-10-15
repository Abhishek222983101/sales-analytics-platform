"""Section 12: live demo — cells with forms the presenter can change and re-run."""
import json


def add(md, code, states):
    STATES = json.dumps(states)

    md(r'''
## 🎛️ 12. Live demo — change the inputs and watch the answers change
Everything above shows *saved* results. This section is a **playground**: each cell has a small form. Change a value (dropdown, slider or box), press the ▶ button, and the answer is **recomputed from scratch** — proof that nothing is hard-coded.

| Cell | What you can change | What happens |
|---|---|---|
| 12.1 | nothing — just run it | A map of every column: which came with the file, which we created (including the class) |
| 12.2 | a **Row ID** | Shows that exact row of the dataset, its class, and what the model says about it |
| 12.3 | the **details of a new order** | The model predicts *high-value* / *standard* live |
| 12.4 | the **cut-off in dollars** | The class is redefined, the model is retrained, and the confusion matrix changes |
| 12.5 | **two groups** | Any two categories / segments / regions are compared on the spot |

*In Colab the form fields appear inside each form cell. To see the code behind a form, use the cell's ⋮ menu → "Show code".  Run sections 1–6 first (Runtime ▸ Run all does it).*
''')

    # ------------------------------------------------------------------ 12.1
    md(r'''
### 12.1 Where is everything in the dataset?
The table lists **every column** with its position, type, an example value, and **where it comes from**. The original file has 18 columns; the modelling table has more because *we created* some (the last one, `high_value`, is **the class**).
''')
    code(r'''
created = {
    "postal_imputed":     "created by us (3.3) — flags the 11 ZIP codes we filled in",
    "order_year":         "created by us (3.6) — from Order Date",
    "order_quarter":      "created by us (3.6) — from Order Date",
    "order_month":        "created by us (3.6) — from Order Date",
    "order_month_name":   "created by us (3.6) — from Order Date",
    "order_weekday":      "created by us (3.6) — from Order Date",
    "ship_days":          "created by us (3.6) — Ship Date minus Order Date",
    "log_sales":          "created by us (3.6) — log(1 + Sales)",
    "is_outlier":         "created by us (3.7) — Sales above the IQR fence",
    "state_pop":          "SCRAPED from Wikipedia (section 2), joined on State",
    "state_pop_millions": "SCRAPED from Wikipedia — population in millions",
    "high_value":         "★ THE CLASS — created by us in section 6: 1 if Sales ≥ cut-off, else 0",
}
col_map = pd.DataFrame({
    "position": range(1, mdl.shape[1] + 1),
    "column":   mdl.columns,
    "type":     [str(t) for t in mdl.dtypes],
    "example":  [str(mdl[c].iloc[0])[:36] for c in mdl.columns],
})
col_map["where it comes from"] = col_map["column"].map(lambda c: created.get(c, "in the raw file"))

print(f"The raw file has {raw.shape[1]} columns and {len(raw):,} rows.  Our modelling table has {mdl.shape[1]} columns and {len(mdl):,} rows.")
print(f"Raw file: {DATA_URL}")

# In Google Colab, also save the data as real files you can open from the Files pane (the folder icon on the left)
try:
    import google.colab                              # this import only works inside Colab
    raw.to_csv("train_raw.csv", index=False)
    clean.to_csv("train_clean.csv", index=False)
    print("\nSaved train_raw.csv (as downloaded) and train_clean.csv (after cleaning).")
    print("→ Open the folder icon on the left, then double-click either file to see the dataset itself.")
except ImportError:
    pass

col_map
''')

    # ------------------------------------------------------------------ 12.2
    md(r'''
### 12.2 Find any row — type a Row ID
The code (visible) looks a row up by its `Row ID`; the form cell below it lets you type one. Try `3406` (a row the model never saw in training), then `3407` (the duplicate we removed in pre-processing), `2698` (the biggest Technology sale), `1`, or any number from 1 to 9,800.
''')
    code(r'''
def show_row(row_id):
    r = raw[raw["Row ID"] == row_id]
    if r.empty:
        print(f"There is no Row ID {row_id}. Valid IDs run from {raw['Row ID'].min()} to {raw['Row ID'].max()}.")
        return
    print(f"ROW {row_id} — exactly as it is in the raw file.  (Open the CSV in Excel: it is line {int(r.index[0]) + 2}, because line 1 holds the column names.)\n")
    display(pd.DataFrame({"column #": range(1, r.shape[1] + 1), "column": r.columns, "value": r.iloc[0].astype(str).values}).set_index("column #"))

    m = mdl[mdl["Row ID"] == row_id]
    if m.empty:
        print("\nThis row does not appear in the cleaned data: it was removed in pre-processing as a duplicate (section 3.4).")
        return
    actual = int(m["high_value"].iloc[0])
    p = fitted["Random Forest"].predict_proba(m[FEATURES])[0, 1]
    guess = int(p >= 0.5)
    name = {0: "standard", 1: "high-value"}
    print(f"\nTHE CLASS of this row:  high_value = {actual}  ({name[actual]})   because Sales = ${m['Sales'].iloc[0]:,.2f} and the cut-off is ${threshold:,.2f}")
    print(f"WHAT THE MODEL SAYS (it is never shown Sales):  {name[guess]}  — probability of high-value {p:.0%}   →  {'✔ correct' if guess == actual else '✘ wrong'}")
    print(f"(This row was in the {'TEST' if m.index[0] in X_test.index else 'TRAINING'} set.)")
''')
    code(r'''
#@title 🔎 Find any row of the dataset — type a Row ID, then press ▶ { display-mode: "form" }
row_id = 3406  #@param {type:"integer"}

try:
    show_row(row_id)
except NameError:
    print("⚠️ This cell needs the cells above it to have run first.  Click  Runtime ▸ Run all  (wait until it finishes), then run this cell again.")
''')

    # ------------------------------------------------------------------ 12.3
    md(r'''
### 12.3 Predict a brand-new order
The model has only ever seen the training data, so any order you invent here is *new* to it. Pick the product, customer type, place, shipping mode and date; the model answers *high-value* or *standard* and shows how the answer would change for **every other product**.
''')
    code(r'''
sub_to_cat      = clean.drop_duplicates("Sub-Category").set_index("Sub-Category")["Category"].to_dict()
state_to_region = clean.drop_duplicates("State").set_index("State")["Region"].to_dict()
pop_millions    = (state_pop.set_index("state")["population_2020"] / 1e6).to_dict()
typical_days    = clean.groupby("Ship Mode")["ship_days"].median().to_dict()      # a realistic delivery time per ship mode

def order_frame(sub_categories, segment, state, ship_mode, order_month, order_year):
    """One row per sub-category, every other input the same — in exactly the format the model was trained on."""
    return pd.DataFrame({
        "Category": [sub_to_cat[s] for s in sub_categories], "Sub-Category": list(sub_categories),
        "Segment": segment, "Region": state_to_region[state], "Ship Mode": ship_mode,
        "order_month": order_month, "order_quarter": (order_month - 1) // 3 + 1, "order_year": order_year,
        "ship_days": typical_days[ship_mode], "state_pop_millions": pop_millions[state]})[FEATURES]

def predict_new_order(sub_category, segment, state, ship_mode, order_month, order_year):
    model = fitted["Random Forest"]
    p = model.predict_proba(order_frame([sub_category], segment, state, ship_mode, order_month, order_year))[0, 1]
    print("THE ORDER YOU ENTERED")
    print(f"  product      : {sub_category}  (category: {sub_to_cat[sub_category]})")
    print(f"  customer     : {segment}")
    print(f"  delivered to : {state}  (region: {state_to_region[state]}, population {pop_millions[state]:.1f} million)")
    print(f"  shipping     : {ship_mode}  (about {typical_days[ship_mode]:.0f} days)   ordered in month {order_month} of {order_year}")
    print(f"\nMODEL SAYS: {'HIGH-VALUE' if p >= 0.5 else 'STANDARD'}  — chance of high-value = {p:.0%}   (high-value = a sale of ${threshold:,.2f} or more)")

    subs = sorted(sub_to_cat)
    sweep = pd.Series(model.predict_proba(order_frame(subs, segment, state, ship_mode, order_month, order_year))[:, 1], index=subs).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(sweep.index, sweep.values * 100, color=[PALETTE[0] if s == sub_category else "#C9C9D6" for s in sweep.index])
    ax.axvline(50, color="#2B2B33", ls="--", lw=1)
    ax.set(title="Same order, different product — chance of 'high-value' (%)", xlabel="probability (%)", xlim=(0, 100))
    plt.tight_layout(); plt.show()
    print("The pink bar is the product you chose. Change any input in the form and run again.")
''')
    code(r'''
#@title 🧪 Predict a NEW order — change any input, then press ▶ { display-mode: "form" }
sub_category = "Chairs"  #@param ["Accessories", "Appliances", "Art", "Binders", "Bookcases", "Chairs", "Copiers", "Envelopes", "Fasteners", "Furnishings", "Labels", "Machines", "Paper", "Phones", "Storage", "Supplies", "Tables"]
segment = "Consumer"  #@param ["Consumer", "Corporate", "Home Office"]
state = "California"  #@param __STATES__
ship_mode = "Standard Class"  #@param ["Same Day", "First Class", "Second Class", "Standard Class"]
order_month = 11  #@param {type:"slider", min:1, max:12, step:1}
order_year = 2018  #@param {type:"slider", min:2015, max:2018, step:1}

try:
    predict_new_order(sub_category, segment, state, ship_mode, order_month, order_year)
except NameError:
    print("⚠️ This cell needs the cells above it to have run first.  Click  Runtime ▸ Run all  (wait until it finishes), then run this cell again.")
'''.replace("__STATES__", STATES))

    # ------------------------------------------------------------------ 12.4
    md(r'''
### 12.4 Change the class definition — what counts as "high-value"?
The class was *our* choice: a sale of $210.57 or more (the top 25%). Type a different dollar amount below: the class is rebuilt, the Random Forest is retrained on the same training rows, and a new confusion matrix appears. Note how **accuracy rises just because "high-value" becomes rarer** (always answering "standard" gets easier) — recall and F1 tell the real story.
''')
    code(r'''
from sklearn.ensemble import RandomForestClassifier

def retrain_for_cut_off(cut):
    y_new = (mdl["Sales"] >= cut).astype(int)
    y_tr, y_te = y_new.loc[X_train.index], y_new.loc[X_test.index]
    model = Pipeline([("prep", make_preprocessor()),
                      ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                                       n_jobs=2, random_state=RANDOM_STATE))])
    return y_new, y_tr, y_te, model

def try_cut_off(cut):
    y_new, y_tr, y_te, model = retrain_for_cut_off(cut)
    print(f"CLASS DEFINITION:  high-value = Sales ≥ ${cut:,.2f}")
    print(f"  → {int(y_new.sum()):,} high-value rows ({y_new.mean():.1%}) and {int((1 - y_new).sum()):,} standard rows ({1 - y_new.mean():.1%}) out of {len(y_new):,}")
    if min(y_tr.sum(), (1 - y_tr).sum()) < 20 or min(y_te.sum(), (1 - y_te).sum()) < 5:
        print("  One of the two classes is almost empty at this cut-off, so a model cannot learn it — try a number between about 25 and 1,000.")
        return
    model.fit(X_train, y_tr)
    pred, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"\nA lazy model that always says 'standard' would be {1 - y_te.mean():.1%} accurate.")
    print(f"Retrained Random Forest:  accuracy {accuracy_score(y_te, pred):.1%} · precision {precision_score(y_te, pred, zero_division=0):.2f} · "
          f"recall {recall_score(y_te, pred):.2f} · F1 {f1_score(y_te, pred):.2f} · ROC-AUC {roc_auc_score(y_te, proba):.2f}")
    display(pd.DataFrame([[tn, fp], [fn, tp]], index=["really standard", "really high-value"], columns=["model said standard", "model said high-value"]))
''')
    code(r'''
#@title 🎚️ Change the CLASS definition — type a dollar amount, then press ▶ { display-mode: "form" }
cut_off_in_dollars = 210.57  #@param {type:"number"}

try:
    try_cut_off(cut_off_in_dollars)
except NameError:
    print("⚠️ This cell needs the cells above it to have run first.  Click  Runtime ▸ Run all  (wait until it finishes), then run this cell again.")
''')
    md(r'''
Several cut-offs side by side (takes a few seconds) — watch "lazy accuracy" climb while recall and F1 fall:
''')
    code(r'''
rows = []
for cut in [50, 100, 210.57, 500, 1000]:
    y_new, y_tr, y_te, model = retrain_for_cut_off(cut)
    pred = model.fit(X_train, y_tr).predict(X_test)
    rows.append({"cut-off ($)": cut, "share that is high-value": y_new.mean(), "lazy 'always standard' accuracy": 1 - y_te.mean(),
                 "model accuracy": accuracy_score(y_te, pred), "recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred)})
pd.DataFrame(rows).set_index("cut-off ($)").style.format({"share that is high-value": "{:.1%}", "lazy 'always standard' accuracy": "{:.1%}",
                                                          "model accuracy": "{:.1%}", "recall": "{:.2f}", "F1": "{:.2f}"})
''')

    # ------------------------------------------------------------------ 12.5
    md(r'''
### 12.5 Compare any two groups
Pick two values from the **same** list — categories, customer segments or regions — and the comparison from section 5.7 is recomputed for them.
''')
    code(r'''
from scipy import stats

GROUPS = {"Category": ["Furniture", "Office Supplies", "Technology"],
          "Segment":  ["Consumer", "Corporate", "Home Office"],
          "Region":   ["Central", "East", "South", "West"]}

def compare_groups(a, b):
    column = next((c for c, values in GROUPS.items() if a in values and b in values), None)
    if column is None or a == b:
        print("Pick two DIFFERENT values from the same list:\n  " + "\n  ".join(f"{c}: {', '.join(v)}" for c, v in GROUPS.items()))
        return
    d = clean[clean[column].isin([a, b])].copy()
    d["high_value"] = d["Sales"] >= threshold

    def measures(x):
        yearly = x.groupby("order_year")["Sales"].sum()
        return pd.Series({"Total revenue ($)": x["Sales"].sum(), "Order lines sold": len(x),
                          "Average sale per line ($)": x["Sales"].mean(), "Typical (median) sale ($)": x["Sales"].median(),
                          "High-value lines (%)": 100 * x["high_value"].mean(),
                          "Revenue per order containing it ($)": x["Sales"].sum() / x["Order ID"].nunique(),
                          "Growth 2015 → 2018 (%)": 100 * (yearly[2018] / yearly[2015] - 1)})
    table = pd.DataFrame({v: measures(d[d[column] == v]) for v in (a, b)})
    fmt = {"Total revenue ($)": "${:,.0f}", "Order lines sold": "{:,.0f}", "Average sale per line ($)": "${:,.0f}", "Typical (median) sale ($)": "${:,.0f}",
           "High-value lines (%)": "{:.1f}%", "Revenue per order containing it ($)": "${:,.0f}", "Growth 2015 → 2018 (%)": "{:+.0f}%"}
    pretty = table.astype(object).copy()
    for row, f in fmt.items():
        pretty.loc[row] = [f.format(v) for v in table.loc[row]]
    print(f"COMPARING  {a}  vs  {b}   (column: {column})")
    display(pretty)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7))
    for ax, (title, row, label) in zip(axes, [("Total revenue", "Total revenue ($)", lambda v: f"${v/1000:,.0f}k"),
                                              ("Number of order lines", "Order lines sold", lambda v: f"{v:,.0f}"),
                                              ("Average sale per line", "Average sale per line ($)", lambda v: f"${v:,.0f}"),
                                              ("% high-value lines", "High-value lines (%)", lambda v: f"{v:.0f}%")]):
        vals = table.loc[row, [a, b]]
        bars = ax.bar([a, b], vals.values, color=[PALETTE[1], PALETTE[0]], width=0.6)
        ax.bar_label(bars, labels=[label(v) for v in vals.values], padding=3, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11.5); ax.set_ylim(0, vals.max() * 1.22)
        ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
    plt.tight_layout(); plt.show()

    xs, ys = d.loc[d[column] == a, "Sales"], d.loc[d[column] == b, "Sales"]
    U, p = stats.mannwhitneyu(xs, ys)
    print(f"Is the gap in sale size real? A random {a} sale is bigger than a random {b} sale in {U / (len(xs) * len(ys)):.0%} of pairs "
          f"(50% = no difference).  Mann-Whitney p-value: {'below 0.001' if p < 0.001 else f'{p:.3f}'}.")
''')
    code(r'''
#@title ⚖️ Compare ANY two groups — pick two values from the same list, then press ▶ { display-mode: "form" }
group_A = "Technology"  #@param ["Furniture", "Office Supplies", "Technology", "Consumer", "Corporate", "Home Office", "Central", "East", "South", "West"]
group_B = "Office Supplies"  #@param ["Furniture", "Office Supplies", "Technology", "Consumer", "Corporate", "Home Office", "Central", "East", "South", "West"]

try:
    compare_groups(group_A, group_B)
except NameError:
    print("⚠️ This cell needs the cells above it to have run first.  Click  Runtime ▸ Run all  (wait until it finishes), then run this cell again.")
''')
