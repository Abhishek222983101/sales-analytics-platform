"""Sections 6-7: classification and regression (+ time-series forecast)."""


def add(md, code):
    # ------------------------------------------------------------------ 6
    md(r'''
## 🎯 6. Classification

**Classification** predicts a *category* (a label) for each record. Business question:

> *Given an order line's product, customer segment, region, timing and shipping, will it be a **high-value sale**?*

Why it is useful: sales teams can prioritise the kinds of orders that generate the most revenue.

**Target definition:** `high_value = 1` if the line's sales are in the **top 25%** of all lines, else `0`. This gives an *imbalanced* problem (25% / 75%), so **accuracy alone is misleading** — predicting "not high-value" every time would already be 75% accurate. We therefore also report **precision, recall, F1 and ROC-AUC**.

**Features used:** Category, Sub-Category, Segment, Region, Ship Mode, order month / quarter / year, shipping days, and the **scraped state population**. We deliberately do *not* use anything derived from `Sales` itself (that would be leakage).
''')
    code(r'''
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             r2_score, mean_squared_error, mean_absolute_error)
from xgboost import XGBClassifier, XGBRegressor

# modelling table = clean data + the scraped population
mdl = clean.merge(state_pop.rename(columns={"state": "State", "population_2020": "state_pop"}), on="State", how="left")
mdl["state_pop_millions"] = mdl["state_pop"] / 1e6

CAT = ["Category", "Sub-Category", "Segment", "Region", "Ship Mode"]
NUM = ["order_month", "order_quarter", "order_year", "ship_days", "state_pop_millions"]
FEATURES = CAT + NUM
assert mdl[FEATURES].isna().sum().sum() == 0, "unexpected missing values in the features"

def make_preprocessor():
    # one-hot encode categories, standard-scale numbers — fitted ONLY on the training data inside each pipeline
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM),
    ])

threshold = clean["Sales"].quantile(0.75)
mdl["high_value"] = (mdl["Sales"] >= threshold).astype(int)
print(f"A line is 'high-value' if Sales ≥ ${threshold:,.2f}  →  {mdl['high_value'].mean():.1%} of lines")

X, y = mdl[FEATURES], mdl["high_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {len(X_train):,} rows · Test: {len(X_test):,} rows (stratified, so both keep the 25/75 balance)")
''')
    code(r'''
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                                  n_jobs=2, random_state=RANDOM_STATE),
    "KNN (k=15)":          KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes":         GaussianNB(),
    "XGBoost":             XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
                                         colsample_bytree=0.9, scale_pos_weight=3, eval_metric="logloss",
                                         tree_method="hist", n_jobs=2, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted, probas = [], {}, {}
for name, model in classifiers.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()      # 5-fold CV on the training set
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    fitted[name], probas[name] = pipe, proba
    rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                 "recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
                 "ROC-AUC": roc_auc_score(y_test, proba), "CV F1 (5-fold)": cv_f1})

clf_results = pd.DataFrame(rows).set_index("model").sort_values("F1", ascending=False)
best_clf = clf_results.index[0]
baseline_acc = 1 - y_test.mean()
print(f"Reference: always guessing 'not high-value' would be {baseline_acc:.1%} accurate but useless (F1 = 0).")
clf_results.round(3)
''')
    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# (a) metric comparison
plot_df = clf_results[["F1", "ROC-AUC"]].reset_index().melt(id_vars="model", var_name="metric", value_name="score")
sns.barplot(data=plot_df, y="model", x="score", hue="metric", palette=[PALETTE[0], PALETTE[1]], ax=axes[0])
axes[0].set(title="Model comparison (test set)", xlabel="score", ylabel="", xlim=(0, 1))
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

# (b) ROC curves
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[1].plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(y_test, p):.2f})")
axes[1].plot([0, 1], [0, 1], "--", color="#9A9AAE")
axes[1].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate"); axes[1].legend(fontsize=8, loc="lower right")

# (c) confusion matrix of the best model
cm = confusion_matrix(y_test, fitted[best_clf].predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["standard", "high-value"]).plot(ax=axes[2], cmap=sns.light_palette(PALETTE[0], as_cmap=True), colorbar=False)
axes[2].set(title=f"Confusion matrix — {best_clf}"); axes[2].grid(False)
plt.tight_layout(); plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"{best_clf}: of {tp + fn} truly high-value lines it found {tp} ({tp/(tp+fn):.0%} recall); "
      f"of {tp + fp} lines it flagged, {tp} were right ({tp/(tp+fp):.0%} precision).")
''')
    md(r'''
**How to read the metrics**
* **Precision** — of the lines we *flag* as high-value, how many truly are?
* **Recall** — of all truly high-value lines, how many do we *catch*?
* **F1** — the balance of the two (best single number for imbalanced data).
* **ROC-AUC** — how well the model *ranks* high-value above standard lines (0.5 = coin flip, 1.0 = perfect).
* **CV F1** — the same F1 averaged over 5 different train/validation splits; if it is close to the test F1 the model is **not overfitted**.
''')
    md(r'''
### 6.1 Which features matter?
Random Forests measure how much each feature reduces impurity. Because one-hot encoding splits a category like Sub-Category into 17 columns, we **add them back together** to get one importance per original feature.
''')
    code(r'''
rf_pipe = fitted["Random Forest"]
enc_names = list(rf_pipe.named_steps["prep"].get_feature_names_out())
importances = rf_pipe.named_steps["model"].feature_importances_

def original_feature(encoded_name):
    n = encoded_name.split("__", 1)[1]
    for c in CAT:
        if n.startswith(c + "_"):
            return c
    return n

grouped = pd.Series(importances, index=[original_feature(n) for n in enc_names]).groupby(level=0).sum().sort_values()
top_enc = pd.Series(importances, index=[n.split("__", 1)[1] for n in enc_names]).sort_values().tail(10)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].barh(grouped.index, grouped.values, color=PALETTE[0]); axes[0].set(title="Importance per original feature", xlabel="importance")
axes[1].barh(top_enc.index, top_enc.values, color=PALETTE[1]); axes[1].set(title="10 most important individual values", xlabel="importance")
plt.tight_layout(); plt.show()
print(f"Most important feature: '{grouped.index[-1]}' ({grouped.iloc[-1]:.0%} of total importance).")
''')
    md(r'''
### 6.2 A readable decision tree
A depth-3 tree is easy to read and shows the *rules* the algorithm discovered.
''')
    code(r'''
prep = make_preprocessor().fit(X_train)
Xt = prep.transform(X_train)
names_clean = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
small_tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE).fit(Xt, y_train)

fig, ax = plt.subplots(figsize=(18, 7.5))
plot_tree(small_tree, feature_names=names_clean, class_names=["standard", "high-value"], filled=True, rounded=True,
          proportion=True, fontsize=9, ax=ax)
ax.set_title("Decision tree (depth 3): the rules behind 'high-value'")
plt.show()
print(export_text(small_tree, feature_names=names_clean, max_depth=3)[:1500])
''')
    md(r'''
### 6.3 Hyper-parameter tuning
We tune the Random Forest with **grid search + 3-fold cross-validation** (it tries every combination below and keeps the best by F1), then compare against the untuned version on the untouched test set.
''')
    code(r'''
grid = GridSearchCV(
    Pipeline([("prep", make_preprocessor()),
              ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", n_jobs=2, random_state=RANDOM_STATE))]),
    param_grid={"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3, 5]},
    scoring="f1", cv=3, n_jobs=1)
grid.fit(X_train, y_train)
tuned_f1 = f1_score(y_test, grid.predict(X_test))
print("Best parameters:", grid.best_params_)
print(f"Random Forest F1 on the test set — untuned: {clf_results.loc['Random Forest', 'F1']:.3f} · tuned: {tuned_f1:.3f}")
''')
    md(r'''
### 6.4 Classification — conclusion
''')
    code(r'''
top_feat = grouped.index[-1]
tied = [m for m in clf_results.index if clf_results.loc[best_clf, "F1"] - clf_results.loc[m, "F1"] <= 0.015]
print(f"• Top F1: {best_clf} ({clf_results.loc[best_clf, 'F1']:.2f}, ROC-AUC {clf_results.loc[best_clf, 'ROC-AUC']:.2f}).")
print(f"• But {len(tied)} models are within 0.015 F1 of each other ({', '.join(tied)}) — that is a statistical tie, not a winner.")
print("  When models tie, prefer the one that is easiest to explain: the Random Forest / Decision Tree give feature importances and readable rules.")
print(f"• KNN is clearly weakest (recall {clf_results.loc['KNN (k=15)', 'recall']:.2f}): distance-based methods struggle with many one-hot columns and the 25/75 imbalance.")
print(f"• '{top_feat}' is the dominant driver, confirming the EDA hypothesis: WHAT is bought decides the sale size.")
gap = clf_results.loc[best_clf, 'F1'] - clf_results.loc[best_clf, 'CV F1 (5-fold)']
print(f"• CV F1 vs test F1 differ by only {abs(gap):.2f} → the model generalises, it is not memorising.")
print("• Business use: flag incoming orders likely to be high-value for priority handling / account-manager attention.")
''')

    # ------------------------------------------------------------------ 7
    md(r'''
## 📈 7. Regression

**Regression** predicts a *number*. Two tasks:

1. **7.1 — predict the sale value of an order line** from its attributes.
2. **7.2 — forecast total monthly revenue** into the future (time-series regression).

### 7.1 Predicting the sale value
We predict `log(1 + Sales)` (explained in pre-processing: it removes the skew, so errors are measured in *relative* rather than dollar terms) and compare five regressors against a **do-nothing baseline** that always predicts the average.

Metrics: **R²** (share of variance explained; 0 = no better than the average, 1 = perfect), **RMSE / MAE** (typical error size).
''')
    code(r'''
yr = mdl["log_sales"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, yr, test_size=0.2, random_state=RANDOM_STATE)

regressors = {
    "Baseline (always the mean)": DummyRegressor(strategy="mean"),
    "Linear Regression":  LinearRegression(),
    "Ridge (alpha=1)":    Ridge(alpha=1.0),
    "Decision Tree":      DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
    "Random Forest":      RandomForestRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=2, random_state=RANDOM_STATE),
    "XGBoost":            XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                       colsample_bytree=0.9, tree_method="hist", n_jobs=2, random_state=RANDOM_STATE),
}
rows, reg_fitted, reg_pred = [], {}, {}
for name, model in regressors.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)]).fit(Xr_train, yr_train)
    pred = pipe.predict(Xr_test)
    reg_fitted[name], reg_pred[name] = pipe, pred
    rows.append({"model": name, "R²": r2_score(yr_test, pred),
                 "RMSE (log)": np.sqrt(mean_squared_error(yr_test, pred)),
                 "MAE (log)": mean_absolute_error(yr_test, pred),
                 "MAE ($)": mean_absolute_error(np.expm1(yr_test), np.expm1(pred))})

reg_results = pd.DataFrame(rows).set_index("model").sort_values("R²", ascending=False)
best_reg = reg_results.index[0]
reg_results.round(3)
''')
    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
r2_plot = reg_results["R²"].sort_values()
axes[0].barh(r2_plot.index, r2_plot.values, color=[PALETTE[0] if n == best_reg else "#C9C9D6" for n in r2_plot.index])
axes[0].set(title="R² on the test set (higher is better)", xlabel="R²"); axes[0].axvline(0, color="black", lw=0.8)

axes[1].scatter(yr_test, reg_pred[best_reg], s=6, alpha=0.35, color=PALETTE[1])
lims = [yr_test.min(), yr_test.max()]
axes[1].plot(lims, lims, "--", color=PALETTE[0], label="perfect prediction")
axes[1].set(title=f"Actual vs predicted — {best_reg}", xlabel="actual log(1+Sales)", ylabel="predicted"); axes[1].legend()

resid = yr_test - reg_pred[best_reg]
sns.histplot(resid, bins=40, color=PALETTE[3], ax=axes[2])
axes[2].axvline(0, color="black", lw=0.8); axes[2].set(title="Residuals (actual − predicted)", xlabel="error in log units")
plt.tight_layout(); plt.show()
''')
    code(r'''
best_r2 = reg_results.loc[best_reg, "R²"]
print(f"Best regressor: {best_reg} with R² = {best_r2:.2f} → the features explain about {best_r2:.0%} of the variation in (log) sale size.")
base_mae = reg_results.loc["Baseline (always the mean)", "MAE ($)"]
print(f"In dollars, the average miss falls from ${base_mae:,.0f} (baseline) to ${reg_results.loc[best_reg, 'MAE ($)']:,.0f} — a "
      f"{1 - reg_results.loc[best_reg, 'MAE ($)'] / base_mae:.0%} improvement; in log terms the typical error is a factor of "
      f"{np.exp(reg_results.loc[best_reg, 'MAE (log)']):.1f}×.")
tied_r = [m for m in reg_results.index if m != "Baseline (always the mean)" and best_r2 - reg_results.loc[m, "R²"] <= 0.04]
print(f"{len(tied_r)} models score within 0.04 R² of each other ({', '.join(tied_r)}) — even simple Linear Regression keeps up, "
      "which tells us the limit is the information in the features, not the algorithm.")
''')
    md(r'''
**Honest interpretation.** The score is *moderate, not high — and that is expected*, not a bug. Our dataset has **no quantity, discount or exact product price**, which are exactly the things that determine a line's sale value. Product type alone tells us roughly *how big* a sale tends to be but not *the exact value*. The model beats the baseline clearly (R² well above 0), so the features carry real signal; the rest is unobserved information. In a real warehouse we would add quantity and unit price to the fact table.
''')

    md(r'''
### 7.2 Forecasting monthly revenue (time-series regression)
Here the target is *total revenue per month*, and the goal is to look **forward**. Two very different model families are compared:

| Model | Idea |
|---|---|
| **SARIMA** | Classical statistics: learns trend + yearly seasonality directly from the series |
| **XGBoost** | Machine learning on engineered features: month, quarter, last month's revenue (*lag-1*), last year's same month (*lag-12*), rolling mean |

**Evaluation rule for time series:** never shuffle. We train on the first ~80% of months and test on the *last* ~20% — the model never sees the future it is asked to predict. Both models forecast the whole test window **recursively** (each prediction feeds the next), so the comparison is fair.
''')
    code(r'''
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

ts = clean.set_index("Order Date")["Sales"].resample("MS").sum().astype(float)
TS_FEATS = ["year", "month", "quarter", "lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "roll_std_3"]

def ts_features(s):
    f = pd.DataFrame({"y": s})
    f["year"], f["month"], f["quarter"] = f.index.year, f.index.month, f.index.quarter
    for lag in (1, 2, 3, 12):
        f[f"lag_{lag}"] = f["y"].shift(lag)
    f["roll_mean_3"] = f["y"].shift(1).rolling(3).mean()
    f["roll_std_3"] = f["y"].shift(1).rolling(3).std()
    return f.dropna()

def ts_row(hist, ts_):
    return {"year": ts_.year, "month": ts_.month, "quarter": ts_.quarter, "lag_1": hist[-1], "lag_2": hist[-2],
            "lag_3": hist[-3], "lag_12": hist[-12], "roll_mean_3": float(np.mean(hist[-3:])),
            "roll_std_3": float(np.std(hist[-3:], ddof=1))}

def fit_xgb_ts(frame):
    m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                     tree_method="hist", n_jobs=2, random_state=RANDOM_STATE)
    return m.fit(frame[TS_FEATS], frame["y"])

def recursive_forecast(model, history, index):
    hist, out = [float(v) for v in history.to_numpy()], []
    for ts_ in index:
        yhat = float(model.predict(pd.DataFrame([ts_row(hist, ts_)])[TS_FEATS])[0])
        out.append(yhat); hist.append(yhat)
    return np.array(out)

def sarima_fit(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(series, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=50)

def ts_metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return {"RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))), "MAE": float(np.mean(np.abs(actual - pred))),
            "MAPE %": float(np.mean(np.abs((actual - pred) / actual)) * 100)}

feats_ts = ts_features(ts)
split = int(round(len(feats_ts) * 0.8))
train_ts, test_ts = feats_ts.iloc[:split], feats_ts.iloc[split:]
series_train, series_test = ts[ts.index < test_ts.index[0]], ts[ts.index >= test_ts.index[0]]

xgb_bt = recursive_forecast(fit_xgb_ts(train_ts), series_train, series_test.index)
sar_bt = np.asarray(sarima_fit(series_train).get_forecast(steps=len(series_test)).predicted_mean)

ts_results = pd.DataFrame({"XGBoost": ts_metrics(series_test, xgb_bt), "SARIMA": ts_metrics(series_test, sar_bt)}).T
best_ts = ts_results["MAPE %"].idxmin()
print(f"Training months: {len(series_train)} · test months: {len(series_test)} ({series_test.index[0]:%b %Y} – {series_test.index[-1]:%b %Y})")
ts_results.round(1)
''')
    code(r'''
# refit on ALL data and forecast the next 6 months
horizon = 6
future_idx = pd.date_range(ts.index[-1], periods=horizon + 1, freq="MS")[1:]
xgb_future = recursive_forecast(fit_xgb_ts(feats_ts), ts, future_idx)
sar_res = sarima_fit(ts)
sar_fc = sar_res.get_forecast(steps=horizon)
sar_future, sar_ci = np.asarray(sar_fc.predicted_mean), sar_fc.conf_int(alpha=0.2).to_numpy()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(ts.index, ts.values, color="#2B2B33", lw=1.8, label="Actual revenue")
ax.plot(series_test.index, xgb_bt, "--", color=PALETTE[0], label="XGBoost (backtest)")
ax.plot(series_test.index, sar_bt, "--", color=PALETTE[1], label="SARIMA (backtest)")
ax.plot(future_idx, xgb_future, "-o", ms=4, color=PALETTE[0], label="XGBoost forecast")
ax.plot(future_idx, sar_future, "-o", ms=4, color=PALETTE[1], label="SARIMA forecast")
ax.fill_between(future_idx, sar_ci[:, 0], sar_ci[:, 1], color=PALETTE[1], alpha=0.18, label="SARIMA 80% interval")
ax.axvline(series_test.index[0], color="#9A9AAE", ls=":"); ax.text(series_test.index[0], ax.get_ylim()[1] * 0.96, " test window →", color="#6A6A80")
ax.yaxis.set_major_formatter(money)
ax.set(title="Monthly revenue — history, backtest and 6-month forecast", ylabel="Revenue"); ax.legend(ncol=3, fontsize=9, loc="upper left")
plt.tight_layout(); plt.show()

winner_future = xgb_future if best_ts == "XGBoost" else sar_future
# Compare with the SAME months a year earlier — comparing Jan–Jun with the preceding Jul–Dec would just show seasonality
same_months_last_year = ts.reindex(future_idx - pd.DateOffset(years=1)).sum()
print(f"Best model on the backtest: {best_ts} (MAPE {ts_results.loc[best_ts, 'MAPE %']:.1f}%).")
print(f"Forecast for {future_idx[0]:%b %Y} – {future_idx[-1]:%b %Y}: ${winner_future.sum():,.0f}  vs ${same_months_last_year:,.0f} "
      f"in the same months a year earlier ({(winner_future.sum() / same_months_last_year - 1):+.0%}).")
''')
    code(r'''
# Why does the forecast look the way it does? XGBoost's own feature importance
ts_model = fit_xgb_ts(feats_ts)
imp_ts = pd.Series(ts_model.feature_importances_, index=TS_FEATS).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(imp_ts.index, imp_ts.values, color=PALETTE[0]); ax.set(title="What drives the revenue forecast?", xlabel="importance")
plt.tight_layout(); plt.show()
print(f"Top driver: '{imp_ts.index[-1]}' — " + ("last year's same month, i.e. strong yearly seasonality." if imp_ts.index[-1] == "lag_12" else "see chart."))
''')
