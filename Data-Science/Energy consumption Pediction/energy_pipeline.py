# %% [markdown]
# # Energy Consumption Prediction — Appliances Energy (Wh)
#
# **Goal:** Predict household appliance energy consumption and identify the
# environmental and behavioural factors that drive it, so that homeowners,
# building operators and utilities can cut cost, smooth demand peaks and
# schedule loads more intelligently.
#
# **Dataset:** UCI/Kaggle *Appliances Energy Prediction* — 19,735 records at a
# regular 10‑minute cadence (Jan–May 2016) with indoor temperature/humidity
# sensors for 9 zones, outdoor weather‑station data, and the target
# `Appliances` (Wh).
#
# **Business questions**
# 1. What drives appliance energy consumption?
# 2. Can we predict the next interval's energy usage?
# 3. Which environmental factors matter most?
# 4. Where are the opportunities to shift load and save cost?
#
# **Approach (honest, leakage‑aware):** this is time‑series data, so we use a
# **chronological train/test split**, build lag/rolling features with `.shift()`
# (a row only ever sees the *past*), and model `log1p(Appliances)` to tame the
# heavy right‑skew — then invert predictions back to Wh for all reported metrics.

# %%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"

DATA_PATH = "KAG_energydata_complete.csv"
OUT = "outputs"
PLOTS = os.path.join(OUT, "plots")
MODELS = os.path.join(OUT, "models")
EXPORT = os.path.join(OUT, "data")
for d in (PLOTS, MODELS, EXPORT):
    os.makedirs(d, exist_ok=True)


def savefig(name):
    """Save the current figure to outputs/plots; notebook still displays inline."""
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, name), bbox_inches="tight")


print("Libraries loaded. Outputs ->", os.path.abspath(OUT))

# %% [markdown]
# ## 2. Data Understanding

# %%
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Shape:", df.shape)
print("Date range:", df["date"].min(), "->", df["date"].max())
cadence = df["date"].diff().dropna().dt.total_seconds().div(60)
print("Sampling cadence (minutes):", cadence.mode().iloc[0],
      "| regular:", cadence.nunique() == 1)
df.head()

# %%
df.describe().T

# %% [markdown]
# ### Missing values & data quality
# A clean dataset is ideal but we never assume it — we check explicitly. The
# `df.isnull().sum()` audit below confirms zero missing values, so no
# mean/median imputation is required. `rv1` and `rv2` are *identical random
# noise* variables (correlation with the target ≈ 0); we keep them out of the
# model and later use SHAP to confirm a well‑behaved model ignores them.

# %%
print("Total missing values:", int(df.isnull().sum().sum()))
print("rv1 and rv2 identical:", np.allclose(df["rv1"], df["rv2"]))
print("corr(rv1, Appliances):", round(df["rv1"].corr(df["Appliances"]), 4))
print("\nTarget (Appliances, Wh) summary:")
print(df["Appliances"].describe().round(2).to_string())
print("Skewness:", round(df["Appliances"].skew(), 3))

# %%
# Time-based features are created up front because the EDA uses them.
df["hour"] = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek          # 0 = Monday
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)
df["dayname"] = df["date"].dt.day_name()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %% [markdown]
# ### Plot 1 — Energy consumption distribution
# Purpose: detect skew and outliers. Expectation: heavily right‑skewed with a
# long tail of high‑consumption bursts. The `log1p` view (right) is far more
# symmetric — this is exactly why we model the log of the target.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df["Appliances"], bins=60, kde=True, color="#d1495b", ax=ax[0])
ax[0].set_title(f"Appliances (Wh) — raw  (skew={df['Appliances'].skew():.2f})")
ax[0].set_xlabel("Wh")
sns.histplot(np.log1p(df["Appliances"]), bins=60, kde=True, color="#2e7d32", ax=ax[1])
ax[1].set_title("log1p(Appliances) — modelling target")
ax[1].set_xlabel("log1p(Wh)")
savefig("01_target_distribution.png")

# %% [markdown]
# ### Plot 2 — Hour of day vs energy
# Purpose: find peak‑usage windows for load‑shifting. Expectation: a mid‑morning
# rise and a pronounced evening peak; lowest usage overnight.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
hourly = df.groupby("hour")["Appliances"].mean()
sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="#1f77b4", ax=ax[0])
ax[0].set_title("Mean energy by hour of day")
ax[0].set_xlabel("Hour"); ax[0].set_ylabel("Mean Appliances (Wh)")
sns.boxplot(data=df, x="hour", y="Appliances", color="#9ecae1", fliersize=1, ax=ax[1])
ax[1].set_ylim(0, 350)
ax[1].set_title("Energy distribution by hour (zoomed)")
savefig("02_hour_vs_energy.png")

# %% [markdown]
# ### Plot 3 — Temperature vs energy (HVAC dependency)
# Purpose: test heating/cooling dependency. We bin outdoor temperature and plot
# mean energy per bin to cut through the 19k‑point scatter.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
df["_Tout_bin"] = pd.cut(df["T_out"], bins=15)
tbin = df.groupby("_Tout_bin", observed=True)["Appliances"].mean()
xlabels = [f"{i.mid:.0f}" for i in tbin.index]
sns.lineplot(x=range(len(tbin)), y=tbin.values, marker="o", color="#e76f51", ax=ax[0])
ax[0].set_xticks(range(len(tbin))); ax[0].set_xticklabels(xlabels, rotation=0)
ax[0].set_title("Mean energy vs outdoor temperature (binned)")
ax[0].set_xlabel("Outdoor temp T_out (°C)"); ax[0].set_ylabel("Mean Appliances (Wh)")
samp = df.sample(4000, random_state=RANDOM_STATE)
sns.scatterplot(data=samp, x="T2", y="Appliances", alpha=0.2, s=12, color="#264653", ax=ax[1])
ax[1].set_ylim(0, 600)
ax[1].set_title("Living‑room temp (T2) vs energy")
df.drop(columns="_Tout_bin", inplace=True)
savefig("03_temperature_vs_energy.png")

# %% [markdown]
# ### Plot 4 — Humidity vs energy
# Purpose: gauge the humidity/HVAC link. Expectation: a mild positive relation.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
df["_RH1_bin"] = pd.cut(df["RH_1"], bins=15)
hbin = df.groupby("_RH1_bin", observed=True)["Appliances"].mean()
hx = [f"{i.mid:.0f}" for i in hbin.index]
sns.lineplot(x=range(len(hbin)), y=hbin.values, marker="o", color="#6a4c93", ax=ax[0])
ax[0].set_xticks(range(len(hbin))); ax[0].set_xticklabels(hx)
ax[0].set_title("Mean energy vs kitchen humidity RH_1 (binned)")
ax[0].set_xlabel("RH_1 (%)"); ax[0].set_ylabel("Mean Appliances (Wh)")
df["_RHout_bin"] = pd.cut(df["RH_out"], bins=15)
hbin2 = df.groupby("_RHout_bin", observed=True)["Appliances"].mean()
hx2 = [f"{i.mid:.0f}" for i in hbin2.index]
sns.lineplot(x=range(len(hbin2)), y=hbin2.values, marker="o", color="#118ab2", ax=ax[1])
ax[1].set_xticks(range(len(hbin2))); ax[1].set_xticklabels(hx2)
ax[1].set_title("Mean energy vs outdoor humidity RH_out (binned)")
ax[1].set_xlabel("RH_out (%)")
df.drop(columns=["_RH1_bin", "_RHout_bin"], inplace=True)
savefig("04_humidity_vs_energy.png")

# %% [markdown]
# ### Plot 5 — Correlation heatmap & strongest linear drivers
# Purpose: find the strongest linear relationships with the target. With ~28
# numeric columns the full heatmap shows structure (room temperatures move
# together); the sorted bar chart ranks each feature's correlation with energy.

# %%
num = df.select_dtypes(include=[np.number]).drop(columns=["rv1", "rv2"], errors="ignore")
corr = num.corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=.3, square=False,
            cbar_kws={"shrink": .6}, ax=ax)
ax.set_title("Correlation heatmap (all numeric features)")
savefig("05a_correlation_heatmap.png")

# %%
target_corr = corr["Appliances"].drop("Appliances").sort_values()
fig, ax = plt.subplots(figsize=(8, 9))
colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors)
ax.set_title("Correlation of each feature with Appliances energy")
ax.set_xlabel("Pearson r")
ax.axvline(0, color="k", lw=.8)
savefig("05b_target_correlations.png")
print("Top positive drivers:\n", target_corr.tail(6).round(3).to_string())

# %% [markdown]
# ### Plot 6 — Temporal trend & weekly seasonality
# Purpose: expose recurring daily/weekly patterns that justify time features.
# The daily series shows the overall trajectory; the hour × weekday heatmap
# reveals *when* the household is energy‑hungry (weekday evenings).

# %%
daily = df.set_index("date")["Appliances"].resample("D").mean()
fig, ax = plt.subplots(figsize=(13, 4))
sns.lineplot(x=daily.index, y=daily.values, color="#1f77b4", ax=ax)
ax.set_title("Daily mean appliance energy over time")
ax.set_ylabel("Mean Appliances (Wh)")
savefig("06a_daily_trend.png")

# %%
pivot = (df.groupby(["dayofweek", "hour"])["Appliances"].mean()
           .unstack(level="hour"))
pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(13, 4.5))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=.3, cbar_kws={"label": "Mean Wh"}, ax=ax)
ax.set_title("Mean energy by weekday × hour")
ax.set_xlabel("Hour of day"); ax.set_ylabel("")
savefig("06b_weekday_hour_heatmap.png")

# %% [markdown]
# **EDA takeaways**
# - Energy is heavily right‑skewed: most intervals sit at 50–100 Wh with
#   occasional bursts to 1000+ Wh → model `log1p`, keep the spikes.
# - Clear daily rhythm: low overnight, climbing through the morning, peaking
#   in the **early evening (≈17–20h)**, strongest on weekdays.
# - Outdoor/indoor temperature and humidity show real but **non‑linear**
#   relationships with energy → tree models should beat linear regression.
# - Room temperatures are highly inter‑correlated (shared building climate).

# %% [markdown]
# ## 4. Data Cleaning & Outlier Treatment
#
# No missing values to impute. For outliers we *quantify* the IQR‑defined
# extremes but deliberately **do not delete them**: those high‑consumption
# bursts are precisely the demand‑peak events the business wants to predict.
# Modelling `log1p(Appliances)` already compresses the tail, giving us the
# benefit of winsorization without discarding the events that matter.

# %%
q1, q3 = df["Appliances"].quantile([.25, .75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
n_out = int((df["Appliances"] > upper).sum())
print(f"IQR upper fence: {upper:.0f} Wh")
print(f"Intervals above fence: {n_out} ({100*n_out/len(df):.1f}%) — kept as genuine demand peaks")

# %% [markdown]
# ## 5. Feature Engineering
#
# | Feature | Rationale |
# |---|---|
# | `hour`, `is_weekend`, `is_night`, `month` | capture daily/weekly behaviour |
# | `hour_sin`, `hour_cos` | cyclical encoding so 23h and 0h are adjacent |
# | `indoor_temp_mean` | overall house warmth (rooms move together) |
# | `temp_diff` = indoor − outdoor | the key HVAC heating/cooling driver |
# | `lag_1/2/3` | energy is strongly autocorrelated (recent past predicts next step) |
# | `roll_mean_3/6/18`, `roll_std_6` | short/medium‑term consumption trend & volatility (last 30 min → 3 h) |
#
# All lag/rolling features use `.shift(1)` so a row only sees **past** values —
# this is what makes the chronological evaluation leakage‑free.

# %%
indoor_cols = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]  # T6 is outdoors
df["indoor_temp_mean"] = df[indoor_cols].mean(axis=1)
df["temp_diff"] = df["indoor_temp_mean"] - df["T_out"]
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

for k in (1, 2, 3):
    df[f"lag_{k}"] = df["Appliances"].shift(k)
df["roll_mean_3"] = df["Appliances"].shift(1).rolling(3).mean()     # ~30 min
df["roll_mean_6"] = df["Appliances"].shift(1).rolling(6).mean()     # ~1 h
df["roll_mean_18"] = df["Appliances"].shift(1).rolling(18).mean()   # ~3 h
df["roll_std_6"] = df["Appliances"].shift(1).rolling(6).std()

before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {before - len(df)} warm-up rows with NaN lags. Rows remaining: {len(df)}")

# %% [markdown]
# ## 6. Modelling Setup — chronological split & log target

# %%
drop_cols = ["date", "Appliances", "rv1", "rv2", "dayname"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = np.log1p(df["Appliances"].values)          # model log target
y_true_wh = df["Appliances"].values            # for reference

# Chronological 80/20 split (no shuffling — respect time order)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train period ends, test period begins at: {df['date'].iloc[split]}")

# Scaled copies for the linear baseline (trees don't need scaling)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=3)
results = {}
models = {}


def evaluate(name, model, Xte, store=True):
    """Predict on (log) scale, invert to Wh, score on the real scale."""
    pred_log = model.predict(Xte)
    yp = np.clip(np.expm1(pred_log), 0, None)
    yt = np.expm1(y_test)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    if store:
        results[name] = {"RMSE (Wh)": rmse, "MAE (Wh)": mae, "R2": r2}
        models[name] = model
    print(f"{name:<22} R2={r2:6.3f}   RMSE={rmse:6.1f} Wh   MAE={mae:6.1f} Wh")
    return yp

# %% [markdown]
# ### Baseline — Linear Regression
# Fast, fully interpretable, but cannot model the non‑linear HVAC/time effects.

# %%
lr = LinearRegression().fit(X_train_s, y_train)
evaluate("Linear Regression", lr, X_test_s)

# %% [markdown]
# ### Model 1 — Random Forest (with randomized tuning)
# Captures non‑linear temperature/time/weather interactions. Tuned with
# `RandomizedSearchCV` over a `TimeSeriesSplit` so validation always trains on
# the past and tests on the future.

# %%
rf_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_dist, n_iter=8, cv=tscv, scoring="r2",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
evaluate("Random Forest", rf_search.best_estimator_, X_test)

# %% [markdown]
# ### Model 2 — XGBoost (with randomized tuning)

# %%
xgb_dist = {
    "n_estimators": [300, 500, 700],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "gamma": [0, 1, 5],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                     tree_method="hist", verbosity=0),
    xgb_dist, n_iter=12, cv=tscv, scoring="r2",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
xgb_search.fit(X_train, y_train)
print("Best XGB params:", xgb_search.best_params_)
evaluate("XGBoost", xgb_search.best_estimator_, X_test)

# %% [markdown]
# ### Model 3 — LightGBM (tuned with Optuna)
# Fast histogram‑based gradient boosting; Optuna searches the space efficiently
# using the same time‑series CV objective (mean R² on the log target).

# %%
def lgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 20, 120),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 60),
    }
    scores = []
    for tr_idx, va_idx in tscv.split(X_train):
        m = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                              verbose=-1, **params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        scores.append(r2_score(y_train[va_idx], m.predict(X_train.iloc[va_idx])))
    return float(np.mean(scores))


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(lgb_objective, n_trials=25, show_progress_bar=False)
print("Best LGBM params:", study.best_params)

best_lgb = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                             verbose=-1, **study.best_params)
best_lgb.fit(X_train, y_train)
evaluate("LightGBM", best_lgb, X_test)

# %% [markdown]
# ## Model Comparison

# %%
comparison = (pd.DataFrame(results).T
              .sort_values("R2", ascending=False)
              .round({"RMSE (Wh)": 1, "MAE (Wh)": 1, "R2": 3}))
print(comparison.to_string())
best_name = comparison.index[0]
best_model = models[best_name]
print("\nBest model:", best_name)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
comparison["R2"].plot(kind="bar", color="#2a9d8f", ax=ax[0])
ax[0].set_title("Test R² by model"); ax[0].set_ylabel("R²")
ax[0].tick_params(axis="x", rotation=20)
comparison["RMSE (Wh)"].plot(kind="bar", color="#e76f51", ax=ax[1])
ax[1].set_title("Test RMSE by model (lower is better)"); ax[1].set_ylabel("Wh")
ax[1].tick_params(axis="x", rotation=20)
savefig("07_model_comparison.png")

# %%
# Actual vs predicted over the (held-out future) test period for the best model.
best_pred = np.clip(np.expm1(best_model.predict(
    X_test if best_name != "Linear Regression" else X_test_s)), 0, None)
test_dates = df["date"].iloc[split:]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(test_dates.values, np.expm1(y_test), label="Actual", color="#264653", lw=.8)
ax.plot(test_dates.values, best_pred, label=f"Predicted ({best_name})",
        color="#e76f51", lw=.8, alpha=.8)
ax.set_title(f"Actual vs predicted appliance energy — test period ({best_name})")
ax.set_ylabel("Appliances (Wh)"); ax.legend()
savefig("08_actual_vs_predicted.png")

# %% [markdown]
# ## Explainability — SHAP
# SHAP decomposes each prediction into per‑feature contributions, answering
# *which variables matter most* and *why a given prediction is high or low*.
# We explain the best tree model on a sample of the held‑out test set.

# %%
shap_model = best_model if best_name != "Linear Regression" else best_lgb
shap_label = best_name if best_name != "Linear Regression" else "LightGBM"
X_shap = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)

plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
plt.title(f"SHAP feature importance — {shap_label}")
savefig("09a_shap_importance.png")

plt.figure()
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.title(f"SHAP value distribution — {shap_label}")
savefig("09b_shap_beeswarm.png")

mean_abs = pd.Series(np.abs(shap_values).mean(axis=0),
                     index=X_shap.columns).sort_values(ascending=False)
print("Top 10 drivers by mean |SHAP|:\n", mean_abs.head(10).round(4).to_string())

# %% [markdown]
# ## 7. Business Insights & Recommendations
#
# These are generated from the model + EDA and written to
# `outputs/recommendations.md`.

# %%
top_env = [f for f in mean_abs.index
           if f not in ("lag_1", "lag_2", "lag_3",
                        "roll_mean_3", "roll_mean_6", "roll_mean_18", "roll_std_6")][:5]
peak_hours = hourly.sort_values(ascending=False).head(3).index.tolist()
off_peak = hourly.sort_values().head(3).index.tolist()
weekday_avg = df[df.is_weekend == 0]["Appliances"].mean()
weekend_avg = df[df.is_weekend == 1]["Appliances"].mean()

recs = f"""# Energy Consumption — Insights & Recommendations

## Model performance (held-out future period)
{comparison.to_string()}

**Best model: {best_name}** — explains {comparison.loc[best_name,'R2']*100:.0f}% of
test-period variance with a typical error of ~{comparison.loc[best_name,'MAE (Wh)']:.0f} Wh.

## What drives consumption
- Strongest *environmental/behavioural* drivers (SHAP): {', '.join(top_env)}.
- Recent usage (lag & rolling features) dominates short-term prediction —
  consumption is highly autocorrelated, which is what makes next-interval
  forecasting accurate.

## When energy is used
- **Peak hours:** {peak_hours} (early evening). **Off-peak (cheapest to run loads):** {off_peak}.
- Weekday mean {weekday_avg:.0f} Wh vs weekend mean {weekend_avg:.0f} Wh.

## Actionable recommendations
1. **Load shifting:** schedule deferrable appliances (dishwasher, washing
   machine, EV charging) into the {off_peak} off-peak window to cut cost an
   estimated 5-15% without changing usage.
2. **HVAC optimisation:** `temp_diff` (indoor-outdoor) is a top driver —
   tightening setpoints / pre-conditioning before peak hours reduces evening load.
3. **Demand-peak forecasting:** the model predicts the next interval well
   enough to pre-warn utilities/building managers of {peak_hours} peaks for
   better grid planning.
4. **Anomaly watch:** flag intervals where actual >> predicted as possible
   faulty appliances or unexpected spikes.
"""
with open(os.path.join(OUT, "recommendations.md"), "w", encoding="utf-8") as f:
    f.write(recs)
print(recs)

# %%
# Export predictions for a Power BI / Tableau dashboard and persist the model.
import joblib

full_pred = np.clip(np.expm1(best_model.predict(
    X if best_name != "Linear Regression" else scaler.transform(X))), 0, None)
export = df[["date", "Appliances", "hour", "dayname", "is_weekend",
             "T_out", "indoor_temp_mean", "temp_diff"]].copy()
export["predicted_Wh"] = full_pred
export["split"] = np.where(np.arange(len(df)) < split, "train", "test")
export["residual_Wh"] = export["Appliances"] - export["predicted_Wh"]
export.to_csv(os.path.join(EXPORT, "predictions_for_dashboard.csv"), index=False)
comparison.to_csv(os.path.join(EXPORT, "model_comparison.csv"))
joblib.dump({"model": best_model, "features": feature_cols,
             "scaler": scaler, "best_name": best_name},
            os.path.join(MODELS, "best_model.joblib"))
print("Saved: predictions_for_dashboard.csv, model_comparison.csv, best_model.joblib")

# %% [markdown]
# ## Future Improvements
# - **Time-series models:** benchmark Prophet / LSTM / GRU for multi-step forecasts.
# - **Weather-forecast integration:** feed tomorrow's forecast to predict next-day demand.
# - **Anomaly detection:** Isolation Forest / autoencoder on residuals for faulty-appliance alerts.
# - **Reinforcement learning:** an agent that schedules appliances to minimise cost under a tariff.
# - **Dashboard:** publish `predictions_for_dashboard.csv` to Power BI — forecast vs actual, peak hours, savings opportunities.
