"""Phase 9 - Explainability with SHAP.

SHAP values on the trained XGBoost model reveal which weather drivers move the
next-hour temperature prediction (humidity, pressure, wind, recent-temperature
lags, etc.). TreeSHAP is exact and fast for gradient-boosted trees, so this is
the natural explainer for our best tabular model.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

import config as C
import modeling as M

plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name):
    fig.savefig(f"{C.FIG_DIR}/{name}.png", dpi=110)
    plt.close(fig)
    print(f"  saved {name}.png")


def main():
    print("[Phase 9] Loading XGBoost model + data ...")
    df = M.load_feature_table()
    s = M.tabular_splits(df)
    feat_cols = s["feat_cols"]

    xgb = XGBRegressor()
    xgb.load_model(f"{C.MODEL_DIR}/xgboost.json")

    # Explain on a sample of the test set (keep it fast).
    Xte = pd.DataFrame(s["X_test"], columns=feat_cols)
    sample = Xte.sample(min(2000, len(Xte)), random_state=42)

    print("[Phase 9] Computing SHAP values (TreeSHAP) ...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(sample)

    # 1. Global importance (bar)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False,
                      max_display=18)
    save(fig, "22_shap_importance_bar")

    # 2. Beeswarm (direction + magnitude)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, sample, show=False, max_display=18)
    save(fig, "23_shap_beeswarm")

    # Mean |SHAP| ranking table
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
               .sort_values("mean_abs_shap", ascending=False)
               .reset_index(drop=True))
    ranking.to_csv(f"{C.TABLE_DIR}/phase9_shap_ranking.csv", index=False)

    # Highlight the business-relevant drivers explicitly.
    drivers = {}
    for name in ["humidity", "pressure", "wind_speed", "humidity_lag_1",
                 "pressure_lag_1", "temp_lag_1", "temp_roll_mean_24"]:
        if name in ranking["feature"].values:
            drivers[name] = float(
                ranking.loc[ranking.feature == name, "mean_abs_shap"].iloc[0])
    with open(f"{C.TABLE_DIR}/phase9_key_drivers.json", "w") as f:
        json.dump(drivers, f, indent=2)

    print("\n[Phase 9] Top 12 drivers by mean |SHAP|:")
    print(ranking.head(12).to_string(index=False))
    print("[Phase 9] Done.")


if __name__ == "__main__":
    main()
