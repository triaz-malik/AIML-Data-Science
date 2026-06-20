"""Phase 4 - Baseline Models (classical ML before deep learning).

Linear Regression, Random Forest, XGBoost on the engineered feature table,
predicting next-hour temperature. Also includes a naive persistence baseline
(predict temp[t+1] = temp[t]) as the floor every model must beat.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import config as C
import modeling as M


def main():
    print("[Phase 4] Loading features ...")
    df = M.load_feature_table()
    s = M.tabular_splits(df)
    Xtr, ytr = s["X_train"], s["y_train"]
    Xva, yva = s["X_val"], s["y_val"]
    Xte, yte = s["X_test"], s["y_test"]
    # Combine train+val for final fit of baselines (they don't early-stop except XGB).
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])

    results = {}
    preds_store = {}

    # --- Naive persistence baseline ---
    # temp_lag_1 column holds temperature[t-1]; current temp is the 'temperature'
    # column. Persistence predicts target(t+1)=temperature(t).
    temp_idx = s["feat_cols"].index("temperature")
    naive_pred = Xte[:, temp_idx]
    results["Persistence"] = M.metrics(yte, naive_pred)
    print("  Persistence:", results["Persistence"])

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(Xtrva, ytrva)
    p = lr.predict(Xte)
    results["LinearRegression"] = M.metrics(yte, p)
    preds_store["LinearRegression"] = p
    print("  LinearRegression:", results["LinearRegression"])

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    rf.fit(Xtrva, ytrva)
    p = rf.predict(Xte)
    results["RandomForest"] = M.metrics(yte, p)
    preds_store["RandomForest"] = p
    print("  RandomForest:", results["RandomForest"])
    joblib.dump(rf, f"{C.MODEL_DIR}/random_forest.joblib")

    # --- XGBoost (with early stopping on val) ---
    xgb = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=42, early_stopping_rounds=30, eval_metric="rmse",
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p = xgb.predict(Xte)
    results["XGBoost"] = M.metrics(yte, p)
    preds_store["XGBoost"] = p
    print("  XGBoost:", results["XGBoost"])
    xgb.save_model(f"{C.MODEL_DIR}/xgboost.json")

    # Feature importances (XGBoost + RF) for later reference
    fi = pd.DataFrame({
        "feature": s["feat_cols"],
        "xgb_importance": xgb.feature_importances_,
        "rf_importance": rf.feature_importances_,
    }).sort_values("xgb_importance", ascending=False)
    fi.to_csv(f"{C.TABLE_DIR}/phase4_feature_importance.csv", index=False)

    # Save metrics + test predictions for the combined comparison later.
    with open(f"{C.TABLE_DIR}/phase4_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        f"{C.TABLE_DIR}/phase4_test_predictions.npz",
        y_test=yte,
        test_index=s["test_index"].astype("int64").values,
        **preds_store,
    )
    print("[Phase 4] Done. Metrics saved.")
    print(pd.DataFrame(results).T[["MAE", "RMSE", "MAPE", "R2"]])


if __name__ == "__main__":
    main()
