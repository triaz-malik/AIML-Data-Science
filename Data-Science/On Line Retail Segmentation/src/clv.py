"""
Phase 8 - Customer Lifetime Value (future-revenue) prediction.

Business question: "How much will each customer spend over the next 6 months?"
Knowing this lets the retailer concentrate retention/marketing budget on the
customers who will actually pay it back.

Protocol (avoids look-ahead leakage):
  * Calibration window  : first CLV_CALIBRATION_MONTHS months -> build features
  * Holdout window      : following CLV_HOLDOUT_MONTHS months -> target revenue
  Only customers active during calibration are modelled; their target is the
  revenue they generate in the holdout window (0 if they churn).

The target is heavily right-skewed with a spike at 0, so we train on log1p(spend)
and report metrics back on the original currency scale. We compare Random Forest,
XGBoost and LightGBM on RMSE / MAE / R^2 and keep the best for SHAP (Phase 9).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sns.set_theme(style="whitegrid", palette=config.PALETTE)

FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketValue",
            "ProductDiversity", "Tenure", "AvgInterpurchase"]


def build_supervised(df: pd.DataFrame):
    start = df["InvoiceDate"].min()
    cal_end = start + pd.DateOffset(months=config.CLV_CALIBRATION_MONTHS)
    hold_end = cal_end + pd.DateOffset(months=config.CLV_HOLDOUT_MONTHS)
    print(f"  calibration: {start.date()} -> {cal_end.date()}")
    print(f"  holdout    : {cal_end.date()} -> {hold_end.date()}")

    cal = df[df["InvoiceDate"] < cal_end]
    hold = df[(df["InvoiceDate"] >= cal_end) & (df["InvoiceDate"] < hold_end)]

    # Calibration-period features (Recency measured to cal_end).
    g = cal.groupby("CustomerID")
    feats = g.agg(
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        ProductDiversity=("StockCode", "nunique"),
        last=("InvoiceDate", "max"),
        first=("InvoiceDate", "min"),
    )
    feats["Recency"] = (cal_end - feats["last"]).dt.days
    feats["Tenure"] = (feats["last"] - feats["first"]).dt.days
    feats["AvgBasketValue"] = (feats["Monetary"] / feats["Frequency"]).round(2)
    orders = (cal.groupby(["CustomerID", "Invoice"])["InvoiceDate"].max()
                 .reset_index())
    feats["AvgInterpurchase"] = (orders.sort_values("InvoiceDate")
                                 .groupby("CustomerID")["InvoiceDate"]
                                 .apply(lambda s: s.diff().dt.days.mean())
                                 .fillna(0).round(1))

    target = hold.groupby("CustomerID")["Revenue"].sum().rename("future_revenue")
    feats = feats.join(target).fillna({"future_revenue": 0.0})
    return feats.reset_index()


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train on log1p target, evaluate on original currency scale."""
    yl_train = np.log1p(y_train)
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1,
            random_state=config.RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=config.RANDOM_STATE),
        "LightGBM": LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=config.RANDOM_STATE, verbose=-1),
    }
    results, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, yl_train)
        pred = np.clip(np.expm1(model.predict(X_test)), 0, None)
        results.append({
            "model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })
        fitted[name] = model
    return pd.DataFrame(results).sort_values("RMSE"), fitted


def plot_results(results: pd.DataFrame, y_test, best_pred, best_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=results.melt(id_vars="model",
                                  value_vars=["RMSE", "MAE"]),
                x="model", y="value", hue="variable", ax=axes[0])
    axes[0].set_title("CLV model error (lower = better)")
    axes[0].set_ylabel("Error (currency)")

    cap = np.quantile(y_test, 0.99)
    axes[1].scatter(np.clip(y_test, 0, cap), np.clip(best_pred, 0, cap),
                    s=8, alpha=0.3, color="#3b528b")
    axes[1].plot([0, cap], [0, cap], ls="--", color="grey")
    axes[1].set_title(f"{best_name}: predicted vs actual (99th pct clip)")
    axes[1].set_xlabel("Actual future revenue")
    axes[1].set_ylabel("Predicted future revenue")
    fig.savefig(config.FIGURE_DIR / "10_clv_model_comparison.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    print("Building calibration/holdout supervised set ...")
    data = build_supervised(df)
    print(f"  {len(data):,} customers; "
          f"{(data['future_revenue'] > 0).mean():.1%} active in holdout")

    X, y = data[FEATURES], data["future_revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE)

    results, fitted = evaluate_models(X_train, X_test, y_train, y_test)
    print("\nModel comparison:")
    print(results.round(2).to_string(index=False))

    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    best_pred = np.clip(np.expm1(best_model.predict(X_test)), 0, None)
    plot_results(results, y_test.values, best_pred, best_name)

    results.to_csv(config.REPORT_DIR / "clv_model_comparison.csv", index=False)
    joblib.dump({"model": best_model, "name": best_name, "features": FEATURES},
                config.MODEL_DIR / "clv_model.joblib")
    # Persist the supervised frame for SHAP (Phase 9).
    data.to_parquet(config.PROCESSED_DIR / "clv_dataset.parquet", index=False)

    print(f"\nBest model: {best_name} (RMSE={results.iloc[0]['RMSE']:.2f}, "
          f"R2={results.iloc[0]['R2']:.3f})")
    print(f"Saved CLV model -> {config.MODEL_DIR / 'clv_model.joblib'}")


if __name__ == "__main__":
    run()
