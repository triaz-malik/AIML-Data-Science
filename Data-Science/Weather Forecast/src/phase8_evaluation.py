"""Phase 8 - Evaluation.

Consolidates metrics from the baselines (Phase 4) and the deep-learning models
(Phase 5), builds the master comparison table (MAE / RMSE / MAPE / R2) and the
key comparison figures: actual-vs-predicted, error distribution, model bars,
and the multi-step error-growth curve (Phase 7).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as C

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name):
    fig.savefig(f"{C.FIG_DIR}/{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default or {}


def main():
    print("[Phase 8] Consolidating metrics ...")
    base = load_json(f"{C.TABLE_DIR}/phase4_baseline_metrics.json")
    dl = load_json(f"{C.TABLE_DIR}/phase5_dl_metrics.json")

    all_metrics = {**base, **dl}
    table = pd.DataFrame(all_metrics).T[["MAE", "RMSE", "MAPE", "R2"]]
    table = table.sort_values("RMSE")
    table.to_csv(f"{C.TABLE_DIR}/master_comparison.csv")
    # Markdown version for the report
    with open(f"{C.TABLE_DIR}/master_comparison.md", "w") as f:
        f.write(table.round(4).to_markdown())
    print(table)

    # 1. Model comparison bars (RMSE & MAE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = table.index.tolist()
    sns.barplot(x=table["RMSE"], y=order, ax=axes[0], color="#c0392b")
    axes[0].set_title("Test RMSE by Model (lower is better)")
    axes[0].set_xlabel("RMSE (°C)")
    sns.barplot(x=table["MAE"], y=order, ax=axes[1], color="#2980b9")
    axes[1].set_title("Test MAE by Model")
    axes[1].set_xlabel("MAE (°C)")
    save(fig, "17_model_comparison")

    # 2. Actual vs predicted (best DL model) on a test slice
    dl_preds_path = f"{C.TABLE_DIR}/phase5_test_predictions.npz"
    try:
        npz = np.load(dl_preds_path)
        y_test = npz["y_test"]
        # pick best DL model by RMSE
        best_dl = min(dl, key=lambda k: dl[k]["RMSE"]) if dl else None
        if best_dl:
            pred = npz[best_dl]
            sl = slice(0, 300)
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(y_test[sl], label="Actual", color="#2c3e50", lw=1.3)
            ax.plot(pred[sl], label=f"Predicted ({best_dl})", color="#e74c3c",
                    lw=1.1, alpha=0.85)
            ax.set_title(f"Actual vs Predicted — {best_dl} (first 300 test hours)")
            ax.set_xlabel("Test hour")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            save(fig, "18_actual_vs_predicted")

            # 3. Error distribution
            err = y_test - pred
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(err, bins=60, kde=True, color="#8e44ad", ax=ax)
            ax.set_title(f"Prediction Error Distribution — {best_dl}")
            ax.set_xlabel("Error (°C)")
            save(fig, "19_error_distribution")
    except FileNotFoundError:
        print("  (DL predictions not found; skipping AvP plot)")

    # 4. Multi-step error growth (Phase 7)
    try:
        samp = np.load(f"{C.TABLE_DIR}/phase7_samples.npz")
        fig, ax = plt.subplots(figsize=(10, 4))
        for H in [24, 48, 168]:
            key = f"{H}h_perstep_rmse"
            if key in samp:
                ax.plot(range(1, len(samp[key]) + 1), samp[key],
                        label=f"{H}h model")
        ax.set_title("Multi-Step Forecast Error Growth (RMSE vs lead time)")
        ax.set_xlabel("Lead time (hours ahead)")
        ax.set_ylabel("RMSE (°C)")
        ax.legend()
        save(fig, "20_multistep_error_growth")

        # 5. Example 7-day forecast
        if "168h_true" in samp:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(samp["168h_true"], label="Actual", color="#2c3e50")
            ax.plot(samp["168h_pred"], label="Forecast", color="#e67e22")
            ax.set_title("Example 7-Day (168h) Temperature Forecast")
            ax.set_xlabel("Hours ahead")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            save(fig, "21_example_7day_forecast")
    except FileNotFoundError:
        print("  (Multi-step samples not found; skipping growth plot)")

    # Save multi-step metrics into a tidy table if present
    ms = load_json(f"{C.TABLE_DIR}/phase7_multistep_metrics.json")
    if ms:
        pd.DataFrame(ms).T.to_csv(f"{C.TABLE_DIR}/multistep_comparison.csv")

    print("[Phase 8] Done.")


if __name__ == "__main__":
    main()
