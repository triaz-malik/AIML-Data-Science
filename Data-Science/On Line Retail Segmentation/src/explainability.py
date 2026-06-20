"""
Phase 9 - Explainability (SHAP on the CLV model).

A black-box CLV score is hard to act on. SHAP decomposes each prediction into
per-feature contributions, answering:

  * Globally: which behaviours drive future spend? (summary / importance plots)
  * Locally : why is THIS customer predicted high or low value? (waterfall)

The CLV model is trained on log1p(future_revenue), so SHAP values are in log-
revenue space — directionally interpretable (positive pushes value up), which is
what the business needs to understand the "why" behind a score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def run() -> None:
    bundle = __import__("joblib").load(config.MODEL_DIR / "clv_model.joblib")
    model, features = bundle["model"], bundle["features"]
    data = pd.read_parquet(config.PROCESSED_DIR / "clv_dataset.parquet")

    # Explain on a manageable sample for speed/clarity.
    sample = data.sample(min(1500, len(data)), random_state=config.RANDOM_STATE)
    X = sample[features]

    print("Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global: beeswarm summary
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP summary — drivers of predicted future spend")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "11_shap_summary.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close()

    # Global: mean |SHAP| bar = feature importance
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP feature importance (mean |impact|)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "12_shap_importance.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close()

    importance = (pd.DataFrame({"feature": features,
                                "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False))
    importance.to_csv(config.REPORT_DIR / "shap_importance.csv", index=False)

    # Local: explain the highest-value customer in the sample
    top_idx = sample["future_revenue"].values.argmax()
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[top_idx],
        feature_names=features, show=False)
    plt.title("Why this customer is high-value (SHAP waterfall)")
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "13_shap_waterfall_topcustomer.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close()

    print("\nSHAP global importance (drivers of future spend):")
    print(importance.round(4).to_string(index=False))
    print(f"\nSaved SHAP figures -> {config.FIGURE_DIR}")


if __name__ == "__main__":
    run()
