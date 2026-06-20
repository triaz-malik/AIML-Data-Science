"""Model explainability (SHAP) and diagnostic plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def plot_predicted_vs_actual(y_true_log, y_pred_log, name: str):
    """Scatter of predicted vs actual SalePrice (back-transformed to $)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolor="none")
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=2)
    ax.set_xlabel("Actual SalePrice ($)")
    ax.set_ylabel("Predicted SalePrice ($)")
    ax.set_title(f"Predicted vs Actual — {name}")
    path = C.FIGURE_DIR / "07_predicted_vs_actual.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.ROOT_DIR)}")


def shap_summary(model, X_sample: pd.DataFrame, name: str):
    """TreeExplainer SHAP summary for a fitted tree model."""
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP summary")
        return

    print(f"Computing SHAP values for {name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    path = C.FIGURE_DIR / "08_shap_summary.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.ROOT_DIR)}")

    # Bar of mean |SHAP| as a clean feature-importance ranking.
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                      max_display=15)
    path = C.FIGURE_DIR / "09_shap_importance.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.ROOT_DIR)}")
