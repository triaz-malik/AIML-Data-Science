"""Evaluate saved models on the held-out test split: metrics, plots, comparison table."""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, auc, confusion_matrix,
                             precision_recall_curve, roc_curve)

import config as C
from preprocessing import get_split

sns.set_theme(style="whitegrid")
LABELS = ["normal (0)", "anomaly (1)"]


def _load_models():
    models = {}
    for name, path in C.MODEL_FILES.items():
        if path.exists():
            models[name] = joblib.load(path)
    return models


def run():
    X_train, X_test, y_train, y_test = get_split()
    models = _load_models()
    if not models:
        raise SystemExit("No trained models found. Run train_models.py first.")

    # 1) Confusion matrices (grid) -----------------------------------------
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.suptitle("Confusion Matrices (held-out test)", y=1.03)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "07_confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[eval] saved 07_confusion_matrices.png")

    # 2) ROC curves (overlay) ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "08_roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[eval] saved 08_roc_curves.png")

    # 3) Precision-Recall curves -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP-AUC={auc(rec, prec):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "09_pr_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[eval] saved 09_pr_curves.png")

    # 4) Comparison table (from metrics.json) ------------------------------
    if C.METRICS_JSON.exists():
        metrics = json.loads(C.METRICS_JSON.read_text())
        rows = []
        for name, m in metrics["models"].items():
            t = m["test"]
            rows.append({
                "Model": name,
                "Accuracy": t["accuracy"], "Precision": t["precision"],
                "Recall": t["recall"], "F1": t["f1"], "ROC AUC": t["roc_auc"],
                "CV F1 (5-fold)": m["cv_f1_mean"],
            })
        table = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
        table.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
        print("\n[eval] === Model Comparison (held-out test) ===")
        print(table.round(4).to_string(index=False))
        print(f"\n[eval] BEST: {metrics.get('best_model')}")

        # bar chart of F1
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=table, x="F1", y="Model", ax=ax, palette="crest",
                    hue="Model", legend=False)
        ax.set_xlim(min(0.9, table["F1"].min() - 0.02), 1.0)
        ax.set_title("Model F1 Comparison (held-out test)")
        for i, v in enumerate(table["F1"]):
            ax.text(v, i, f" {v:.4f}", va="center")
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "10_model_comparison.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("[eval] saved 10_model_comparison.png + model_comparison.csv")


if __name__ == "__main__":
    run()
