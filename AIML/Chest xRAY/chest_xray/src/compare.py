"""Aggregate test metrics across models into a comparison table + bar chart.

Reads outputs/reports/test_metrics_*.json (one per evaluated model) and writes:
  outputs/reports/model_comparison.csv
  outputs/figures/model_comparison.png

Run after evaluating each model:
  python -m src.compare
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C


def run():
    rows = []
    for f in sorted(C.REPORT_DIR.glob("test_metrics_*.json")):
        d = json.loads(f.read_text())
        m = d["default_0.5"]
        r = d["recall_tuned"]
        rows.append({
            "model": d["model"],
            "recall": m["recall"], "precision": m["precision"],
            "f1": m["f1"], "auc": m["auc"], "accuracy": m["accuracy"],
            "recall_tuned": r["recall"], "recall_tuned_thr": r["threshold"],
            "fn_default": m["fn"],
        })
    if not rows:
        print("No test_metrics_*.json found. Run src.evaluate first.")
        return
    df = pd.DataFrame(rows).sort_values("recall", ascending=False)
    df.to_csv(C.REPORT_DIR / "model_comparison.csv", index=False)
    print(df.to_string(index=False))

    ax = df.set_index("model")[["recall", "f1", "auc", "accuracy"]].plot(
        kind="bar", figsize=(9, 5))
    ax.set_title("Model comparison (test set)")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "model_comparison.png", dpi=120)
    plt.close()
    print(f"\nSaved -> {C.REPORT_DIR/'model_comparison.csv'} and "
          f"{C.FIG_DIR/'model_comparison.png'}")


if __name__ == "__main__":
    run()
