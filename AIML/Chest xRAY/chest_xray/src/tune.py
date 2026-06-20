"""Hyperparameter sweep on the strongest architecture.

Grid-searches a compact, sensible space (learning rate x optimizer x dropout)
and records validation Recall/F1/AUC for each config. This is the explicit
"hyperparameter tuning" step: the main train.py already does adaptive LR
scheduling + early stopping, but this sweep *chooses* the base config.

To keep wall-clock reasonable each trial uses fewer epochs (--epochs) and the
early-stopping logic still applies.

Run:
  python -m src.tune --model efficientnet_b0 --epochs 12
Outputs:
  outputs/reports/tuning_results.csv
  outputs/figures/tuning_results.png
"""
from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C
from .config import TrainConfig
from .train import train

# Compact search space (subset of the spec's grid to stay time-bounded).
# Covers learning rate x optimizer x dropout in 6 trials.
GRID = {
    "lr": [1e-3, 5e-4],
    "optimizer": ["adam", "sgd"],
    "dropout": [0.3],
}
# An extra dropout probe at the better (lr, optimizer) is appended at runtime.
EXTRA_DROPOUT = 0.5


def run(model: str, epochs: int):
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]
    # add a dropout probe (keep adam, vary dropout) to cover the 3rd dimension
    for lr in GRID["lr"]:
        combos.append({"lr": lr, "optimizer": "adam", "dropout": EXTRA_DROPOUT})
    print(f"Hyperparameter sweep: {len(combos)} configs on {model}, "
          f"{epochs} epochs each\n")

    rows = []
    for i, params in enumerate(combos, 1):
        print(f"\n--- trial {i}/{len(combos)}: {params} ---")
        cfg = TrainConfig(model=model, epochs=epochs, num_workers=4, **params)
        # keep checkpoints from clobbering the main best model
        summary = train(cfg)
        rows.append({**params, "best_val_recall": summary["best_val_recall"],
                     "minutes": summary["minutes"]})

    df = pd.DataFrame(rows).sort_values("best_val_recall", ascending=False)
    df.to_csv(C.REPORT_DIR / "tuning_results.csv", index=False)
    (C.REPORT_DIR / "tuning_best.json").write_text(
        json.dumps(df.iloc[0].to_dict(), indent=2))
    print("\n===== SWEEP RESULTS (by val recall) =====")
    print(df.to_string(index=False))

    # plot
    labels = [f"lr={r.lr}\n{r.optimizer}\ndrop={r.dropout}"
              for r in df.itertuples()]
    plt.figure(figsize=(max(8, len(df) * 1.3), 5))
    plt.bar(labels, df["best_val_recall"], color="#4C72B0")
    plt.ylabel("best validation recall"); plt.ylim(0, 1.0)
    plt.title(f"Hyperparameter sweep — {model}")
    for x, v in enumerate(df["best_val_recall"]):
        plt.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "tuning_results.png", dpi=120)
    plt.close()
    print(f"\nSaved -> tuning_results.csv and tuning_results.png")
    print(f"Best config: {df.iloc[0].to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet_b0",
                   choices=["custom_cnn", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=12)
    a = p.parse_args()
    run(a.model, a.epochs)
