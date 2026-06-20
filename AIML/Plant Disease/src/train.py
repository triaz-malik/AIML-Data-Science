"""Phases 5-6 — train the baseline CNN, ResNet50 and EfficientNetB0 and
compare them.

Run:
  python -m src.train                 # train all three (default epochs)
  python -m src.train --models resnet50 --epochs 8
  python -m src.train --smoke         # 1 epoch, 60 train batches — wiring test
"""
from __future__ import annotations

import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C
from .data import (build_label_encoder, class_weights, make_loaders,
                   make_split, scan_dataset)
from .engine import fit

DEFAULT_MODELS = ["baseline_cnn", "resnet50", "efficientnet_b0"]


def prepare(use_clean: bool = True):
    """Scan (or load clean catalog), build leak-free split + label encoder."""
    clean_csv = C.SPLITS_DIR / "clean_catalog.csv"
    if use_clean and clean_csv.exists():
        df = pd.read_csv(clean_csv)
        print(f"Loaded clean catalog: {len(df):,} files")
    else:
        df = scan_dataset()
        print(f"Scanned: {len(df):,} files")
    df = make_split(df)
    df.to_csv(C.SPLITS_DIR / "split.csv", index=False)
    le = build_label_encoder(df)
    with open(C.MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    sizes = df["split"].value_counts().to_dict()
    print(f"Split (files): {sizes}")
    print(f"Classes: {len(le.classes_)} -> label_encoder.pkl saved")
    return df, le


def plot_curves(name: str, history: dict):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    ax[0].plot(ep, history["train_loss"], marker="o")
    ax[0].set_title(f"{name} — training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, history["val_acc"], marker="o", label="val acc")
    ax[1].plot(ep, history["val_f1"], marker="s", label="val macro-F1")
    ax[1].set_title(f"{name} — validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.savefig(C.PLOTS_DIR / f"train_curve_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--smoke", action="store_true",
                    help="fast wiring test: 1 epoch, few batches")
    args = ap.parse_args()

    C.seed_everything()
    df, le = prepare()
    n_classes = len(le.classes_)
    cw = class_weights(df, le)
    train_loader, val_loader, test_loader = make_loaders(df, le, args.batch_size)

    if args.smoke:
        # truncate loaders for a quick wiring check
        from itertools import islice

        class _Lim:
            def __init__(self, loader, n): self.loader, self.n = loader, n
            def __iter__(self): return islice(iter(self.loader), self.n)
            def __len__(self): return self.n
            @property
            def dataset(self): return self.loader.dataset
        train_loader = _Lim(train_loader, 60)
        val_loader = _Lim(val_loader, 20)
        args.epochs = 1

    results = []
    for name in args.models:
        print(f"\n=== Training {name} ({args.epochs} epochs) ===")
        save_path = C.MODELS_DIR / f"{name}.pt"
        res = fit(
            name, train_loader, val_loader, n_classes,
            class_weights=cw, epochs=args.epochs,
            save_path=save_path, le=le,
        )
        plot_curves(name, res.history)
        with open(C.MODELS_DIR / f"{name}_history.json", "w") as f:
            json.dump(res.history, f, indent=2)
        results.append({
            "model": name,
            "val_acc": round(res.best_val_acc, 4),
            "val_f1": round(res.best_val_f1, 4),
            "minutes": round(res.seconds / 60, 1),
            "ckpt": res.ckpt_path,
        })
        print(f"  -> best val_acc={res.best_val_acc:.4f} "
              f"val_f1={res.best_val_f1:.4f} in {res.seconds/60:.1f} min")

    comp = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    comp.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison (val) ===")
    print(comp.to_string(index=False))

    # comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(comp)); import numpy as np
    ax.bar([i - 0.2 for i in x], comp["val_acc"], width=0.4, label="val acc")
    ax.bar([i + 0.2 for i in x], comp["val_f1"], width=0.4, label="val macro-F1")
    ax.set_xticks(list(x)); ax.set_xticklabels(comp["model"], rotation=15)
    ax.set_ylim(0, 1); ax.set_title("Model comparison (validation)"); ax.legend()
    for i, (a, fscore) in enumerate(zip(comp["val_acc"], comp["val_f1"])):
        ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, fscore + 0.01, f"{fscore:.2f}", ha="center", fontsize=8)
    fig.savefig(C.PLOTS_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison -> {(C.REPORTS_DIR/'model_comparison.csv')}")


if __name__ == "__main__":
    main()
