"""Phase 7 — Hyperparameter tuning with Optuna.

Tunes learning rate, batch size, dropout, optimizer and weight decay on the
chosen architecture. To keep trials cheap, each trial trains a few epochs on a
stratified *subsample*; the winning config is then retrained on the full data
to produce ``best_model.pt``.

Run:  python -m src.tune --model efficientnet_b0 --trials 15
"""
from __future__ import annotations

import argparse
import json
import pickle

import optuna
import pandas as pd
from torch.utils.data import DataLoader

from . import config as C
from .data import (PlantDataset, build_label_encoder, class_weights,
                   eval_transform, make_split, scan_dataset, train_transform)
from .engine import fit
from .train import prepare


def _subsample(df, per_split_frac=0.35, seed=C.SEED):
    """Shrink each split for fast trials (group-safe: rows keep their split)."""
    out = []
    for s, sub in df.groupby("split"):
        out.append(sub.groupby("label", group_keys=False)
                      .sample(frac=per_split_frac, random_state=seed))
    return pd.concat(out).reset_index(drop=True)


def _loaders(df, le, batch_size):
    def mk(split, tf, shuffle):
        sub = df[df.split == split].reset_index(drop=True)
        return DataLoader(PlantDataset(sub, le, tf), batch_size=batch_size,
                          shuffle=shuffle, num_workers=C.NUM_WORKERS,
                          pin_memory=(C.DEVICE == "cuda"),
                          persistent_workers=(C.NUM_WORKERS > 0),
                          prefetch_factor=(4 if C.NUM_WORKERS > 0 else None),
                          drop_last=shuffle)
    return mk("train", train_transform(), True), mk("val", eval_transform(), False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--trial-epochs", type=int, default=3)
    ap.add_argument("--final-epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--subsample", type=float, default=0.35)
    args = ap.parse_args()

    C.seed_everything()
    df, le = prepare()
    n_classes = len(le.classes_)
    cw = class_weights(df, le)
    sdf = _subsample(df, args.subsample)
    print(f"Tuning {args.model} on subsample: "
          f"{sdf['split'].value_counts().to_dict()}")

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        tr, va = _loaders(sdf, le, batch_size)
        res = fit(args.model, tr, va, n_classes, class_weights=cw,
                  epochs=args.trial_epochs, lr=lr, weight_decay=wd,
                  optimizer=optimizer, dropout=dropout, verbose=False)
        trial.set_user_attr("val_acc", res.best_val_acc)
        return res.best_val_f1

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    print(f"\nBest trial: val_f1={study.best_value:.4f}  params={best}")
    (C.REPORTS_DIR / "best_hparams.json").write_text(
        json.dumps({"model": args.model, "val_f1": study.best_value,
                    "params": best}, indent=2), encoding="utf-8")

    # save optimisation history + importances
    try:
        import matplotlib
        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import (plot_optimization_history,
                                                     plot_param_importances)
        import matplotlib.pyplot as plt
        ax = plot_optimization_history(study); ax.figure.tight_layout()
        ax.figure.savefig(C.PLOTS_DIR / "optuna_history.png", dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
        ax = plot_param_importances(study); ax.figure.tight_layout()
        ax.figure.savefig(C.PLOTS_DIR / "optuna_importances.png", dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
    except Exception as e:
        print(f"(viz skipped: {e})")

    # ---- retrain best config on FULL data -> best_model.pt ----
    print(f"\nRetraining {args.model} with best params on full data "
          f"({args.final_epochs} epochs)...")
    tr, va = _loaders(df, le, best["batch_size"])
    res = fit(args.model, tr, va, n_classes, class_weights=cw,
              epochs=args.final_epochs, lr=best["lr"],
              weight_decay=best["weight_decay"], optimizer=best["optimizer"],
              dropout=best["dropout"], le=le,
              save_path=C.MODELS_DIR / "best_model.pt")
    print(f"best_model.pt -> val_acc={res.best_val_acc:.4f} "
          f"val_f1={res.best_val_f1:.4f}")
    # trial table
    study.trials_dataframe().to_csv(C.REPORTS_DIR / "optuna_trials.csv", index=False)


if __name__ == "__main__":
    main()
