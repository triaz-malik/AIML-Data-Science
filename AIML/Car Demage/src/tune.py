"""Phase 7 - Hyperparameter tuning with Optuna.

Searches learning rate, batch size, dropout and weight decay for a given model,
optimizing validation accuracy. To keep tuning fast we split a tuning-validation
set out of the TRAINING data (the real validation/ set stays untouched as the
final test set).

Usage:
    python src/tune.py --model efficientnet_b0 --trials 15 --epochs 5
"""
from __future__ import annotations

import argparse
import json

import optuna
import torch
from torch.utils.data import DataLoader

from config import MODELS_DIR, NUM_WORKERS, REPORTS_DIR, TRAIN_DIR, set_seed
from data import CarDamageDataset, build_transforms, list_images
from engine import train_model
from models import build_model


def make_loaders(img_size, batch_size, val_frac=0.2):
    samples = list_images(TRAIN_DIR)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(samples), generator=g).tolist()
    n_val = int(len(samples) * val_frac)
    val_idx, tr_idx = set(perm[:n_val]), perm[n_val:]
    tr = [samples[i] for i in tr_idx]
    va = [samples[i] for i in val_idx]
    tr_ds = CarDamageDataset(tr, build_transforms(img_size, train=True))
    va_ds = CarDamageDataset(va, build_transforms(img_size, train=False))
    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()
    set_seed()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        tr_loader, va_loader = make_loaders(args.img_size, batch_size)
        model = build_model(args.model, num_classes=2, dropout=dropout, pretrained=True)
        _, _, best_acc = train_model(
            model, tr_loader, va_loader, epochs=args.epochs, lr=lr,
            weight_decay=weight_decay, patience=3, verbose=False,
        )
        trial.report(best_acc, step=0)
        return best_acc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    print(f"Tuning {args.model}: {args.trials} trials x up to {args.epochs} epochs ...")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    result = {"model": args.model, "best_value": study.best_value,
              "best_params": study.best_params,
              "trials": [{"value": t.value, "params": t.params} for t in study.trials]}
    out = REPORTS_DIR / f"optuna_{args.model}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved -> {out}")
    print("\nRetrain the final model with these params via src/train.py, e.g.:")
    p = study.best_params
    print(f"  python src/train.py --model {args.model} --epochs 15 "
          f"--lr {p['lr']:.2e} --batch-size {p['batch_size']} --dropout {p['dropout']:.2f} "
          f"--weight-decay {p['weight_decay']:.2e}")


if __name__ == "__main__":
    main()
