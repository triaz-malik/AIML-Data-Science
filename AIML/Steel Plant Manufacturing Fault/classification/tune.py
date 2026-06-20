"""Phase 4: Optuna hyperparameter tuning for the winning architecture.

Tunes learning rate, batch size, dropout, optimizer and input width. To stay
tractable each trial trains a subset for a few epochs with median pruning; the
goal is to *rank* configurations, not fully train them.

Usage:
    python -m classification.tune --model resnet50 --trials 15 --epochs 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src import config, utils
from .data import ClsDataset, build_transforms, make_splits, pos_weights
from .models import build_model

# built once and reused across trials
_TRAIN_DF, _VAL_DF, _ANN = None, None, None


def _loaders(batch, img_w, train_n, val_n, workers):
    tr = _TRAIN_DF.sample(min(train_n, len(_TRAIN_DF)), random_state=config.SEED).reset_index(drop=True)
    va = _VAL_DF.sample(min(val_n, len(_VAL_DF)), random_state=config.SEED).reset_index(drop=True)
    size = (256, img_w)
    train_ds = ClsDataset(tr, img_size=size, transform=build_transforms(True))
    val_ds = ClsDataset(va, img_size=size, transform=build_transforms(False))
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0)
    return tl, vl, tr


def make_objective(model_name, epochs, train_n, val_n, workers):
    device = utils.get_device()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch = trial.suggest_categorical("batch", [16, 32, 48])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        opt_name = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
        img_w = trial.suggest_categorical("img_w", [384, 512, 640])

        tl, vl, tr = _loaders(batch, img_w, train_n, val_n, workers)
        model = build_model(model_name, num_classes=4, dropout=dropout).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights(tr).to(device))
        if opt_name == "adamw":
            optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif opt_name == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_f1 = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y in tl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            model.eval()
            t_all, p_all = [], []
            with torch.no_grad():
                for x, y in vl:
                    x = x.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                        prob = torch.sigmoid(model(x))
                    t_all.append(y.numpy())
                    p_all.append(prob.float().cpu().numpy())
            t, p = np.concatenate(t_all), np.concatenate(p_all)
            f1 = f1_score(t, (p >= 0.5).astype(int), average="macro", zero_division=0)
            best_f1 = max(best_f1, f1)
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_f1

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["custom_cnn", "resnet50", "efficientnet_b0"])
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train_n", type=int, default=2500)
    ap.add_argument("--val_n", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    global _TRAIN_DF, _VAL_DF, _ANN
    utils.seed_everything()
    config.ensure_dirs()
    _TRAIN_DF, _VAL_DF, _ANN = make_splits()
    print(f"[tune {args.model}] {args.trials} trials x {args.epochs} epochs "
          f"(train_n={args.train_n}, val_n={args.val_n})", flush=True)

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=f"{args.model}_tune")
    study.optimize(make_objective(args.model, args.epochs, args.train_n, args.val_n, args.workers),
                   n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest macroF1={study.best_value:.4f}")
    print("Best params:", study.best_params)

    # persist results
    df = study.trials_dataframe()
    df.to_csv(config.REPORTS_DIR / f"{args.model}_optuna_trials.csv", index=False)
    out = {"model": args.model, "best_value": study.best_value, "best_params": study.best_params,
           "n_trials": len(study.trials),
           "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}
    with open(config.REPORTS_DIR / f"{args.model}_optuna_best.json", "w") as f:
        json.dump(out, f, indent=2)

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        plot_optimization_history(study)
        plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent / "classification" / "06_optuna_history.png", bbox_inches="tight"); plt.close()
        plot_param_importances(study)
        plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent / "classification" / "07_optuna_importance.png", bbox_inches="tight"); plt.close()
    except Exception as e:
        print("viz skipped:", e)

    print(f"[tune {args.model}] DONE -> {args.model}_optuna_best.json", flush=True)


if __name__ == "__main__":
    main()
