"""Train a multi-label steel-defect classifier.

Usage:
    python -m classification.train --model resnet50 --epochs 30
    python -m classification.train --model custom_cnn --epochs 1 --subset 500   # smoke

Artifacts:
    models/<model>_best.pth          best checkpoint (by val macro-F1)
    reports/<model>_history.json     per-epoch metrics
    reports/<model>_val_preds.npz    val targets + probabilities (for the notebook)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src import config, utils
from .data import ClsDataset, build_transforms, make_splits, pos_weights, LABEL_COLS
from .models import build_model


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, all_t, all_p = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            losses.append(loss.item())
            all_t.append(y.cpu().numpy())
            all_p.append(torch.sigmoid(logits).float().cpu().numpy())
    t = np.concatenate(all_t)
    p = np.concatenate(all_p)
    pred = (p >= 0.5).astype(int)
    per_class_f1 = f1_score(t, pred, average=None, zero_division=0)
    macro_f1 = f1_score(t, pred, average="macro", zero_division=0)
    aucs = []
    for c in range(t.shape[1]):
        try:
            aucs.append(roc_auc_score(t[:, c], p[:, c]))
        except ValueError:
            aucs.append(float("nan"))
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "per_class_auc": [float(x) for x in aucs],
        "mean_auc": float(np.nanmean(aucs)),
    }, t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["custom_cnn", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_h", type=int, default=256)
    ap.add_argument("--img_w", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="limit dataset size for smoke tests")
    args = ap.parse_args()

    utils.seed_everything()
    config.ensure_dirs()
    device = utils.get_device()
    print(f"[{args.model}] device={device} img={args.img_h}x{args.img_w} batch={args.batch}")

    train_df, val_df, _ = make_splits()
    if args.subset:
        train_df = train_df.sample(args.subset, random_state=config.SEED).reset_index(drop=True)
        val_df = val_df.sample(max(args.subset // 4, 50), random_state=config.SEED).reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)}")

    size = (args.img_h, args.img_w)
    train_ds = ClsDataset(train_df, img_size=size, transform=build_transforms(True), clahe=args.clahe)
    val_ds = ClsDataset(val_df, img_size=size, transform=build_transforms(False), clahe=args.clahe)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(args.model, num_classes=4, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights(train_df).to(device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_f1, best_state, best_pp, no_improve = -1.0, None, None, 0
    ckpt = config.MODELS_DIR / f"{args.model}_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tr_losses = time.time(), []
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            tr_losses.append(loss.item())

        val_metrics, vt, vp = evaluate(model, val_ld, device, criterion)
        sched.step(val_metrics["macro_f1"])
        rec = {"epoch": epoch, "train_loss": float(np.mean(tr_losses)),
               "lr": optim.param_groups[0]["lr"], "secs": round(time.time() - t0, 1), **val_metrics}
        history.append(rec)
        f1s = ", ".join(f"{config.CLASS_NAMES[i+1]}:{v:.2f}" for i, v in enumerate(val_metrics["per_class_f1"]))
        print(f"  e{epoch:02d} {rec['secs']:.0f}s train_loss={rec['train_loss']:.3f} "
              f"val_loss={val_metrics['loss']:.3f} macroF1={val_metrics['macro_f1']:.3f} "
              f"meanAUC={val_metrics['mean_auc']:.3f} [{f1s}]", flush=True)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_pp = (vt, vp)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop @ epoch {epoch} (best macroF1={best_f1:.3f})")
                break

    torch.save({"model": args.model, "state_dict": best_state, "best_macro_f1": best_f1,
                "img_size": size, "label_cols": LABEL_COLS}, ckpt)
    np.savez(config.REPORTS_DIR / f"{args.model}_val_preds.npz", targets=best_pp[0], probs=best_pp[1])
    with open(config.REPORTS_DIR / f"{args.model}_history.json", "w") as f:
        json.dump({"args": vars(args), "best_macro_f1": best_f1, "history": history}, f, indent=2)
    print(f"[{args.model}] DONE best_macro_f1={best_f1:.4f} -> {ckpt.name}", flush=True)


if __name__ == "__main__":
    main()
