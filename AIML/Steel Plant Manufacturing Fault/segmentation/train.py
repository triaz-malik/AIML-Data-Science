"""Train the U-Net defect segmenter.

Usage:
    python -m segmentation.train --epochs 30 --batch 16
    python -m segmentation.train --epochs 1 --subset 400 --workers 0   # smoke

Trains on defective images only (masks are otherwise all-zero), validates on a
mixed split. Selection metric: mean validation Dice over the 4 defect channels.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, utils
from .data import SegDataset, make_splits
from .unet import UNet, DiceBCELoss, dice_per_class


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, dices = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        losses.append(loss.item())
        dices.append(dice_per_class(logits.float(), y).cpu().numpy())
    return float(np.mean(losses)), np.mean(dices, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_h", type=int, default=256)
    ap.add_argument("--img_w", type=int, default=512)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()

    utils.seed_everything()
    config.ensure_dirs()
    device = utils.get_device()
    print(f"[unet] device={device} img={args.img_h}x{args.img_w} base={args.base} batch={args.batch}")

    train_df, val_df, ann = make_splits(defective_only=True)
    if args.subset:
        train_df = train_df.sample(args.subset, random_state=config.SEED).reset_index(drop=True)
        val_df = val_df.sample(max(args.subset // 4, 40), random_state=config.SEED).reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} (defective-only train)")

    size = (args.img_h, args.img_w)
    tl = DataLoader(SegDataset(train_df, ann, img_size=size, train=True), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0, drop_last=True)
    vl = DataLoader(SegDataset(val_df, ann, img_size=size, train=False), batch_size=args.batch,
                    shuffle=False, num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0)

    model = UNet(in_ch=3, out_ch=4, base=args.base).to(device)
    criterion = DiceBCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best, best_state, no_improve = [], -1.0, None, 0
    ckpt = config.MODELS_DIR / "unet_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tr = time.time(), []
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            tr.append(loss.item())

        val_loss, dice = evaluate(model, vl, device, criterion)
        mean_dice = float(np.mean(dice))
        sched.step(mean_dice)
        dstr = ", ".join(f"{config.CLASS_NAMES[i+1]}:{dice[i]:.3f}" for i in range(4))
        rec = {"epoch": epoch, "train_loss": float(np.mean(tr)), "val_loss": val_loss,
               "mean_dice": mean_dice, "per_class_dice": [float(d) for d in dice],
               "lr": optim.param_groups[0]["lr"], "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"  e{epoch:02d} {rec['secs']:.0f}s train_loss={rec['train_loss']:.3f} "
              f"val_loss={val_loss:.3f} meanDice={mean_dice:.3f} [{dstr}]", flush=True)

        if mean_dice > best:
            best, best_state, no_improve = mean_dice, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop @ epoch {epoch} (best meanDice={best:.3f})")
                break

    torch.save({"model": "unet", "state_dict": best_state, "best_mean_dice": best,
                "img_size": size, "base": args.base}, ckpt)
    with open(config.REPORTS_DIR / "unet_history.json", "w") as f:
        json.dump({"args": vars(args), "best_mean_dice": best, "history": history}, f, indent=2)
    print(f"[unet] DONE best_mean_dice={best:.4f} -> {ckpt.name}", flush=True)


if __name__ == "__main__":
    main()
