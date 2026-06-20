"""Training loop with two-phase transfer learning.

Features:
  * class-weighted CrossEntropy (handles 2.69:1 imbalance)
  * phase 1: frozen backbone, train head; phase 2: unfreeze + low LR fine-tune
  * checkpoints the best model by val RECALL (the hospital KPI)
  * early stopping on the monitored metric
  * saves training-history plot + history.csv

Run examples:
  python -m src.train --model custom_cnn      --epochs 20
  python -m src.train --model resnet50        --epochs 25
  python -m src.train --model efficientnet_b0 --epochs 25 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from . import config as C
from .config import TrainConfig
from .dataset import compute_class_weights, make_loaders
from .metrics import compute_metrics
from .models import build_model, set_backbone_trainable
from .utils import count_params, get_device, set_seed


def make_optimizer(name: str, params, lr: float, wd: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'")


@torch.no_grad()
def evaluate_split(model, loader, criterion, device) -> tuple[dict, float]:
    model.eval()
    losses, probs, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        probs.append(torch.softmax(logits, 1)[:, C.POSITIVE_IDX].cpu().numpy())
        targets.append(y.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    return compute_metrics(y_true, y_prob), float(np.mean(losses))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> float:
    model.train()
    running = 0.0
    for x, y in tqdm(loader, leave=False, desc="train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            enabled=(device.type == "cuda")):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def plot_history(hist: pd.DataFrame, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    for m in ("val_recall", "val_f1", "val_auc", "val_accuracy"):
        axes[1].plot(hist["epoch"], hist[m], label=m.replace("val_", ""))
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.suptitle(f"Training history — {model_name}")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"history_{model_name}.png", dpi=120)
    plt.close()


def train(cfg: TrainConfig) -> dict:
    set_seed(C.SEED)
    device = get_device()
    print(f"Device: {device}  |  model: {cfg.model}")

    loaders = make_loaders(cfg)
    model = build_model(cfg.model, dropout=cfg.dropout).to(device)
    total, trainable = count_params(model)
    print(f"Params: {total:,} total")

    # class-weighted loss
    weight = (compute_class_weights(C.MANIFEST_DIR / "train.csv").to(device)
              if cfg.use_class_weights else None)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=cfg.label_smoothing)
    print(f"Class weights: {None if weight is None else weight.tolist()}")

    is_transfer = cfg.model.lower() != "custom_cnn"
    if is_transfer and cfg.freeze_epochs > 0:
        set_backbone_trainable(model, cfg.model, trainable=False)
    optimizer = make_optimizer(cfg.optimizer,
                               filter(lambda p: p.requires_grad, model.parameters()),
                               cfg.lr, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score, best_path = -1.0, C.CKPT_DIR / f"{cfg.model}_best.pt"
    epochs_no_improve, history = 0, []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # phase switch: unfreeze backbone and drop LR
        if is_transfer and epoch == cfg.freeze_epochs + 1:
            print(f"-- unfreezing backbone, lr -> {cfg.finetune_lr}")
            set_backbone_trainable(model, cfg.model, trainable=True)
            optimizer = make_optimizer(cfg.optimizer, model.parameters(),
                                       cfg.finetune_lr, cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2)

        tr_loss = train_one_epoch(model, loaders["train"], criterion,
                                  optimizer, device, scaler)
        val_m, val_loss = evaluate_split(model, loaders["val"], criterion, device)
        scheduler.step(val_m[cfg.monitor])

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in val_m.items()
                  if k in ("recall", "precision", "f1", "auc", "accuracy")}}
        history.append(row)
        print(f"[{epoch:02d}/{cfg.epochs}] "
              f"tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} | "
              f"recall={val_m['recall']:.4f} f1={val_m['f1']:.4f} "
              f"auc={val_m['auc']:.4f} acc={val_m['accuracy']:.4f}")

        score = val_m[cfg.monitor]
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(),
                        "cfg": asdict(cfg), "epoch": epoch,
                        "val_metrics": val_m}, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no {cfg.monitor} gain in {cfg.early_stop_patience}).")
                break

    elapsed = time.time() - t0
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(C.REPORT_DIR / f"history_{cfg.model}.csv", index=False)
    plot_history(hist_df, cfg.model)

    print(f"\nBest val {cfg.monitor}: {best_score:.4f}  ({elapsed/60:.1f} min)")
    print(f"Best checkpoint -> {best_path}")
    summary = {"model": cfg.model, "best_val_" + cfg.monitor: best_score,
               "minutes": round(elapsed / 60, 2), "checkpoint": str(best_path)}
    (C.REPORT_DIR / f"train_summary_{cfg.model}.json").write_text(
        json.dumps(summary, indent=2))
    return summary


def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train a pneumonia classifier")
    p.add_argument("--model", default=d.model,
                   choices=["custom_cnn", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--optimizer", default=d.optimizer,
                   choices=["adam", "rmsprop", "sgd"])
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--freeze-epochs", type=int, default=d.freeze_epochs)
    p.add_argument("--finetune-lr", type=float, default=d.finetune_lr)
    p.add_argument("--no-class-weights", action="store_true")
    a = p.parse_args()
    return TrainConfig(
        model=a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, optimizer=a.optimizer, dropout=a.dropout,
        num_workers=a.num_workers, freeze_epochs=a.freeze_epochs,
        finetune_lr=a.finetune_lr, use_class_weights=not a.no_class_weights)


if __name__ == "__main__":
    train(parse_args())
