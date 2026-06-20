"""Phases 4-6 - Train a model (cnn | resnet50 | efficientnet_b0).

Usage:
    python src/train.py --model resnet50 --epochs 15 --batch-size 32 --lr 1e-4
    python src/train.py --model all          # train all three sequentially

Saves:
    models/<name>.pth                 best weights
    reports/figures/history_<name>.png training curves
    reports/<name>_history.json        per-epoch metrics
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEVICE, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, set_seed
from data import build_loaders
from engine import train_model
from models import MODEL_NAMES, build_model


def plot_history(history, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history.train_loss) + 1)
    axes[0].plot(epochs, history.train_loss, "-o", label="train")
    axes[0].plot(epochs, history.val_loss, "-o", label="val")
    axes[0].set_title(f"{name} — loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, history.train_acc, "-o", label="train")
    axes[1].plot(epochs, history.val_acc, "-o", label="val")
    axes[1].set_title(f"{name} — accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"history_{name}.png", dpi=120)
    plt.close(fig)


def train_one(name: str, args) -> float:
    print(f"\n=== Training {name} on {DEVICE} ===")
    img_size = 160 if name == "cnn" else args.img_size
    train_loader, val_loader = build_loaders(
        img_size=img_size, batch_size=args.batch_size, num_workers=args.num_workers, augment=True
    )
    model = build_model(name, num_classes=2, dropout=args.dropout, pretrained=not args.no_pretrained)
    lr = args.lr if name != "cnn" else max(args.lr, 1e-3)   # baseline trains from scratch -> higher lr
    _, history, best_val_acc = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=lr, weight_decay=args.weight_decay, patience=args.patience,
    )
    tag = getattr(args, "tag", "") or ""
    out = MODELS_DIR / f"{name}{tag}.pth"
    import torch

    torch.save({"model_name": name, "state_dict": model.state_dict(),
                "img_size": img_size, "best_val_acc": best_val_acc}, out)
    plot_history(history, f"{name}{tag}")
    (REPORTS_DIR / f"{name}{tag}_history.json").write_text(
        json.dumps({"train_loss": history.train_loss, "val_loss": history.val_loss,
                    "train_acc": history.train_acc, "val_acc": history.val_acc,
                    "best_val_acc": best_val_acc}, indent=2), encoding="utf-8")
    print(f"  saved -> {out}  (best val_acc={best_val_acc:.4f})")
    return best_val_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50",
                    help="cnn | resnet50 | efficientnet_b0 | all")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for saved model/report filenames, e.g. _tuned")
    args = ap.parse_args()
    set_seed()

    names = MODEL_NAMES if args.model == "all" else [args.model]
    results = {n: train_one(n, args) for n in names}
    print("\n=== Summary (best val accuracy) ===")
    for n, acc in results.items():
        print(f"  {n:18s} {acc:.4f}")


if __name__ == "__main__":
    main()
