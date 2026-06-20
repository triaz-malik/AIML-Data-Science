"""Phase 8 - Evaluation: metrics, confusion matrix, ROC & PR curves, comparison.

Usage:
    python src/evaluate.py                 # evaluate every models/*.pth
    python src/evaluate.py --model resnet50
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CLASS_NAMES, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, set_seed
from data import build_loaders
from engine import predict
from models import build_model


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    name = ckpt["model_name"]
    model = build_model(name, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    return name, model, ckpt.get("img_size", 224)


def plot_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} — confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_{name}.png", dpi=120)
    plt.close(fig)


def evaluate_one(path, batch_size=32, num_workers=4):
    name, model, img_size = load_model(path)
    _, val_loader = build_loaders(img_size=img_size, batch_size=batch_size,
                                  num_workers=num_workers, augment=False)
    y, pred, prob = predict(model, val_loader)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }
    plot_confusion(confusion_matrix(y, pred), name)
    print(f"\n[{name}]")
    print(classification_report(y, pred, target_names=CLASS_NAMES, zero_division=0))
    return metrics, (y, prob, name)


def plot_curves(roc_data):
    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for y, prob, name in roc_data:
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 8 — ROC curves"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=120); plt.close(fig)
    # PR
    fig, ax = plt.subplots(figsize=(7, 6))
    for y, prob, name in roc_data:
        prec, rec, _ = precision_recall_curve(y, prob)
        ax.plot(rec, prec, label=f"{name} (AP={auc(rec, prec):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Phase 8 — Precision-Recall curves"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    set_seed()

    if args.model == "all":
        paths = sorted(MODELS_DIR.glob("*.pth"))
    else:
        paths = [MODELS_DIR / f"{args.model}.pth"]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No models found in models/. Train first with src/train.py.")
        return

    all_metrics, roc_data = [], []
    for p in paths:
        m, rd = evaluate_one(p, args.batch_size, args.num_workers)
        all_metrics.append(m); roc_data.append(rd)
    plot_curves(roc_data)

    # comparison table
    order = ["cnn", "resnet50", "efficientnet_b0"]
    all_metrics.sort(key=lambda m: order.index(m["model"]) if m["model"] in order else 99)
    header = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows = [[m["model"], f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
             f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['roc_auc']:.4f}"] for m in all_metrics]
    md = ["# Phase 8 — Model Comparison (held-out validation set)\n",
          "Positive class = **damage**.\n",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    best = max(all_metrics, key=lambda m: m["f1"])
    md.append(f"\n**Best model by F1: `{best['model']}` "
              f"(F1={best['f1']:.4f}, accuracy={best['accuracy']:.4f}, ROC-AUC={best['roc_auc']:.4f}).**")
    md += ["\n## Figures\n",
           "![ROC](figures/roc_curves.png)", "![PR](figures/pr_curves.png)"]
    for m in all_metrics:
        md.append(f"![confusion {m['model']}](figures/confusion_{m['model']}.png)")
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print("\n=== Comparison ===")
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(r))
    print(f"\nBest by F1: {best['model']}")
    print(f"Report -> {REPORTS_DIR / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
