"""Evaluate a trained checkpoint on the held-out TEST set.

Produces (outputs/figures/ and outputs/reports/):
  cm_<model>.png            confusion matrix (default + recall-tuned threshold)
  roc_<model>.png           ROC curve (with AUC)
  pr_<model>.png            precision-recall curve
  test_metrics_<model>.json full KPI dict at both thresholds
  misclassified_<model>.csv list of errors for error analysis

Run:
  python -m src.evaluate --model efficientnet_b0
  python -m src.evaluate --model efficientnet_b0 --ckpt outputs/checkpoints/efficientnet_b0_best.pt
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_curve, roc_curve)

from . import config as C
from .config import TrainConfig
from .dataset import ChestXrayDataset
from .metrics import best_threshold_for_recall, compute_metrics
from .models import build_model
from .utils import get_device, set_seed
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1)[:, C.POSITIVE_IDX].cpu().numpy()
        probs.append(p)
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(probs)


def plot_confusion(y_true, y_pred, title, fname):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=C.CLASSES, cmap="Blues",
        colorbar=False)
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / fname, dpi=120)
    plt.close()


def plot_roc(y_true, y_prob, model_name, auc):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#C44E52")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {model_name}"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"roc_{model_name}.png", dpi=120)
    plt.close()


def plot_pr(y_true, y_prob, model_name):
    prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=C.POSITIVE_IDX)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, color="#4C72B0")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {model_name}")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"pr_{model_name}.png", dpi=120)
    plt.close()


def evaluate(model_name: str, ckpt_path: str | None, min_precision: float):
    set_seed(C.SEED)
    device = get_device()
    ckpt_path = ckpt_path or str(C.CKPT_DIR / f"{model_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = TrainConfig(**ckpt["cfg"]) if "cfg" in ckpt else TrainConfig(model=model_name)
    model = build_model(model_name, dropout=cfg.dropout, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = ChestXrayDataset(C.MANIFEST_DIR / "test.csv", cfg.img_size, train=False)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    y_true, y_prob = predict(model, loader, device)

    # default 0.5 threshold
    m_default = compute_metrics(y_true, y_prob, 0.5)
    # recall-optimised threshold (hospital priority), with a precision floor
    t_recall = best_threshold_for_recall(y_true, y_prob, min_precision=min_precision)
    m_recall = compute_metrics(y_true, y_prob, t_recall)

    print(f"\n=== TEST RESULTS — {model_name} ===")
    for name, m in [("threshold=0.50", m_default),
                    (f"threshold={t_recall:.2f} (recall-tuned, prec>={min_precision})",
                     m_recall)]:
        print(f"\n[{name}]")
        print(f"  Recall (KPI) : {m['recall']:.4f}")
        print(f"  Precision    : {m['precision']:.4f}")
        print(f"  F1           : {m['f1']:.4f}")
        print(f"  AUC          : {m['auc']:.4f}")
        print(f"  Accuracy     : {m['accuracy']:.4f}")
        print(f"  FN rate      : {m['false_negative_rate']:.4f}  "
              f"(TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']})")

    print("\n" + classification_report(
        y_true, (y_prob >= 0.5).astype(int), target_names=C.CLASSES,
        zero_division=0))

    # figures
    plot_confusion(y_true, (y_prob >= 0.5).astype(int),
                   f"{model_name} (thr=0.50)", f"cm_{model_name}_default.png")
    plot_confusion(y_true, (y_prob >= t_recall).astype(int),
                   f"{model_name} (thr={t_recall:.2f}, recall-tuned)",
                   f"cm_{model_name}_recall.png")
    plot_roc(y_true, y_prob, model_name, m_default["auc"])
    plot_pr(y_true, y_prob, model_name)

    # error analysis: list misclassified samples at default threshold
    df = pd.read_csv(C.MANIFEST_DIR / "test.csv").reset_index(drop=True)
    df["prob_pneumonia"] = y_prob
    df["pred"] = (y_prob >= 0.5).astype(int)
    errors = df[df["pred"] != df["label_idx"]].copy()
    errors["error_type"] = np.where(errors["label_idx"] == C.POSITIVE_IDX,
                                    "FALSE_NEGATIVE (missed pneumonia)",
                                    "FALSE_POSITIVE")
    errors.sort_values("prob_pneumonia").to_csv(
        C.REPORT_DIR / f"misclassified_{model_name}.csv", index=False)

    out = {"model": model_name, "checkpoint": ckpt_path,
           "default_0.5": m_default,
           "recall_tuned": {**m_recall, "threshold": t_recall}}
    (C.REPORT_DIR / f"test_metrics_{model_name}.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nSaved metrics, confusion/ROC/PR figures, and error list for "
          f"{len(errors)} misclassified images.")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["custom_cnn", "resnet50", "efficientnet_b0"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--min-precision", type=float, default=0.80,
                   help="precision floor when choosing the recall-tuned threshold")
    a = p.parse_args()
    evaluate(a.model, a.ckpt, a.min_precision)
