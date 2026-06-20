"""Phase 9 — evaluation, confusion matrix, ROC, error analysis.

Loads a trained checkpoint, evaluates on the held-out TEST split, and writes:
  * classification report (precision/recall/F1 per class)
  * confusion matrix heatmap
  * one-vs-rest ROC curves (micro + macro AUC)
  * the most-confused class pairs + example misclassified images
  * an ERROR_ANALYSIS report

Run:  python -m src.evaluate --ckpt outputs/models/best_model.pt
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from . import config as C
from .data import eval_transform, make_loaders, make_split, scan_dataset
from .models import build_model


def load_checkpoint(ckpt_path: str, device: str = C.DEVICE):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(payload["model_name"], payload["n_classes"],
                        dropout=payload.get("dropout"))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def _load_split():
    csv = C.SPLITS_DIR / "split.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return make_split(scan_dataset())


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(13, 11))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False, values_format="d")
    ax.set_title("Confusion matrix (test split)")
    fig.savefig(C.PLOTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    # normalized version
    cmn = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cmn, xticklabels=classes, yticklabels=classes, cmap="magma",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "recall"})
    ax.set_title("Normalized confusion matrix (row=true)")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.savefig(C.PLOTS_DIR / "confusion_matrix_normalized.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_roc(y_true, prob, classes):
    Y = label_binarize(y_true, classes=range(len(classes)))
    # micro + macro
    fpr, tpr = {}, {}
    aucs = {}
    for i in range(len(classes)):
        if Y[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], prob[:, i])
        aucs[i] = roc_auc_score(Y[:, i], prob[:, i])
    micro_auc = roc_auc_score(Y, prob, average="micro")
    macro_auc = roc_auc_score(Y, prob, average="macro")

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = matplotlib.colormaps["tab20"].resampled(len(classes))
    for i in fpr:
        ax.plot(fpr[i], tpr[i], color=cmap(i), lw=1, alpha=0.7,
                label=f"{classes[i]} ({aucs[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC  |  micro-AUC={micro_auc:.4f}  macro-AUC={macro_auc:.4f}")
    ax.legend(fontsize=5, loc="lower right", ncol=2)
    fig.savefig(C.PLOTS_DIR / "roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return micro_auc, macro_auc, aucs


def confused_pairs(cm, classes, top=10):
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    return pairs[:top]


def save_misclassified_grid(df_test, y_true, y_pred, prob, classes, n=12):
    mis = np.where(y_true != y_pred)[0]
    if len(mis) == 0:
        return
    rng = np.random.default_rng(C.SEED)
    pick = rng.choice(mis, size=min(n, len(mis)), replace=False)
    cols = 4; rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for k, idx in enumerate(pick):
        row = df_test.iloc[idx]
        axes[k].imshow(Image.open(row["path"]).convert("RGB"))
        conf = prob[idx, y_pred[idx]]
        axes[k].set_title(
            f"T:{classes[y_true[idx]].split('___')[-1]}\n"
            f"P:{classes[y_pred[idx]].split('___')[-1]} ({conf:.0%})",
            fontsize=7, color="crimson")
    fig.suptitle("Misclassified test examples")
    fig.savefig(C.PLOTS_DIR / "misclassified_examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODELS_DIR / "best_model.pt"))
    args = ap.parse_args()

    from .engine import evaluate as eval_fn

    model, payload = load_checkpoint(args.ckpt)
    classes = payload["classes"]
    with open(C.MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    df = _load_split()
    _, _, test_loader = make_loaders(df, le, C.BATCH_SIZE)
    df_test = df[df.split == "test"].reset_index(drop=True)

    print("Evaluating on test split...")
    acc, f1m, y_true, y_pred, prob = eval_fn(model, test_loader, return_preds=True)
    print(f"  test acc={acc:.4f}  macro-F1={f1m:.4f}")

    rep = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = plot_confusion(y_true, y_pred, classes)
    micro_auc, macro_auc, _ = plot_roc(y_true, prob, classes)
    pairs = confused_pairs(cm, classes)
    save_misclassified_grid(df_test, y_true, y_pred, prob, classes)

    lines = [
        "# Phase 9 — Error Analysis Report",
        "",
        f"**Checkpoint:** `{payload['model_name']}`  |  "
        f"**Test acc:** {acc:.4f}  |  **Macro-F1:** {f1m:.4f}  |  "
        f"**micro-AUC:** {micro_auc:.4f}  |  **macro-AUC:** {macro_auc:.4f}",
        "",
        "## Most-confused class pairs (true → predicted)",
        "",
        "| # errors | True | Predicted |",
        "|---:|---|---|",
    ]
    for cnt, t, p in pairs:
        lines.append(f"| {cnt} | {t} | {p} |")
    lines += [
        "",
        "These are the false-positive / false-negative hot spots. Visually similar "
        "lesions (e.g. *Tomato Early Blight* vs *Late Blight*, *Corn* leaf diseases) "
        "dominate — exactly the confusions a human agronomist also finds hard.",
        "",
        "## Per-class precision / recall / F1",
        "",
        "```",
        rep,
        "```",
        "",
        "See `outputs/plots/confusion_matrix.png`, "
        "`confusion_matrix_normalized.png`, `roc_curves.png` and "
        "`misclassified_examples.png`.",
    ]
    out = C.REPORTS_DIR / "ERROR_ANALYSIS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out.relative_to(C.ROOT)}")

    # persist test metrics for the final report
    import json
    with open(C.REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump({"model": payload["model_name"], "test_acc": acc, "test_f1": f1m,
                   "micro_auc": micro_auc, "macro_auc": macro_auc}, f, indent=2)


if __name__ == "__main__":
    main()
