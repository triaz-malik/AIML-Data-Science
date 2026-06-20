"""Assemble notebooks/03_classification.ipynb — Phase 3 results & comparison.

Run AFTER training has produced reports/<model>_history.json and *_val_preds.npz.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

nb = new_notebook()
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md("""# Phase 3 — Multi-label Classification: Results
**AI-Powered Steel Surface Defect Detection**

Three models trained on the 5080 to predict the 4 defect classes as independent
sigmoid outputs (all-zero ⇒ no defect):

- **Custom CNN** — 4-block VGG-style baseline (from scratch)
- **ResNet50** — ImageNet transfer learning
- **EfficientNetB0** — ImageNet transfer learning

Loss: class-weighted `BCEWithLogitsLoss` (pos_weight = neg/pos per class) to
counter the heavy imbalance found in EDA. Selection metric: **validation macro-F1**.
""")

code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, classification_report

from src import config
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
config.ensure_dirs()
MODELS = ["custom_cnn", "resnet50", "efficientnet_b0"]
NAMES = {"custom_cnn": "Custom CNN", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNetB0"}
CLS = [config.CLASS_NAMES[c] for c in config.DEFECT_CLASSES]

hist = {m: json.load(open(config.REPORTS_DIR/f"{m}_history.json")) for m in MODELS}
preds = {m: np.load(config.REPORTS_DIR/f"{m}_val_preds.npz") for m in MODELS}
print("Loaded:", ", ".join(MODELS))
""")

md("## 1. Training curves")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for m in MODELS:
    h = pd.DataFrame(hist[m]["history"])
    axes[0].plot(h.epoch, h.val_loss, marker="o", ms=3, label=NAMES[m])
    axes[1].plot(h.epoch, h.macro_f1, marker="o", ms=3, label=NAMES[m])
axes[0].set_title("Validation loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].set_title("Validation macro-F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent/"classification"/"01_training_curves.png", bbox_inches="tight"); plt.show()
""")

md("## 2. Model comparison — best macro-F1 & mean AUC")

code("""rows = []
for m in MODELS:
    best = max(hist[m]["history"], key=lambda r: r["macro_f1"])
    rows.append({"model": NAMES[m], "best_macro_f1": round(best["macro_f1"],4),
                 "mean_auc": round(best["mean_auc"],4),
                 **{f"F1_{CLS[i]}": round(best["per_class_f1"][i],3) for i in range(4)},
                 **{f"AUC_{CLS[i]}": round(best["per_class_auc"][i],3) for i in range(4)},
                 "epochs_run": len(hist[m]["history"])})
comp = pd.DataFrame(rows).sort_values("best_macro_f1", ascending=False)
comp.to_csv(config.REPORTS_DIR/"classification_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=comp, x="model", y="best_macro_f1", ax=axes[0], palette="crest")
axes[0].set_title("Best validation macro-F1"); axes[0].set_ylim(0, 1)
for i, v in enumerate(comp.best_macro_f1): axes[0].text(i, v+0.01, f"{v:.3f}", ha="center")
sns.barplot(data=comp, x="model", y="mean_auc", ax=axes[1], palette="flare")
axes[1].set_title("Mean ROC-AUC"); axes[1].set_ylim(0, 1)
for i, v in enumerate(comp.mean_auc): axes[1].text(i, v+0.01, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent/"classification"/"02_model_comparison.png", bbox_inches="tight"); plt.show()
comp
""")

md("## 3. Per-class F1 — where each model wins")

code("""f1mat = pd.DataFrame({NAMES[m]: max(hist[m]["history"], key=lambda r: r["macro_f1"])["per_class_f1"]
                      for m in MODELS}, index=CLS)
plt.figure(figsize=(8,4))
sns.heatmap(f1mat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Per-class validation F1"); plt.ylabel("defect class")
plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent/"classification"/"03_per_class_f1.png", bbox_inches="tight"); plt.show()
""")

md("## 4. Best model — per-class confusion matrices & ROC")

code("""best_model = comp.iloc[0]["model"]
best_key = [k for k,v in NAMES.items() if v==best_model][0]
t, p = preds[best_key]["targets"], preds[best_key]["probs"]
pred = (p >= 0.5).astype(int)
print("Best model:", best_model)
print(classification_report(t, pred, target_names=CLS, zero_division=0))

mcm = multilabel_confusion_matrix(t, pred)
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for i, cls in enumerate(CLS):
    sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0,i],
                xticklabels=["pred-", "pred+"], yticklabels=["true-", "true+"])
    axes[0,i].set_title(f"{cls} confusion")
    fpr, tpr, _ = roc_curve(t[:,i], p[:,i])
    axes[1,i].plot(fpr, tpr); axes[1,i].plot([0,1],[0,1],"--",c="gray")
    axes[1,i].set_title(f"{cls} ROC"); axes[1,i].set_xlabel("FPR"); axes[1,i].set_ylabel("TPR")
plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent/"classification"/"04_best_confusion_roc.png", bbox_inches="tight"); plt.show()
""")

md("## 5. Qualitative — sample predictions from the best model")

code("""import cv2
from classification.models import build_model
from classification.data import make_splits, build_transforms

ckpt = torch.load(config.MODELS_DIR/f"{best_key}_best.pth", map_location="cpu", weights_only=False)
model = build_model(best_key, num_classes=4)
model.load_state_dict(ckpt["state_dict"]); model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
H, W = ckpt["img_size"]; tf = build_transforms(False)

_, val_df, _ = make_splits()
sample = val_df.sample(6, random_state=1).reset_index(drop=True)
fig, axes = plt.subplots(6, 1, figsize=(14, 13))
for ax, (_, row) in zip(axes, sample.iterrows()):
    img = cv2.cvtColor(cv2.imread(str(config.TRAIN_IMG_DIR/row.ImageId)), cv2.COLOR_BGR2RGB)
    x = tf(torch.from_numpy(cv2.resize(img,(W,H))).permute(2,0,1)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0].cpu().numpy()
    true = [CLS[i] for i in range(4) if row[f"c{i+1}"]==1] or ["NoDefect"]
    pred = [f"{CLS[i]}({prob[i]:.2f})" for i in range(4) if prob[i]>=0.5] or ["NoDefect"]
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"TRUE: {', '.join(true)}   |   PRED: {', '.join(pred)}", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(config.REPORTS_DIR.parent/"classification"/"05_sample_predictions.png", bbox_inches="tight"); plt.show()
""")

md("""## Summary

See `classification/` for all figures and `reports/classification_comparison.csv`
for the metrics table. The best model (by validation macro-F1) is used as the
defect **screening** stage; Phase 5 segmentation then localises the defect and
Phase 6 (Grad-CAM) explains the classifier's decision.

Next → **Phase 4: hyperparameter tuning with Optuna** on the winning architecture.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_classification.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out, "with", len(cells), "cells")
