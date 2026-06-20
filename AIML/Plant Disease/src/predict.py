"""Inference on the loose, unlabeled images in ``test/`` (the demo images), with
disease prediction + confidence + agronomic recommendation (Phases 8/10 tie-in).

Run:  python -m src.predict --ckpt outputs/models/best_model.pt
Outputs: outputs/reports/test_predictions.csv and a visual grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from . import config as C
from .data import IMG_EXTS, eval_transform
from .models import build_model
from .recommend import recommend


def load(ckpt, device=C.DEVICE):
    p = torch.load(ckpt, map_location=device, weights_only=False)
    m = build_model(p["model_name"], p["n_classes"], dropout=p.get("dropout"))
    m.load_state_dict(p["state_dict"]); m.to(device).eval()
    return m, p["classes"]


@torch.no_grad()
def predict_image(model, classes, path, tf, device=C.DEVICE, topk=3):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        prob = model(x).softmax(1)[0].float().cpu()
    conf, idx = prob.max(0)
    top = torch.topk(prob, min(topk, len(classes)))
    topk_list = [(classes[i], float(prob[i])) for i in top.indices]
    return classes[idx], float(conf), topk_list, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODELS_DIR / "best_model.pt"))
    ap.add_argument("--dir", default=str(C.TEST_DIR))
    args = ap.parse_args()

    model, classes = load(args.ckpt)
    tf = eval_transform()
    files = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Predicting {len(files)} images from {args.dir}")

    rows = []
    for f in files:
        label, conf, topk, _ = predict_image(model, classes, f, tf)
        rows.append({
            "file": f.name,
            "prediction": label,
            "confidence": round(conf, 4),
            "top3": "; ".join(f"{c}={p:.2f}" for c, p in topk),
            "recommendation": recommend(label, conf),
        })
    out_df = pd.DataFrame(rows)
    csv = C.REPORTS_DIR / "test_predictions.csv"
    out_df.to_csv(csv, index=False)
    print(f"wrote {csv.relative_to(C.ROOT)}")

    # visual grid with prediction + confidence
    n = len(files); cols = 4; import math
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3.2))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for ax in axes:
        ax.axis("off")
    for k, f in enumerate(files):
        label, conf, _, img = predict_image(model, classes, f, tf)
        axes[k].imshow(img)
        axes[k].set_title(f"{f.name}\n{label.split('___')[0]} / "
                          f"{label.split('___')[-1]}\n{conf:.0%}", fontsize=7)
    fig.suptitle("Predictions on loose test/ images", y=1.0)
    grid = C.PLOTS_DIR / "test_predictions_grid.png"
    fig.savefig(grid, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {grid.relative_to(C.ROOT)}")

    # quick sanity: filename keyword vs prediction crop (the demo files are named)
    print("\nSample predictions:")
    print(out_df[["file", "prediction", "confidence"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
