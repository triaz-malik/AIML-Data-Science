"""Phase 8 — Grad-CAM explainability.

Self-contained Grad-CAM (no extra deps): hooks the last conv block, weights the
activation maps by the gradient of the predicted class, and overlays the
heatmap on the leaf so a farmer can literally see *where* the model looked.

Run:  python -m src.gradcam --ckpt outputs/models/best_model.pt --n 8
"""
from __future__ import annotations

import argparse
import pickle

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config as C
from .data import eval_transform
from .models import build_model, gradcam_target_layer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # GAP over spatial
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, logits.softmax(1)[0, class_idx].item()


def overlay(pil_img, cam):
    img = np.asarray(pil_img.resize((C.IMG_SIZE, C.IMG_SIZE))).astype(np.float32) / 255
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1] / 255
    return np.clip(0.5 * img + 0.5 * heat, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODELS_DIR / "best_model.pt"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location=C.DEVICE, weights_only=False)
    model = build_model(payload["model_name"], payload["n_classes"], dropout=payload.get("dropout"))
    model.load_state_dict(payload["state_dict"]); model.to(C.DEVICE)
    classes = payload["classes"]
    cam_engine = GradCAM(model, gradcam_target_layer(model, payload["model_name"]))
    tf = eval_transform()

    # pick a spread of test images across classes from the split
    import pandas as pd
    df = pd.read_csv(C.SPLITS_DIR / "split.csv")
    test = df[df.split == "test"]
    picks = (test.groupby("label", group_keys=False)
                 .sample(1, random_state=C.SEED)
                 .sample(args.n, random_state=C.SEED))

    rows = len(picks)
    fig, axes = plt.subplots(rows, 2, figsize=(6, rows * 3))
    if rows == 1:
        axes = axes[None, :]
    for r, (_, rec) in enumerate(picks.iterrows()):
        pil = Image.open(rec["path"]).convert("RGB")
        x = tf(pil).unsqueeze(0).to(C.DEVICE)
        cam, idx, conf = cam_engine(x, None)
        axes[r, 0].imshow(pil.resize((C.IMG_SIZE, C.IMG_SIZE))); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"true: {rec['label'].split('___')[-1]}", fontsize=8)
        axes[r, 1].imshow(overlay(pil, cam)); axes[r, 1].axis("off")
        axes[r, 1].set_title(f"pred: {classes[idx].split('___')[-1]} ({conf:.0%})",
                             fontsize=8, color="darkgreen" if classes[idx] == rec["label"] else "crimson")
    fig.suptitle("Grad-CAM: where the model looks (red = high attention)", y=1.0)
    out = C.PLOTS_DIR / "gradcam.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
