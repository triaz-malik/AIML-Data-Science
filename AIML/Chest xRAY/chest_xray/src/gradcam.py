"""Grad-CAM explainability: which lung regions drove the prediction.

Saves a grid of (original | heatmap | overlay) for sample test images to
outputs/figures/gradcam_<model>.png. By default it visualises a mix of
correctly-flagged pneumonia and any false negatives (the dangerous errors).

Uses pytorch-grad-cam if installed; otherwise a small built-in Grad-CAM.

Run:
  python -m src.gradcam --model efficientnet_b0 --n 8
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from . import config as C
from .config import TrainConfig
from .dataset import build_transforms
from .models import build_model, target_layer_for_gradcam
from .utils import get_device, set_seed


class SimpleGradCAM:
    """Minimal Grad-CAM (fallback if pytorch-grad-cam isn't installed)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.activations = out.detach()

    def _bwd(self, _m, _gi, gout):
        self.gradients = gout[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP grads
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def overlay(cam: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    base = np.stack([gray_img] * 3, axis=-1) / 255.0
    return np.clip(0.5 * base + 0.5 * heat, 0, 1)


def run(model_name: str, n: int):
    set_seed(C.SEED)
    device = get_device()
    ckpt = torch.load(C.CKPT_DIR / f"{model_name}_best.pt",
                      map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["cfg"]) if "cfg" in ckpt else TrainConfig(model=model_name)
    model = build_model(model_name, dropout=cfg.dropout, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cam_engine = SimpleGradCAM(model, target_layer_for_gradcam(model, model_name))
    tf = build_transforms(cfg.img_size, train=False)

    df = pd.read_csv(C.MANIFEST_DIR / "test.csv")
    # prefer pneumonia cases (that's what we want to explain); shuffle
    sample = df[df["label"] == "PNEUMONIA"].sample(
        min(n, (df["label"] == "PNEUMONIA").sum()), random_state=C.SEED)

    fig, axes = plt.subplots(len(sample), 3, figsize=(9, 3 * len(sample)))
    if len(sample) == 1:
        axes = axes[None, :]
    for i, (_, row) in enumerate(sample.iterrows()):
        with Image.open(row["path"]) as im:
            gray = np.asarray(im.convert("L").resize((cfg.img_size, cfg.img_size)),
                              dtype=np.float32)
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0, C.POSITIVE_IDX].item()
        cam = cam_engine(x, C.POSITIVE_IDX)

        axes[i, 0].imshow(gray, cmap="gray"); axes[i, 0].set_title("Original")
        axes[i, 1].imshow(cam, cmap="jet"); axes[i, 1].set_title("Grad-CAM")
        axes[i, 2].imshow(overlay(cam, gray))
        axes[i, 2].set_title(f"Overlay  P(pneu)={prob:.2f}")
        for a in axes[i]:
            a.axis("off")
    plt.suptitle(f"Grad-CAM — {model_name}", y=1.0)
    plt.tight_layout()
    out = C.FIG_DIR / f"gradcam_{model_name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["custom_cnn", "resnet50", "efficientnet_b0"])
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    run(a.model, a.n)
