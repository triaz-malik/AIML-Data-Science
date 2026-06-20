"""Phase 9 - Explainable AI: Grad-CAM heatmaps for the trained model.

Shows where the model "looks" when it predicts damage (bumper, door, dent...).

Usage:
    python src/gradcam.py --model resnet50 --n 8
    python src/gradcam.py --model resnet50 --image path/to/car.jpg
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import DEVICE, GRADCAM_DIR, IDX_TO_CLASS, MODELS_DIR, VAL_DIR, set_seed
from data import build_transforms, denormalize, list_images
from models import build_model, gradcam_target_layer


def load(model_name: str):
    ckpt = torch.load(MODELS_DIR / f"{model_name}.pth", map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt.get("img_size", 224)


def run(model_name: str, n: int, single_image: str | None):
    model, img_size = load(model_name)
    tfm = build_transforms(img_size, train=False)
    target_layer = gradcam_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    if single_image:
        samples = [(single_image, -1)]
    else:
        # pick a mix of damage and whole images from validation
        all_s = list_images(VAL_DIR)
        dmg = [s for s in all_s if IDX_TO_CLASS[s[1]] == "damage"][: n // 2 or 1]
        whole = [s for s in all_s if IDX_TO_CLASS[s[1]] == "whole"][: n - len(dmg)]
        samples = dmg + whole

    cols = min(4, len(samples))
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for idx, (path, true_label) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        tensor = tfm(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(tensor).float(), 1)[0]
        pred = int(prob.argmax())
        grayscale = cam(input_tensor=tensor)[0]            # HxW in [0,1]
        rgb = denormalize(tensor[0]).permute(1, 2, 0).numpy()
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        ax = axes[idx // cols][idx % cols]
        ax.imshow(overlay)
        title = f"pred={IDX_TO_CLASS[pred]} ({prob[pred]:.2f})"
        if true_label >= 0:
            title = f"true={IDX_TO_CLASS[true_label]}\n" + title
        ax.set_title(title, color="#C44E52" if pred == 1 else "#4C72B0")
        ax.axis("off")

    for k in range(len(samples), rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle(f"Phase 9 — Grad-CAM ({model_name}): warm regions drive the prediction",
                 fontweight="bold")
    fig.tight_layout()
    out = GRADCAM_DIR / f"gradcam_{model_name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--image", default=None, help="run on a single image path")
    args = ap.parse_args()
    set_seed()
    run(args.model, args.n, args.image)


if __name__ == "__main__":
    main()
