"""Single-image inference helper (used by the CLI and the Streamlit app).

Usage:
    python src/predict.py --model resnet50 --image path/to/car.jpg
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from config import DEVICE, IDX_TO_CLASS, MODELS_DIR
from data import build_transforms
from models import build_model


def load_model(model_name: str = "resnet50"):
    ckpt = torch.load(MODELS_DIR / f"{model_name}.pth", map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt.get("img_size", 224)


@torch.no_grad()
def predict_image(model, img: Image.Image, img_size: int = 224) -> dict:
    tensor = build_transforms(img_size, train=False)(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    prob = torch.softmax(model(tensor).float(), 1)[0].cpu()
    pred = int(prob.argmax())
    return {
        "label": IDX_TO_CLASS[pred],
        "confidence": float(prob[pred]),
        "p_damage": float(prob[1]),
        "p_whole": float(prob[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    model, img_size = load_model(args.model)
    res = predict_image(model, Image.open(args.image), img_size)
    print(f"Prediction: {res['label'].upper()}  (confidence {res['confidence']:.1%})")
    print(f"  P(damage)={res['p_damage']:.3f}  P(whole)={res['p_whole']:.3f}")


if __name__ == "__main__":
    main()
