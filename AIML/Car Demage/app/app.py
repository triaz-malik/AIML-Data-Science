"""Streamlit app — Vehicle Damage Assessment.

Upload a car photo → get a damage / whole verdict + Grad-CAM explanation.

Run:
    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DEVICE, MODELS_DIR  # noqa: E402
from data import build_transforms, denormalize  # noqa: E402
from predict import load_model, predict_image  # noqa: E402

st.set_page_config(page_title="Vehicle Damage Assessment", page_icon="🚗", layout="centered")


@st.cache_resource
def get_model(name: str):
    return load_model(name)


def gradcam_overlay(model, img: Image.Image, img_size: int, model_name: str):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    from models import gradcam_target_layer

    tensor = build_transforms(img_size, train=False)(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model, model_name)])
    grayscale = cam(input_tensor=tensor)[0]
    rgb = denormalize(tensor[0]).permute(1, 2, 0).numpy()
    return show_cam_on_image(rgb, grayscale, use_rgb=True)


st.title("🚗 Vehicle Damage Assessment")
st.caption("Upload a car photo — the model classifies it as **damaged** or **whole** and shows "
           "where it looked (Grad-CAM).")

available = sorted(p.stem for p in MODELS_DIR.glob("*.pth"))
if not available:
    st.error("No trained models found in models/. Run `python src/train.py --model all` first.")
    st.stop()

default_idx = available.index("resnet50") if "resnet50" in available else 0
model_name = st.sidebar.selectbox("Model", available, index=default_idx)
show_cam = st.sidebar.checkbox("Show Grad-CAM explanation", value=True)

uploaded = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    model, img_size = get_model(model_name)
    res = predict_image(model, img, img_size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input", use_container_width=True)
    with col2:
        if show_cam:
            with st.spinner("Computing Grad-CAM..."):
                overlay = gradcam_overlay(model, img, img_size, model_name)
            st.image(overlay, caption="Grad-CAM (warm = drives prediction)", use_container_width=True)

    verdict = res["label"].upper()
    if res["label"] == "damage":
        st.error(f"### Verdict: DAMAGED  ({res['confidence']:.1%} confidence)")
    else:
        st.success(f"### Verdict: WHOLE / UNDAMAGED  ({res['confidence']:.1%} confidence)")

    st.progress(res["p_damage"], text=f"P(damage) = {res['p_damage']:.1%}")
    st.caption(f"Model: {model_name}  ·  Device: {DEVICE}")
else:
    st.info("👆 Upload an image to get a prediction.")
