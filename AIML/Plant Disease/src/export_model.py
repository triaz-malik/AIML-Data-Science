"""Export the trained PyTorch model to ONNX (framework-neutral artifact).

ONNX is the honest cross-framework deliverable: it loads in onnxruntime, TF
(via onnx-tf), TensorRT, mobile, etc. — far more portable than a CPU-trained
.h5. Falls back gracefully if onnx isn't importable.

Run:  python -m src.export_model --ckpt outputs/models/best_model.pt
"""
from __future__ import annotations

import argparse

import torch

from . import config as C
from .models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODELS_DIR / "best_model.pt"))
    ap.add_argument("--out", default=str(C.MODELS_DIR / "best_model.onnx"))
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(payload["model_name"], payload["n_classes"],
                        dropout=payload.get("dropout"))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    try:
        torch.onnx.export(
            model, dummy, args.out,
            input_names=["leaf"], output_names=["logits"],
            dynamic_axes={"leaf": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        # write the class list alongside for downstream consumers
        import json
        (C.MODELS_DIR / "classes.json").write_text(
            json.dumps(payload["classes"], indent=2), encoding="utf-8")
        print(f"Exported ONNX -> {args.out}")
        print(f"Wrote class list -> {C.MODELS_DIR / 'classes.json'}")
    except Exception as e:
        print(f"ONNX export skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
