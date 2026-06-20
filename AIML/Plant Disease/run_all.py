"""Orchestrate the post-training phases end-to-end:

  Phase 7  tune    -> best_model.pt  (+ best_hparams.json, optuna plots)
  Phase 9  evaluate-> confusion matrix, ROC, ERROR_ANALYSIS.md, test_metrics.json
  Phase 8  gradcam -> gradcam.png
  Phase 10 predict -> test_predictions.csv + grid
           export  -> best_model.onnx
  Phase 11-12 report -> business + comparison + final summary

Run:  python run_all.py
Assumes Phases 1-6 already ran (models trained). Idempotent; safe to re-run.
"""
from __future__ import annotations

import runpy
import sys
import time


def step(mod: str, argv: list[str]):
    print(f"\n{'='*70}\n>>> {mod}  {' '.join(argv)}\n{'='*70}", flush=True)
    sys.argv = [mod] + argv
    t0 = time.time()
    runpy.run_module(mod, run_name="__main__")
    print(f"<<< {mod} done in {time.time()-t0:.0f}s", flush=True)


def main():
    ckpt = "outputs/models/best_model.pt"
    step("src.tune", ["--model", "efficientnet_b0", "--trials", "15"])
    step("src.evaluate", ["--ckpt", ckpt])
    step("src.gradcam", ["--ckpt", ckpt, "--n", "8"])
    step("src.predict", ["--ckpt", ckpt])
    step("src.export_model", ["--ckpt", ckpt])
    step("src.report", [])
    print("\nALL PHASES COMPLETE.")


if __name__ == "__main__":
    main()
