"""Run the full pipeline end-to-end (Phases 1-10).

Usage:
    python src/run_all.py            # everything
    python src/run_all.py --fast     # skip the slow tuning search
"""
from __future__ import annotations

import argparse
import importlib
import time

PHASES = [
    ("phase1_eda", "EDA"),
    ("phase2_timeseries", "Time-Series Analysis"),
    ("phase3_features", "Feature Engineering"),
    ("phase4_baselines", "Baseline Models"),
    ("phase5_deep_learning", "Deep Learning (LSTM/GRU/Bi-LSTM)"),
    ("phase6_tuning", "Hyperparameter Tuning"),
    ("phase7_multistep", "Multi-Step Forecasting"),
    ("phase8_evaluation", "Evaluation"),
    ("phase9_explainability", "Explainability (SHAP)"),
    ("phase10_report", "Final Report"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip Phase 6 hyperparameter search")
    args = ap.parse_args()

    for mod_name, label in PHASES:
        if args.fast and mod_name == "phase6_tuning":
            print(f"\n=== Skipping {label} (--fast) ===")
            continue
        print(f"\n{'='*70}\n=== {label} ===\n{'='*70}")
        t0 = time.time()
        mod = importlib.import_module(mod_name)
        mod.main()
        print(f"--- {label} done in {time.time()-t0:.1f}s ---")

    print("\nPipeline complete. See outputs/reports/final_report.md")


if __name__ == "__main__":
    main()
