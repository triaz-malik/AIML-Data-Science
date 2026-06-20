"""
End-to-end orchestrator for the AI-Powered Healthcare Risk Prediction System.

Runs every phase in order and writes all artifacts to ./outputs/.

    python run_all.py

Phases:
    1-2,4  data_prep   (cleaning + feature engineering — used by all steps)
    1-3    eda          (EDA figures + report)
    5-8    modeling     (imbalance comparison, tuning, evaluation, models)
    9-10   explain_risk (SHAP + risk stratification + patient scores)
    11     powerbi      (dashboard-ready exports + build guide)
"""

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import eda
import modeling
import explain_risk
import powerbi_export


def step(label, fn):
    print("\n" + "=" * 70)
    print(f">>> {label}")
    print("=" * 70)
    t0 = time.perf_counter()
    fn()
    print(f"--- {label} done in {time.perf_counter() - t0:.1f}s")


def main():
    t0 = time.perf_counter()
    step("Phase 1-3: EDA", eda.main)
    step("Phase 5-8: Modeling, tuning, evaluation", modeling.main)
    step("Phase 9-10: SHAP + risk stratification", explain_risk.main)
    step("Phase 11: Power BI exports", powerbi_export.export)
    print("\n" + "=" * 70)
    print(f"ALL PHASES COMPLETE in {time.perf_counter() - t0:.1f}s")
    print("Artifacts: ./outputs/  (figures, models, reports, predictions, powerbi)")
    print("=" * 70)


if __name__ == "__main__":
    main()
