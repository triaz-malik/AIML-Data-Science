"""
End-to-end pipeline orchestrator.

Run everything:
    python main.py

Steps:
    1. EDA           -> figures + business-question findings
    2. Modeling      -> train/tune LogReg, RF, XGBoost; evaluate; save models
    3. SHAP          -> global + per-employee explanations
    4. Business value-> ROI from measured recall
    5. Report        -> reports/Employee_Attrition_Report.pdf
"""
from __future__ import annotations

import time

from src import business_value, eda, explain, modeling, report


def main() -> None:
    t0 = time.time()

    print("\n=== [1/5] Exploratory Data Analysis ===")
    eda.run()

    print("\n=== [2/5] Modeling (LogReg / Random Forest / XGBoost) ===")
    model_summary = modeling.run()

    print("\n=== [3/5] SHAP explainability ===")
    explain.run()

    print("\n=== [4/5] Business value / ROI ===")
    winner = next(r for r in model_summary["results"]
                  if r["model"] == model_summary["winner"])
    biz = business_value.compute(model_recall=winner["recall"])
    print(f"Annual savings (recall={winner['recall']:.2f}): "
          f"${biz['annual_savings']:,}")

    print("\n=== [5/5] PDF report ===")
    path = report.build()

    print(f"\nPipeline complete in {time.time() - t0:.0f}s")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
