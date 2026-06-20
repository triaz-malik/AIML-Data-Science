"""
Translate model performance into dollars.

All assumptions live in config.py so the calculation is fully auditable.
The headline number uses the *model's measured recall* on the held-out test
set (not a hand-waved 80%), which makes the ROI defensible in an interview.
"""
from __future__ import annotations

import json

from . import config


def compute(model_recall: float | None = None) -> dict:
    leavers = config.COMPANY_HEADCOUNT * config.BASELINE_ATTRITION_RATE
    annual_loss = leavers * config.COST_PER_LEAVER

    # If we have a measured recall, use it; otherwise fall back to the
    # planning assumption from config.
    detection = model_recall if model_recall is not None else config.SHARE_LEAVERS_DETECTED

    employees_saved = leavers * detection * config.SHARE_DETECTED_RETAINED
    annual_savings = employees_saved * config.COST_PER_LEAVER

    result = {
        "headcount": config.COMPANY_HEADCOUNT,
        "baseline_attrition_rate_pct": config.BASELINE_ATTRITION_RATE * 100,
        "annual_leavers": int(leavers),
        "cost_per_leaver": config.COST_PER_LEAVER,
        "cost_breakdown": {
            "recruitment": config.COST_RECRUITMENT,
            "training": config.COST_TRAINING,
            "productivity": config.COST_PRODUCTIVITY,
        },
        "annual_loss": int(annual_loss),
        "detection_rate_used": round(detection, 3),
        "retention_success_rate": config.SHARE_DETECTED_RETAINED,
        "employees_saved": int(round(employees_saved)),
        "annual_savings": int(round(annual_savings)),
    }
    (config.METRIC_DIR / "business_value.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = compute()
    print(f"Annual loss: ${r['annual_loss']:,}")
    print(f"Employees saved: {r['employees_saved']}  ->  "
          f"Annual savings: ${r['annual_savings']:,}")
