"""
SHAP explainability.

Answers the question HR always asks: "WHY is this employee predicted to leave?"

We use SHAP on the gradient-boosted (XGBoost) model because TreeExplainer gives
exact, fast Shapley values for tree ensembles. Outputs:

    outputs/figures/shap_summary.png        -- global feature importance (beeswarm)
    outputs/figures/shap_bar.png            -- mean |SHAP| ranking
    outputs/figures/shap_employee.png       -- one high-risk employee waterfall
    outputs/metrics/shap_findings.json      -- top drivers + employee narrative
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from . import config
from .data_prep import build_features, load_raw
from .modeling import split_and_transform


def _clean_name(raw: str) -> str:
    """Turn 'num__MonthlyIncome' / 'cat__OverTime_Yes' into readable labels."""
    return raw.split("__", 1)[-1]


def run() -> dict:
    df = build_features(load_raw())
    X_train, X_test, y_train, y_test, feature_names, _ = split_and_transform(df)
    pretty_names = [_clean_name(f) for f in feature_names]

    model = joblib.load(config.MODEL_DIR / "xgboost.joblib")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- Global importance: mean |SHAP| -----------------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_drivers = [
        {"feature": pretty_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order[:12]
    ]

    # --- Beeswarm summary plot --------------------------------------------
    shap.summary_plot(shap_values, X_test, feature_names=pretty_names,
                      show=False, max_display=12)
    plt.title("SHAP Summary — global attrition drivers")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(config.FIG_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --- Bar ranking -------------------------------------------------------
    shap.summary_plot(shap_values, X_test, feature_names=pretty_names,
                      plot_type="bar", show=False, max_display=12)
    plt.title("SHAP Feature Importance (mean |SHAP|)")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(config.FIG_DIR / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --- One high-risk employee -------------------------------------------
    proba = model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(proba))  # most at-risk employee in the test set
    expl = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test[idx],
        feature_names=pretty_names,
    )
    plt.figure()
    shap.plots.waterfall(expl, max_display=10, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(config.FIG_DIR / "shap_employee.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Narrative: top positive contributors for this employee.
    contrib = sorted(
        zip(pretty_names, shap_values[idx]), key=lambda t: t[1], reverse=True
    )
    employee_drivers = [
        {"feature": f, "shap": round(float(v), 4)}
        for f, v in contrib[:5] if v > 0
    ]

    findings = {
        "explained_model": "XGBoost",
        "top_global_drivers": top_drivers,
        "example_employee": {
            "test_index": idx,
            "predicted_risk_pct": round(float(proba[idx]) * 100, 1),
            "actual_attrition": int(y_test[idx]),
            "top_reasons": employee_drivers,
        },
    }
    (config.METRIC_DIR / "shap_findings.json").write_text(json.dumps(findings, indent=2))
    print(f"SHAP complete. Top driver: {top_drivers[0]['feature']}. "
          f"Example employee risk: {findings['example_employee']['predicted_risk_pct']}%")
    return findings


if __name__ == "__main__":
    run()
