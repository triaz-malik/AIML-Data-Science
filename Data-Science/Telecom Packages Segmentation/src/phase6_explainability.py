"""
Phase 6 - Explainability (SHAP).

Explains the tuned tree model's churn predictions. Produces a SHAP
summary (beeswarm) plot and a global feature-importance bar plot, and
prints the top churn drivers - expected to include Customer service
calls, International plan, Day minutes/charges and total revenue.
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import config
from phase2_features import add_features
from phase5_churn import MODEL_FEATURES, run as run_phase5
from utils import load_train_test, savefig, section


def _get_tree_step(pipeline):
    """Extract the underlying tree classifier from the imblearn pipeline."""
    clf = pipeline.named_steps["clf"]
    return clf


def run() -> None:
    section("PHASE 6 - EXPLAINABILITY (SHAP)")

    bundle_path = config.MODEL_DIR / "best_churn_model.joblib"
    if not bundle_path.exists():
        run_phase5()
    bundle = joblib.load(bundle_path)
    pipeline, name = bundle["model"], bundle["name"]
    clf = _get_tree_step(pipeline)

    # SHAP TreeExplainer needs a tree model. If the best model is KNN,
    # fall back to fitting an XGBoost on the same data for explanation.
    train, test = load_train_test()
    train, test = add_features(train), add_features(test)
    X_tr = train[MODEL_FEATURES]
    X_te = test[MODEL_FEATURES]

    if not isinstance(clf, (RandomForestClassifier, XGBClassifier)):
        print(f"Best model ({name}) is not tree-based; fitting XGBoost for SHAP.")
        clf = XGBClassifier(
            random_state=config.RANDOM_STATE, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, n_estimators=300, max_depth=6,
            learning_rate=0.1)
        clf.fit(X_tr, train[config.TARGET])
        name = "XGBoost (for SHAP)"

    print(f"Explaining model: {name}")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_te)

    # RandomForest returns a list [class0, class1]; pick churn class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # newer SHAP may return (n, features, classes)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # ---- beeswarm summary ------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_te, show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle("SHAP Summary - Churn Drivers", y=1.02)
    savefig(fig, "14_shap_summary_beeswarm")

    # ---- global importance bar ------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_te, plot_type="bar", show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle("SHAP Global Feature Importance", y=1.02)
    savefig(fig, "15_shap_importance_bar")

    # ---- ranked drivers --------------------------------------------------
    importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=MODEL_FEATURES)
        .sort_values(ascending=False)
    )
    print("\nTop churn drivers (mean |SHAP|):")
    print(importance.head(10).round(4).to_string())
    importance.to_csv(config.REPORT_DIR / "phase6_shap_importance.csv",
                      header=["mean_abs_shap"])
    print(f"\n[report] {(config.REPORT_DIR / 'phase6_shap_importance.csv').relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
