"""
Phase 5 - Churn Prediction.

Three models (KNN, Random Forest, XGBoost), each in an imbalanced-learn
pipeline with SMOTE, tuned with GridSearchCV over Stratified K-Fold.
The provided 80/20 split is used as train / holdout test.

Outputs metrics, ROC curves, confusion matrix, the best model, and a
Power BI-ready scored table (out-of-fold churn probabilities for all
customers) at outputs/data/telecom_scored.csv
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from phase2_features import add_features
from utils import load_train_test, savefig, section

sns.set_theme(style="whitegrid")

MODEL_FEATURES = [
    "Account length", "International plan", "Voice mail plan",
    "Number vmail messages",
    "Total day minutes", "Total day calls", "Total day charge",
    "Total eve minutes", "Total eve calls", "Total eve charge",
    "Total night minutes", "Total night calls", "Total night charge",
    "Total intl minutes", "Total intl calls", "Total intl charge",
    "Customer service calls",
    "Total Usage Minutes", "Total Charges", "International Usage Ratio",
]


def _models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "KNN": (
            ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=rs)),
                ("clf", KNeighborsClassifier()),
            ]),
            {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]},
        ),
        "RandomForest": (
            ImbPipeline([
                ("smote", SMOTE(random_state=rs)),
                ("clf", RandomForestClassifier(random_state=rs, n_jobs=-1)),
            ]),
            {"clf__n_estimators": [300], "clf__max_depth": [None, 12],
             "clf__min_samples_leaf": [1, 3]},
        ),
        "XGBoost": (
            ImbPipeline([
                ("smote", SMOTE(random_state=rs)),
                ("clf", XGBClassifier(
                    random_state=rs, eval_metric="logloss",
                    tree_method="hist", n_jobs=-1)),
            ]),
            {"clf__n_estimators": [300], "clf__max_depth": [4, 6],
             "clf__learning_rate": [0.05, 0.1]},
        ),
    }


def run() -> dict:
    section("PHASE 5 - CHURN PREDICTION")
    train, test = load_train_test()
    train, test = add_features(train), add_features(test)

    X_tr, y_tr = train[MODEL_FEATURES], train[config.TARGET]
    X_te, y_te = test[MODEL_FEATURES], test[config.TARGET]
    print(f"Train: {len(X_tr):,} (churn {y_tr.mean():.1%})  |  "
          f"Test: {len(X_te):,} (churn {y_te.mean():.1%})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted = {}, {}

    for name, (pipe, grid) in _models().items():
        print(f"\n>>> Tuning {name} ...")
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        proba = best.predict_proba(X_te)[:, 1]
        pred = best.predict(X_te)
        results[name] = {
            "cv_roc_auc": gs.best_score_,
            "test_roc_auc": roc_auc_score(y_te, proba),
            "test_f1": f1_score(y_te, pred),
            "best_params": gs.best_params_,
            "proba": proba,
            "pred": pred,
        }
        fitted[name] = best
        print(f"    best params: {gs.best_params_}")
        print(f"    CV ROC-AUC={gs.best_score_:.3f} | "
              f"Test ROC-AUC={results[name]['test_roc_auc']:.3f} | "
              f"Test F1={results[name]['test_f1']:.3f}")

    # ---- comparison table ------------------------------------------------
    section("MODEL COMPARISON (holdout test)")
    comp = pd.DataFrame({
        n: {"CV ROC-AUC": r["cv_roc_auc"],
            "Test ROC-AUC": r["test_roc_auc"],
            "Test F1": r["test_f1"]}
        for n, r in results.items()
    }).T.round(4)
    print(comp.to_string())
    comp.to_csv(config.REPORT_DIR / "phase5_model_comparison.csv")

    best_name = comp["Test ROC-AUC"].idxmax()
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print("\nClassification report (best model on holdout):")
    print(classification_report(y_te, results[best_name]["pred"],
                                target_names=["Retained", "Churned"]))

    # ---- ROC curves ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_te, r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(title="ROC Curves - Churn Models", xlabel="False Positive Rate",
           ylabel="True Positive Rate")
    ax.legend()
    savefig(fig, "12_roc_curves")

    # ---- confusion matrix (best) ----------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, results[best_name]["pred"],
        display_labels=["Retained", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    savefig(fig, "13_confusion_matrix")

    # ---- persist best model ---------------------------------------------
    joblib.dump(
        {"model": best_model, "features": MODEL_FEATURES, "name": best_name},
        config.MODEL_DIR / "best_churn_model.joblib",
    )
    print(f"[model] {(config.MODEL_DIR / 'best_churn_model.joblib').relative_to(config.PROJECT_ROOT)}")

    # ---- Power BI-ready scored table (out-of-fold probs for everyone) ----
    tr, te = load_train_test()
    full = add_features(pd.concat([tr, te], ignore_index=True))
    Xf, yf = full[MODEL_FEATURES], full[config.TARGET]
    oof = cross_val_predict(best_model, Xf, yf, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    full["Churn Probability"] = oof.round(4)
    full["Churn Risk Band"] = pd.cut(
        full["Churn Probability"], bins=[-0.01, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"])

    # merge segment / package info if available
    if config.SEGMENTED_CSV.exists():
        seg = pd.read_csv(config.SEGMENTED_CSV)
        for col in ["Segment", "Current Package", "Recommended Package",
                    "Revenue Segment", "Usage Segment"]:
            if col in seg.columns and len(seg) == len(full):
                full[col] = seg[col].values

    full.to_csv(config.SCORED_CSV, index=False)
    print(f"[data] {config.SCORED_CSV.relative_to(config.PROJECT_ROOT)} "
          f"(Power BI ready, {full.shape[0]} rows)")

    return {"results": results, "best_name": best_name, "comparison": comp}


if __name__ == "__main__":
    run()
