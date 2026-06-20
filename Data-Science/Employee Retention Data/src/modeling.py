"""
Modeling: preprocessing, three classifiers, hyperparameter tuning, evaluation.

Models
    1. Logistic Regression  -- simple, explainable baseline
    2. Random Forest        -- non-linear, tuned
    3. XGBoost              -- gradient boosting, tuned

Class imbalance (~16% positives) is handled with class_weight="balanced"
(LogReg / RF) and scale_pos_weight (XGBoost). We optimise for ROC-AUC during
tuning because the business cares about ranking risky employees, not raw
accuracy on an imbalanced target.

Artifacts written:
    outputs/models/preprocessor.joblib
    outputs/models/<model>.joblib
    outputs/metrics/model_results.json
    outputs/figures/roc_curves.png
    outputs/figures/confusion_<model>.png
    outputs/figures/model_comparison.png
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config
from .data_prep import build_features, get_feature_columns, load_raw

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    """Scale numerics (helps LogReg) and one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def split_and_transform(df: pd.DataFrame):
    """Stratified 80/20 split, fit preprocessor on train only."""
    numeric, categorical = get_feature_columns(df)
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y,
    )

    pre = build_preprocessor(numeric, categorical)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())

    return (X_train_t, X_test_t, y_train.values, y_test.values,
            feature_names, pre)


# --------------------------------------------------------------------------- #
# Model definitions + tuning grids
# --------------------------------------------------------------------------- #
def _models_and_grids(scale_pos_weight: float):
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
            {"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [4, 6, 8, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic", eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                random_state=config.RANDOM_STATE, n_jobs=-1,
                tree_method="hist",
            ),
            {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
    }


def tune(model, grid, X_train, y_train, n_iter=25):
    """Randomised search with 5-fold stratified CV, scoring on ROC-AUC."""
    search = RandomizedSearchCV(
        model, grid, n_iter=n_iter, scoring="roc_auc", cv=5,
        random_state=config.RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(name, model, X_test, y_test, threshold) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "_proba": proba,  # kept in-memory for ROC plot, stripped before JSON
    }


def plot_confusion(name, cm, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(np.array(cm), display_labels=["Stay", "Leave"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_roc(results, y_test, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_proba"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metrics))
    width = 0.25
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=r["model"])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    df = build_features(load_raw())
    X_train, X_test, y_train, y_test, feature_names, pre = split_and_transform(df)

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train={len(y_train)} Test={len(y_test)} | scale_pos_weight={spw:.2f}")

    joblib.dump(pre, config.MODEL_DIR / "preprocessor.joblib")
    (config.MODEL_DIR / "feature_names.json").write_text(json.dumps(feature_names))

    results = []
    best_params_all = {}
    for name, (model, grid) in _models_and_grids(spw).items():
        print(f"Tuning {name} ...")
        best, params = tune(model, grid, X_train, y_train)
        best_params_all[name] = params
        res = evaluate(name, best, X_test, y_test, config.DECISION_THRESHOLD)
        results.append(res)
        joblib.dump(best, config.MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib")
        plot_confusion(
            name, res["confusion_matrix"],
            config.FIG_DIR / f"confusion_{name.replace(' ', '_').lower()}.png",
        )
        print(f"  {name}: AUC={res['roc_auc']}  recall={res['recall']}  "
              f"precision={res['precision']}  f1={res['f1']}")

    plot_roc(results, y_test, config.FIG_DIR / "roc_curves.png")
    plot_comparison(results, config.FIG_DIR / "model_comparison.png")

    # Winner = highest ROC-AUC (tie-break on recall).
    winner = max(results, key=lambda r: (r["roc_auc"], r["recall"]))
    print(f"Winner: {winner['model']}")

    # Strip in-memory proba arrays before serialising.
    for r in results:
        r.pop("_proba", None)

    summary = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_attrition_rate_pct": round(float(y_test.mean()) * 100, 1),
        "scale_pos_weight": round(spw, 2),
        "decision_threshold": config.DECISION_THRESHOLD,
        "best_params": best_params_all,
        "results": results,
        "winner": winner["model"],
    }
    (config.METRIC_DIR / "model_results.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
