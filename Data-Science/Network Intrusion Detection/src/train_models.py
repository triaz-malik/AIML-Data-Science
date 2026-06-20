"""Train LogReg, Random Forest, XGBoost and an MLP neural net.

For each model:
  - pipeline = preprocessor -> SMOTE -> classifier   (SMOTE runs only on the
    training fold, so there is no leakage into validation)
  - hyperparameters tuned with RandomizedSearchCV (StratifiedKFold)
  - best estimator refit on the full training set and saved as a .pkl
  - StratifiedKFold(5) score reported as the robust cross-validation metric

Also runs a small SMOTE vs ADASYN vs none comparison on XGBoost.
Held-out test metrics are written to models/metrics.json for evaluate.py / report.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import config as C
from preprocessing import build_preprocessor, get_split

CV_TUNE = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_STATE)
CV_EVAL = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def make_pipeline(preprocessor, clf, sampler=None):
    sampler = sampler if sampler is not None else SMOTE(random_state=C.RANDOM_STATE)
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", sampler),
        ("clf", clf),
    ])


def model_search_space(preprocessor):
    """Return {name: (pipeline, param_distributions, n_iter)}."""
    spaces = {}

    # 1) Logistic Regression -- baseline
    spaces["Logistic Regression"] = (
        make_pipeline(preprocessor,
                      LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE)),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        4,
    )

    # 2) Random Forest
    spaces["Random Forest"] = (
        make_pipeline(preprocessor,
                      RandomForestClassifier(random_state=C.RANDOM_STATE, n_jobs=-1)),
        {"clf__n_estimators": [200, 400, 600],
         "clf__max_depth": [None, 20, 30],
         "clf__min_samples_split": [2, 5, 10],
         "clf__max_features": ["sqrt", "log2"]},
        12,
    )

    # 3) XGBoost -- usually the strongest on tabular intrusion data
    spaces["XGBoost"] = (
        make_pipeline(preprocessor,
                      XGBClassifier(random_state=C.RANDOM_STATE, n_jobs=-1,
                                    eval_metric="logloss", tree_method="hist")),
        {"clf__n_estimators": [300, 500, 700],
         "clf__learning_rate": [0.03, 0.1, 0.2],
         "clf__max_depth": [4, 6, 8],
         "clf__subsample": [0.8, 1.0],
         "clf__colsample_bytree": [0.8, 1.0]},
        12,
    )

    # 4) Neural Network (sklearn MLP: 256 -> 128 -> 64 with dropout-like alpha)
    spaces["Neural Network (MLP)"] = (
        make_pipeline(preprocessor,
                      MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                    early_stopping=True, max_iter=60,
                                    random_state=C.RANDOM_STATE)),
        {"clf__alpha": [1e-4, 1e-3, 1e-2],
         "clf__learning_rate_init": [1e-3, 5e-3],
         "clf__batch_size": [128, 256]},
        6,
    )
    return spaces


def evaluate_on_test(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def sampler_comparison(preprocessor, X_train, y_train):
    """Quick SMOTE vs ADASYN vs none comparison on a fixed XGBoost (5-fold F1)."""
    base = lambda: XGBClassifier(random_state=C.RANDOM_STATE, n_jobs=-1,
                                 eval_metric="logloss", tree_method="hist",
                                 n_estimators=300, max_depth=6, learning_rate=0.1)
    variants = {
        "none": ImbPipeline([("prep", preprocessor), ("clf", base())]),
        "SMOTE": make_pipeline(preprocessor, base(), SMOTE(random_state=C.RANDOM_STATE)),
        "ADASYN": make_pipeline(preprocessor, base(), ADASYN(random_state=C.RANDOM_STATE)),
    }
    out = {}
    for name, pipe in variants.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=CV_EVAL, scoring="f1", n_jobs=-1)
        out[name] = {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}
        print(f"[sampler] {name:7s} F1 = {scores.mean():.4f} +/- {scores.std():.4f}")
    return out


def main():
    t0 = time.time()
    X_train, X_test, y_train, y_test = get_split()
    print(f"[train] train={X_train.shape} test={X_test.shape} "
          f"| train attack rate={y_train.mean():.2%}")

    preprocessor = build_preprocessor(X_train)

    # Persist a standalone fitted preprocessor (satisfies 'scaler/encoder' artefacts)
    fitted_prep = build_preprocessor(X_train).fit(X_train)
    joblib.dump(fitted_prep, C.PREPROCESSOR_PKL)
    print(f"[train] saved preprocessor -> {C.PREPROCESSOR_PKL.name}")

    # Imbalance handling comparison ----------------------------------------
    print("\n[train] --- SMOTE vs ADASYN vs none (XGBoost, 5-fold F1) ---")
    sampler_results = sampler_comparison(preprocessor, X_train, y_train)

    # Train + tune each model ----------------------------------------------
    metrics = {"_sampler_comparison": sampler_results, "models": {}}
    for name, (pipe, params, n_iter) in model_search_space(preprocessor).items():
        print(f"\n[train] === {name} ===")
        ts = time.time()
        search = RandomizedSearchCV(
            pipe, params, n_iter=n_iter, scoring="f1", cv=CV_TUNE,
            random_state=C.RANDOM_STATE, n_jobs=-1, refit=True, verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # robust 5-fold CV on the tuned pipeline
        cv_f1 = cross_val_score(best, X_train, y_train, cv=CV_EVAL, scoring="f1", n_jobs=-1)
        test_metrics = evaluate_on_test(best, X_test, y_test)

        joblib.dump(best, C.MODEL_FILES[name])
        metrics["models"][name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "test": test_metrics,
            "model_file": C.MODEL_FILES[name].name,
        }
        print(f"[train] best params: {metrics['models'][name]['best_params']}")
        print(f"[train] 5-fold CV F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
        print(f"[train] held-out test: "
              + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
        print(f"[train] {name} done in {time.time()-ts:.1f}s -> {C.MODEL_FILES[name].name}")

    # Pick best model by held-out F1, save a copy as best_model.pkl ---------
    best_name = max(metrics["models"], key=lambda n: metrics["models"][n]["test"]["f1"])
    metrics["best_model"] = best_name
    joblib.dump(joblib.load(C.MODEL_FILES[best_name]), C.BEST_MODEL_PKL)
    print(f"\n[train] BEST MODEL: {best_name} "
          f"(test F1={metrics['models'][best_name]['test']['f1']:.4f}) "
          f"-> {C.BEST_MODEL_PKL.name}")

    with open(C.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] metrics -> {C.METRICS_JSON.name} | total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
