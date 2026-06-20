"""Baseline model: TF-IDF + Logistic Regression.

Fast, CPU-friendly, and interpretable — the yardstick the transformers must
beat. Run as a script to train, evaluate on the held-out test split, and
persist the fitted pipeline to models/logistic.pkl.

    python -m src.train_baseline --sample 100000 --grid
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from . import config, data, preprocess


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_features=100_000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")),
        ]
    )


def _clean_split(df):
    return preprocess.clean_series(df["text"].tolist())


def train(sample_size: int | None, do_grid: bool) -> dict:
    print(f"Loading data (sample_size={sample_size}) ...")
    train_df = data.load_split("train", sample_size=sample_size)
    # Keep the test split proportional but capped so eval stays quick.
    test_size = None if sample_size is None else max(20_000, sample_size // 5)
    test_df = data.load_split("test", sample_size=test_size)

    print("Cleaning text ...")
    preprocess.ensure_nltk()
    X_train, y_train = _clean_split(train_df), train_df["label"].values
    X_test, y_test = _clean_split(test_df), test_df["label"].values

    pipe = build_pipeline()

    t0 = time.time()
    if do_grid:
        print("Running GridSearchCV (5-fold) ...")
        # Tune regularization strength C. We keep the (liblinear, l2) default
        # rather than gridding `penalty`, which sklearn >=1.8 deprecated.
        grid = {
            "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0],
        }
        search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params: {best_params}")
    else:
        print("Fitting pipeline ...")
        pipe.fit(X_train, y_train)
        model = pipe
        best_params = {"clf__C": 1.0}
    fit_secs = round(time.time() - t0, 1)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.4f}   F1: {f1:.4f}   (fit {fit_secs}s)")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    out_path = config.MODELS_DIR / "logistic.pkl"
    joblib.dump(model, out_path)
    print(f"Saved model -> {out_path}")

    metrics = {
        "model": "tfidf_logreg",
        "sample_size": sample_size,
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "best_params": best_params,
        "fit_seconds": fit_secs,
    }
    (config.MODELS_DIR / "logistic_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train the TF-IDF + LogReg baseline.")
    ap.add_argument("--sample", type=int, default=config.DEFAULT_SAMPLE_SIZE,
                    help="balanced training subsample size (use 0 for full dataset)")
    ap.add_argument("--grid", action="store_true", help="run GridSearchCV hyperparameter tuning")
    args = ap.parse_args()
    sample = None if args.sample == 0 else args.sample
    train(sample, args.grid)


if __name__ == "__main__":
    main()
