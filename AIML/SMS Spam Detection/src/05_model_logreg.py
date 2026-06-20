"""
Phase 5 - Model 1: TF-IDF + Logistic Regression (baseline)
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

- Combines TF-IDF (word 1-2grams + char 3-5grams) with engineered numeric features
- Stratified 80/20 split, 5-fold CV, GridSearch over C and penalty
- Reports accuracy / precision / recall / F1 (macro + per-class)
- Saves model -> outputs/models/logreg.joblib
- Saves confusion matrix -> outputs/figures/06_logreg_confusion.png
- Saves metrics -> reports/metrics_logreg.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_split, CLASSES, SEED
from features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models"; MODELS.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "reports"


def build_pipeline(memory=None) -> Pipeline:
    word_tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                 min_df=2, max_features=15000, stop_words="english")
    char_tfidf = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb",
                                 ngram_range=(3, 4), min_df=3, max_features=8000)
    text_feats = ColumnTransformer([
        ("word", word_tfidf, "text"),
        ("char", char_tfidf, "text"),
        ("num", StandardScaler(with_mean=False), FEATURE_COLS),
    ])
    # sklearn 1.8: `penalty` is deprecated; elastic-net mixing is controlled by
    # `l1_ratio` (0.0 == pure L2 / ridge, 1.0 == pure L1 / lasso).
    clf = LogisticRegression(max_iter=2000, tol=1e-3, class_weight="balanced",
                             solver="saga", l1_ratio=0.0, random_state=SEED)
    # memory caches the (identical) feature step across grid candidates -> computed
    # once per fold instead of once per (candidate x fold).
    return Pipeline([("feats", text_feats), ("clf", clf)], memory=memory)


def main():
    train_df, test_df = load_split()
    Xtr, ytr = train_df, train_df["label5"]
    Xte, yte = test_df, test_df["label5"]
    print(f"train={len(train_df)}  test={len(test_df)}")

    cache_dir = tempfile.mkdtemp(prefix="sklcache_")
    memory = Memory(location=cache_dir, verbose=0)
    pipe = build_pipeline(memory=memory)
    grid = {
        "clf__C": [1.0, 5.0],
        "clf__l1_ratio": [0.0, 1.0],   # 0.0 = L2, 1.0 = L1 (replaces deprecated `penalty`)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=4, verbose=2)
    gs.fit(Xtr, ytr)
    print(f"\nBest params: {gs.best_params_}")
    print(f"Best CV f1_macro: {gs.best_score_:.4f}")
    shutil.rmtree(cache_dir, ignore_errors=True)

    best = gs.best_estimator_
    pred = best.predict(Xte)

    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    f1w = f1_score(yte, pred, average="weighted")
    print(f"\n=== TEST (5-class) ===")
    print(f"accuracy={acc:.4f}  f1_macro={f1m:.4f}  f1_weighted={f1w:.4f}\n")
    print(classification_report(yte, pred, labels=CLASSES, digits=3, zero_division=0))

    # binary view (ham vs spam) derived from 5-class
    bin_true = (test_df["binary_label"] == "spam").astype(int)
    bin_pred = (np.array(pred) != "Normal").astype(int)
    bin_acc = accuracy_score(bin_true, bin_pred)
    bin_f1 = f1_score(bin_true, bin_pred)
    print(f"Derived BINARY ham/spam:  accuracy={bin_acc:.4f}  f1={bin_f1:.4f}")

    # confusion matrix
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"LogReg Confusion Matrix (acc={acc:.3f}, F1m={f1m:.3f})", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIG / "06_logreg_confusion.png", dpi=130)
    plt.close(fig)

    joblib.dump(best, MODELS / "logreg.joblib")
    metrics = {
        "model": "tfidf_logreg",
        "best_params": gs.best_params_,
        "cv_f1_macro": gs.best_score_,
        "test_accuracy": acc, "test_f1_macro": f1m, "test_f1_weighted": f1w,
        "binary_accuracy": bin_acc, "binary_f1": bin_f1,
        "report": classification_report(yte, pred, labels=CLASSES, output_dict=True, zero_division=0),
    }
    (REP / "metrics_logreg.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved model -> outputs/models/logreg.joblib")
    print("Saved metrics -> reports/metrics_logreg.json")


if __name__ == "__main__":
    main()
