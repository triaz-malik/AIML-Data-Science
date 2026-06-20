"""
Phase 5 + 7 + 8 — Baseline model: TF-IDF + Logistic Regression (3-class sentiment).

- Canonical modeling frame: features.parquet rows with non-empty clean_text,
  reset_index(drop=True). The SAME frame + saved split.npz is reused by the
  transformers so every model is scored on the identical test set.
- Stratified 80/20 split (indices saved to data/split.npz).
- 5-fold GridSearchCV over C / class_weight (tuning).
- Metrics: accuracy, precision/recall/F1 (macro+weighted), multiclass ROC-AUC (OVR).
- Saves baseline_metrics.json + a row-normalized confusion matrix figure.
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
DATA = os.path.join(BASE, "data", "features.parquet")
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")
LABELS = ["negative", "neutral", "positive"]


def load_modeling_frame():
    """Shared loader so baseline + transformers align on identical row indices."""
    df = pd.read_parquet(DATA, columns=["clean_text", "reviewText", "sentiment"])
    df = df[df["clean_text"].fillna("").str.len() > 0].reset_index(drop=True)
    return df


def main():
    df = load_modeling_frame()
    X = df["clean_text"].astype(str).to_numpy(dtype=object)
    y = df["sentiment"].astype(str).to_numpy(dtype=object)
    print(f"Modeling rows: {len(df):,}")

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    np.savez(os.path.join(BASE, "data", "split.npz"), train=tr_idx, test=te_idx)
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    print(f"Train {len(Xtr):,} | Test {len(Xte):,}")

    print("Fitting TF-IDF (1-2 grams, max 50k features)...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                            min_df=5, sublinear_tf=True)
    Xtr_v = tfidf.fit_transform(Xtr)
    Xte_v = tfidf.transform(Xte)

    grid = {"C": [0.5, 1.0, 2.0, 5.0], "class_weight": [None, "balanced"]}
    base = LogisticRegression(solver="lbfgs", max_iter=1000)  # l2 is the default
    print("Running 5-fold GridSearchCV...")
    t0 = time.time()
    gs = GridSearchCV(base, grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(Xtr_v, ytr)
    tune_secs = round(time.time() - t0, 1)
    print(f"Best: {gs.best_params_} | CV macro-F1 {gs.best_score_:.4f} | {tune_secs}s")

    best = gs.best_estimator_
    pred = best.predict(Xte_v)
    proba = best.predict_proba(Xte_v)

    # persist fitted vectorizer + model for explainability / error analysis
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    joblib.dump({"tfidf": tfidf, "model": best, "labels": LABELS},
                os.path.join(BASE, "models", "baseline.joblib"))

    yte_bin = label_binarize(yte, classes=LABELS)
    auc_ovr = roc_auc_score(yte_bin, proba, average="macro", multi_class="ovr")

    acc = accuracy_score(yte, pred)
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "task": "3-class sentiment (negative/neutral/positive)",
        "best_params": gs.best_params_,
        "cv_macro_f1": round(float(gs.best_score_), 4),
        "tuning_seconds": tune_secs,
        "test_accuracy": round(float(acc), 4),
        "test_f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
        "test_f1_weighted": round(float(f1_score(yte, pred, average="weighted")), 4),
        "test_precision_macro": round(float(precision_score(yte, pred, average="macro", zero_division=0)), 4),
        "test_recall_macro": round(float(recall_score(yte, pred, average="macro", zero_division=0)), 4),
        "test_roc_auc_ovr_macro": round(float(auc_ovr), 4),
        "n_features": Xtr_v.shape[1],
        "n_test": int(len(yte)),
    }
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    metrics["per_class_f1"] = {k: round(rep[k]["f1-score"], 3) for k in LABELS}

    cv_df = pd.DataFrame(gs.cv_results_)[["param_C", "param_class_weight",
                                          "mean_test_score", "std_test_score"]]
    cv_df = cv_df.sort_values("mean_test_score", ascending=False).round(4)
    cv_df["param_class_weight"] = cv_df["param_class_weight"].astype(str)
    metrics["grid_results"] = cv_df.to_dict(orient="records")

    with open(os.path.join(METRICS, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("grid_results",)}, indent=2))

    cm = confusion_matrix(yte, pred, labels=LABELS, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, cbar_kws={"label": "recall"})
    plt.title("Confusion matrix — TF-IDF + Logistic Regression")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "08_confusion_baseline.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("saved 08_confusion_baseline.png")


if __name__ == "__main__":
    main()
