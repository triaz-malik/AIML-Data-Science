"""
Baseline model: TF-IDF + Logistic Regression.
- Stratified 80/20 train/test split (shared with the transformers via saved indices).
- 5-fold GridSearchCV over C / penalty / solver for hyperparameter tuning.
- Saves metrics JSON, per-class report, and a confusion-matrix figure.
"""
import json
import os
import time

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
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split

BASE = r"C:\Working\AI ML Projetcs\Consumer Complaints"
DATA = os.path.join(BASE, "data", "features.parquet")
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")


def main():
    df = pd.read_parquet(DATA, columns=["clean_text", "category"])
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    X = df["clean_text"].astype(str).to_numpy(dtype=object)
    y = df["category"].astype(str).to_numpy(dtype=object)

    # Stratified split; persist test indices so transformers evaluate on the SAME test set.
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

    # --- Hyperparameter tuning: 5-fold grid search over C ---
    # multinomial LogReg with the lbfgs solver + L2 penalty (the combo that
    # natively supports >=3 classes in scikit-learn 1.8). We sweep the inverse
    # regularization strength C.
    grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    base = LogisticRegression(solver="lbfgs", penalty="l2", max_iter=1000, n_jobs=-1)
    print("Running 5-fold GridSearchCV (this takes several minutes)...")
    t0 = time.time()
    gs = GridSearchCV(base, grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1)
    gs.fit(Xtr_v, ytr)
    tune_secs = round(time.time() - t0, 1)
    print(f"Best params: {gs.best_params_} | CV weighted-F1: {gs.best_score_:.4f} | {tune_secs}s")

    best = gs.best_estimator_
    pred = best.predict(Xte_v)

    acc = accuracy_score(yte, pred)
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "best_params": gs.best_params_,
        "cv_weighted_f1": round(float(gs.best_score_), 4),
        "tuning_seconds": tune_secs,
        "test_accuracy": round(float(acc), 4),
        "test_f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
        "test_f1_weighted": round(float(f1_score(yte, pred, average="weighted")), 4),
        "test_precision_weighted": round(float(precision_score(yte, pred, average="weighted")), 4),
        "test_recall_weighted": round(float(recall_score(yte, pred, average="weighted")), 4),
        "n_features": Xtr_v.shape[1],
    }
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    metrics["per_class_f1"] = {k: round(v["f1-score"], 3)
                               for k, v in rep.items() if k in set(y)}

    # cv table for the report
    cv_df = pd.DataFrame(gs.cv_results_)[["param_C",
                                          "mean_test_score", "std_test_score"]]
    cv_df = cv_df.sort_values("mean_test_score", ascending=False).round(4)
    metrics["grid_results"] = cv_df.to_dict(orient="records")

    with open(os.path.join(METRICS, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("grid_results", "per_class_f1")}, indent=2))

    # --- Confusion matrix ---
    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "recall"})
    plt.title("Confusion matrix — TF-IDF + Logistic Regression (row-normalized)")
    plt.ylabel("True category")
    plt.xlabel("Predicted category")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "10_confusion_baseline.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("saved 10_confusion_baseline.png")


if __name__ == "__main__":
    main()
