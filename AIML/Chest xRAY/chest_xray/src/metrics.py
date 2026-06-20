"""Metric computation. Priority order per spec: Recall > F1 > AUC > Accuracy.

Positive class = PNEUMONIA (index 1). False negatives (missed pneumonia) are
the costly errors, so recall on the positive class is the headline KPI.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """y_prob = P(PNEUMONIA). Returns the full KPI dict."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    pos = C.POSITIVE_IDX
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "recall": recall_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }


def best_threshold_for_recall(y_true, y_prob, min_precision: float = 0.0) -> float:
    """Pick the threshold maximising recall subject to a precision floor.

    Useful for the hospital trade-off: 'highest recall we can get while keeping
    precision >= min_precision'. With min_precision=0 it favours recall outright.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_t, best_r = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(y_true, y_prob, threshold=t)
        if m["precision"] >= min_precision and m["recall"] > best_r:
            best_r, best_t = m["recall"], t
    return float(best_t)
