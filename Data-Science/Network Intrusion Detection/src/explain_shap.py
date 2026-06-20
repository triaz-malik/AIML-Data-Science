"""SHAP explainability for the best tree-based model (summary, bar, waterfall)."""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

import config as C
from preprocessing import get_split

# SHAP TreeExplainer needs a tree model. Prefer XGBoost, fall back to RF.
PREFERRED = ["XGBoost", "Random Forest"]


def _pick_tree_model():
    for name in PREFERRED:
        path = C.MODEL_FILES.get(name)
        if path and path.exists():
            return name, joblib.load(path)
    raise SystemExit("No tree-based model found for SHAP. Run train_models.py first.")


def run(sample_size: int = 1000):
    name, pipe = _pick_tree_model()
    print(f"[shap] explaining: {name}")

    X_train, X_test, y_train, y_test = get_split()

    # The saved object is an imblearn Pipeline: prep -> smote -> clf.
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feat_names = list(prep.get_feature_names_out())

    # Transform a sample of the test set for explanation
    rng = np.random.RandomState(C.RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx]
    X_trans = prep.transform(X_sample)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    # binary classifiers may return a list per class -> take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1) Summary (beeswarm) -------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feat_names, show=False, max_display=20)
    plt.title(f"SHAP Summary - {name}")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "11_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[shap] saved 11_shap_summary.png")

    # 2) Bar (mean |SHAP|) --------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "12_shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[shap] saved 12_shap_bar.png")

    # 3) Waterfall (single attack prediction) -------------------------------
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = np.array(base).ravel()[-1]
    # find a row predicted as attack
    preds = clf.predict(X_trans)
    row = int(np.argmax(preds)) if preds.any() else 0
    expl = shap.Explanation(values=shap_values[row], base_values=base,
                            data=X_trans[row], feature_names=feat_names)
    plt.figure()
    shap.plots.waterfall(expl, max_display=15, show=False)
    plt.title(f"SHAP Waterfall - example #{row}")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "13_shap_waterfall.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[shap] saved 13_shap_waterfall.png")

    # Report top features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])[:12]
    print("[shap] top features by mean|SHAP|:")
    for f, v in top:
        print(f"        {f:35s} {v:.4f}")


if __name__ == "__main__":
    run()
