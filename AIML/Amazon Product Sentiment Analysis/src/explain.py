"""SHAP explainability for the TF-IDF + Logistic Regression baseline.

A linear model over TF-IDF features is the ideal SHAP candidate: exact,
fast (LinearExplainer), and the resulting values map directly back to words.
Produces a summary bar plot of the most influential tokens.

    python -m src.explain --sample 20000
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from . import config, data, preprocess


def explain(sample_size: int = 20_000, background: int = 2000):
    model_path = config.MODELS_DIR / "logistic.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "Train the baseline first: python -m src.train_baseline --sample 100000"
        )
    model = joblib.load(model_path)
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]

    preprocess.ensure_nltk()
    df = data.load_split("train", sample_size=sample_size)
    X = preprocess.clean_series(df["text"].tolist())
    Xt = vec.transform(X)

    # Linear SHAP is exact for a linear model; a small background suffices.
    explainer = shap.LinearExplainer(clf, Xt[:background], feature_names=vec.get_feature_names_out())
    shap_values = explainer(Xt[:background])

    shap.plots.bar(shap_values, max_display=20, show=False)
    path = config.FIGURES_DIR / "06_shap_baseline.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="SHAP explainability for the baseline model.")
    ap.add_argument("--sample", type=int, default=20_000)
    args = ap.parse_args()
    explain(args.sample)


if __name__ == "__main__":
    main()
