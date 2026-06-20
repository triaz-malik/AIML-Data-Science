"""
Phase 9 (Explainability) + Phase 10 (Error Analysis) for the baseline model.

Explainability:
  - Global: top tokens pushing each class (from Logistic Regression coefficients).
  - Local: SHAP token-level attributions for a few example reviews (best-effort;
    falls back to per-token coefficient contributions if SHAP is unavailable).
Error analysis:
  - Re-scores the held-out test set, samples representative misclassifications
    (e.g. predicted-positive-but-negative), and characterises the failure modes.

Outputs: outputs/figures/11_token_importance.png, outputs/metrics/explainability.json,
         outputs/metrics/error_analysis.json
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from train_baseline import LABELS, load_modeling_frame

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")


def global_tokens(tfidf, model, k=18):
    """Top tokens per class by Logistic Regression coefficient."""
    feats = np.array(tfidf.get_feature_names_out())
    out = {}
    for ci, cls in enumerate(model.classes_):
        coefs = model.coef_[ci]
        top = coefs.argsort()[::-1][:k]
        out[cls] = [(feats[i], round(float(coefs[i]), 3)) for i in top]
    return out


def plot_tokens(token_map):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(16, 6))
    cmaps = {"negative": "Reds", "neutral": "Greys", "positive": "Greens"}
    for ax, cls in zip(axes, LABELS):
        toks, vals = zip(*token_map[cls][:12])
        sns.barplot(x=list(vals), y=list(toks), hue=list(toks),
                    palette=cmaps[cls], legend=False, ax=ax)
        ax.set_title(f"Top tokens -> {cls}")
        ax.set_xlabel("LogReg coefficient"); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "11_token_importance.png"), dpi=130, bbox_inches="tight")
    plt.close()


def local_shap(bundle, examples):
    """Best-effort SHAP text explanations; returns list of {text, pred, top_tokens}."""
    tfidf, model = bundle["tfidf"], bundle["model"]

    def predict(texts):
        return model.predict_proba(tfidf.transform(texts))

    results = []
    try:
        import shap
        masker = shap.maskers.Text(r"\W+")
        explainer = shap.Explainer(predict, masker, output_names=LABELS)
        sv = explainer(examples[:6], max_evals=300, silent=True)
        for i, txt in enumerate(examples[:6]):
            pred_idx = int(predict([txt])[0].argmax())
            cls = LABELS[pred_idx]
            vals = sv[i, :, pred_idx].values
            toks = sv[i].data
            order = np.argsort(np.abs(vals))[::-1][:6]
            top = [(str(toks[j]).strip(), round(float(vals[j]), 4))
                   for j in order if str(toks[j]).strip()]
            results.append({"text": txt[:240], "predicted": cls, "top_tokens_shap": top})
    except Exception as e:  # SHAP version / perf hiccups shouldn't kill the phase
        print(f"SHAP unavailable ({e}); falling back to coefficient contributions.")
        feats = np.array(tfidf.get_feature_names_out())
        vocab = {f: i for i, f in enumerate(feats)}
        for txt in examples[:6]:
            vec = tfidf.transform([txt])
            pred_idx = int(model.predict_proba(vec)[0].argmax())
            cls = model.classes_[pred_idx]
            contrib = []
            row = vec.tocoo()
            for j, v in zip(row.col, row.data):
                contrib.append((feats[j], float(model.coef_[pred_idx][j] * v)))
            contrib.sort(key=lambda x: abs(x[1]), reverse=True)
            results.append({"text": txt[:240], "predicted": cls,
                            "top_tokens_coef": [(t, round(c, 4)) for t, c in contrib[:6]]})
    return results


def main():
    bundle = joblib.load(os.path.join(BASE, "models", "baseline.joblib"))
    tfidf, model = bundle["tfidf"], bundle["model"]

    df = load_modeling_frame()
    split = np.load(os.path.join(BASE, "data", "split.npz"))
    te = split["test"]
    test = df.iloc[te].reset_index(drop=True)

    Xte = tfidf.transform(test["clean_text"].astype(str))
    test["pred"] = model.predict(Xte)

    # --- Phase 9: explainability ---
    token_map = global_tokens(tfidf, model)
    plot_tokens(token_map)

    examples = [
        "Battery stopped working after two weeks. Total waste of money.",
        "Absolutely love it, works perfectly and arrived fast. Highly recommend!",
        "It is okay. Does the job but nothing special.",
        "The product quality is poor but the support team was excellent and refunded me.",
        "Cheap material, fell apart after one wash. Very disappointed.",
        "Best purchase I have made all year, worth every penny.",
    ]
    local = local_shap(bundle, examples)
    with open(os.path.join(METRICS, "explainability.json"), "w") as f:
        json.dump({"global_top_tokens": {c: token_map[c] for c in LABELS},
                   "local_examples": local}, f, indent=2)
    print("Wrote explainability.json + 11_token_importance.png")

    # --- Phase 10: error analysis ---
    err = test[test["pred"] != test["sentiment"]].copy()
    err["word_count"] = err["reviewText"].str.split().str.len()

    def samples(true, pred, n=4):
        sub = err[(err["sentiment"] == true) & (err["pred"] == pred)]
        return [{"reviewText": t[:240], "stars_sentiment": true, "predicted": pred}
                for t in sub["reviewText"].head(n)]

    error_report = {
        "test_size": int(len(test)),
        "n_errors": int(len(err)),
        "error_rate": round(len(err) / len(test), 4),
        "confusion_pairs": err.groupby(["sentiment", "pred"]).size()
            .sort_values(ascending=False).head(8)
            .reset_index(name="count").to_dict(orient="records"),
        "false_positive_examples": samples("negative", "positive"),   # missed dissatisfaction
        "false_negative_examples": samples("positive", "negative"),   # over-flagged happy customers
        "neutral_confused_as_positive": samples("neutral", "positive"),
        "mean_word_count_errors": round(float(err["word_count"].mean()), 1),
    }
    with open(os.path.join(METRICS, "error_analysis.json"), "w") as f:
        json.dump(error_report, f, indent=2)
    print(f"Wrote error_analysis.json (error rate {error_report['error_rate']:.1%})")


if __name__ == "__main__":
    main()
