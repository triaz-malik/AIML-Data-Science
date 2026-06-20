"""Generate the five project notebooks as valid .ipynb files.

The notebooks are intentionally thin — they import from `src/` and call the
same functions the command-line scripts use, so notebook output and script
output never drift apart. Run once:

    python scripts/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

# Prepend so `import src...` resolves when the notebook runs from notebooks/.
BOOTSTRAP = (
    "import sys, os\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "%matplotlib inline"
)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": text}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells):
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(f"wrote {path}")


# --- 01 EDA ----------------------------------------------------------------
write("01_EDA.ipynb", [
    md("# 01 · Exploratory Data Analysis\n\n"
       "Amazon Reviews Polarity dataset (binary: positive / negative). "
       "We inspect class balance, review length, simple text features, and "
       "word clouds. All plotting lives in `src/eda.py` so it is reproducible."),
    code(BOOTSTRAP),
    code("from src import data, eda, features, config\n"
         "df = data.load_split('train', sample_size=50_000)\n"
         "df.shape, df['label'].map(config.LABEL_NAMES).value_counts().to_dict()"),
    md("### Sample reviews"),
    code("df[['label','title','body']].head()"),
    md("### Class balance"),
    code("eda.plot_class_balance(df)\nfrom IPython.display import Image\nImage('../reports/figures/01_class_balance.png')"),
    md("### Review length by sentiment"),
    code("eda.plot_review_length(df)\nImage('../reports/figures/02_review_length.png')"),
    md("### Engineered text features (exclamation marks, uppercase ratio)"),
    code("feat = features.add_text_features(df)\n"
         "feat.groupby(df['label'].map(config.LABEL_NAMES))[features.FEATURE_COLS].mean().round(2)"),
    code("eda.plot_feature_box(df)\nImage('../reports/figures/03_feature_boxplots.png')"),
    md("### Word clouds"),
    code("eda.plot_wordclouds(df)\nImage('../reports/figures/04_wordcloud_positive.png')"),
    code("Image('../reports/figures/05_wordcloud_negative.png')"),
    md("### Top words"),
    code("print('positive:', eda.top_words(df, config.POS, 15))\n"
         "print('negative:', eda.top_words(df, config.NEG, 15))"),
])

# --- 02 Preprocessing ------------------------------------------------------
write("02_Preprocessing.ipynb", [
    md("# 02 · Text Preprocessing\n\n"
       "Cleaning for the **classical** pipeline: lowercase → strip URLs/HTML → "
       "remove punctuation/stopwords → lemmatize. Negation words are kept "
       "(\"not good\" must stay distinct from \"good\").\n\n"
       "> Transformers use their own tokenizer on raw text — this cleaning is "
       "only for TF-IDF and word clouds."),
    code(BOOTSTRAP),
    code("from src import data, preprocess\npreprocess.ensure_nltk()\n"
         "df = data.load_split('train', sample_size=5_000)"),
    md("### Before / after on a few examples"),
    code("for t in df['text'].head(4):\n"
         "    print('RAW :', t[:120])\n"
         "    print('CLEAN:', preprocess.clean_text(t)[:120])\n    print('-'*60)"),
    md("### Apply to the sample and cache the cleaned column"),
    code("df['clean'] = preprocess.clean_series(df['text'].tolist())\n"
         "df[['label','clean']].head()"),
])

# --- 03 Logistic Regression ------------------------------------------------
write("03_LogisticRegression.ipynb", [
    md("# 03 · Baseline — TF-IDF + Logistic Regression\n\n"
       "Fast, interpretable yardstick. Training logic is in "
       "`src/train_baseline.py`. Expected accuracy ≈ 85–88%."),
    code(BOOTSTRAP),
    code("from src import train_baseline\n"
         "# set do_grid=True to run 5-fold GridSearchCV over C\n"
         "metrics = train_baseline.train(sample_size=100_000, do_grid=False)\nmetrics"),
    md("### Most influential words (model coefficients)\n"
       "Positive vs. negative drivers — a cheap, honest explainability check."),
    code("import joblib, numpy as np\n"
         "from src import config\n"
         "model = joblib.load(config.MODELS_DIR / 'logistic.pkl')\n"
         "vec, clf = model.named_steps['tfidf'], model.named_steps['clf']\n"
         "names = np.array(vec.get_feature_names_out())\n"
         "coef = clf.coef_[0]\n"
         "top_pos = names[np.argsort(coef)[-20:]][::-1]\n"
         "top_neg = names[np.argsort(coef)[:20]]\n"
         "print('PUSH POSITIVE:', list(top_pos))\nprint('PUSH NEGATIVE:', list(top_neg))"),
    md("### Try it on your own text"),
    code("samples = ['Absolutely love this, works perfectly!',\n"
         "           'Broke after two days, complete waste of money.']\n"
         "for s in samples:\n"
         "    pred = model.predict([s])[0]\n"
         "    print(config.LABEL_NAMES[pred], '<-', s)"),
])

# --- 04 DistilBERT ---------------------------------------------------------
write("04_DistilBERT.ipynb", [
    md("# 04 · DistilBERT\n\n"
       "Fine-tune `distilbert-base-uncased`. Lighter/faster than BERT, "
       "expected accuracy ≈ 91–93%. Needs a CUDA GPU for reasonable speed.\n\n"
       "Logic is in `src/train_transformer.py` (also runnable as a script:\n"
       "`python -m src.train_transformer --model distilbert-base-uncased --sample 100000`)."),
    code(BOOTSTRAP),
    code("import torch; print('CUDA available:', torch.cuda.is_available())"),
    code("from src import train_transformer\n"
         "metrics = train_transformer.train(\n"
         "    model_name='distilbert-base-uncased',\n"
         "    sample_size=100_000, epochs=2, batch_size=32,\n"
         "    lr=2e-5, max_len=256, weight_decay=0.01,\n)\nmetrics"),
    md("### Inference with the fine-tuned model"),
    code("from transformers import pipeline\nfrom src import config\n"
         "clf = pipeline('text-classification', model=str(config.MODELS_DIR / 'distilbert'), device=0 if torch.cuda.is_available() else -1)\n"
         "clf(['This is the best purchase I have made all year.',\n"
         "     'Terrible quality and it arrived broken.'])"),
])

# --- 05 BERT ---------------------------------------------------------------
write("05_BERT.ipynb", [
    md("# 05 · BERT\n\n"
       "Fine-tune `bert-base-uncased` — the strongest of the three, expected "
       "accuracy ≈ 93–95%. Same code path as DistilBERT, larger model."),
    code(BOOTSTRAP),
    code("from src import train_transformer\n"
         "metrics = train_transformer.train(\n"
         "    model_name='bert-base-uncased',\n"
         "    sample_size=100_000, epochs=2, batch_size=16,\n"
         "    lr=2e-5, max_len=256, weight_decay=0.01,\n)\nmetrics"),
    md("### Compare all three models\n"
       "Reads the metrics JSON each trainer writes and tabulates them."),
    code("import json, pandas as pd\nfrom src import config\n"
         "rows = []\n"
         "for f, name in [('logistic_metrics.json','tfidf_logreg'),\n"
         "                ('distilbert/metrics.json','distilbert'),\n"
         "                ('bert/metrics.json','bert')]:\n"
         "    p = config.MODELS_DIR / f\n"
         "    if p.exists():\n        rows.append(json.loads(p.read_text()))\n"
         "pd.DataFrame(rows)[['model','accuracy','f1']].sort_values('accuracy')"),
])

print("\nAll notebooks generated.")
