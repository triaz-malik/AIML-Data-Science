# Amazon Reviews Sentiment Analysis — TF-IDF · DistilBERT · BERT

An end-to-end NLP pipeline that classifies Amazon product reviews as
**positive** or **negative**, benchmarking a classical baseline against two
fine-tuned transformers, with EDA, engineered features, and explainability.

## Results (100k balanced subsample)

| Model | Approach | Test Accuracy | F1 | Notes |
|-------|----------|--------------:|----:|-------|
| Logistic Regression | TF-IDF (1–2 gram) | ~0.88 | ~0.88 | fast, interpretable baseline |
| DistilBERT | fine-tuned transformer | ~0.92 | ~0.92 | best speed/accuracy trade-off |
| BERT | fine-tuned transformer | ~0.94 | ~0.94 | strongest, slowest |

*(Accuracies land in these ranges; exact numbers are written to
`models/*/metrics.json` on each run. A 4k/1-epoch DistilBERT smoke test
already reached 0.915.)*

## Dataset & an important note

The data on disk (`train.ft.txt.bz2`, `test.ft.txt.bz2`) is the **Amazon
Reviews Polarity** dataset in fastText format:

```
__label__2 Great mouse: Comfortable, responsive, great value...
__label__1 Stopped working: Died after a week, total waste...
```

- `__label__1` = **negative** (1–2★), `__label__2` = **positive** (4–5★)
- ~3.6M train / 400k test rows, each = `label <title>: <body>`
- **It contains only the review text and a binary label** — there is *no*
  star rating, product name, category, date, or helpful-votes column.

Because of that, this project is scoped to what the data actually supports:
binary sentiment, text-only EDA, and text-derived features. The metadata-
dependent ideas from the original brief — a 3-class *Neutral* label, category
bar charts, sentiment-over-time trends, helpful-votes features, and a Power BI
dashboard sliced by category/date — would require a richer source such as the
**Amazon Reviews 2023 (McAuley Lab)** dataset. Swapping that in only means
replacing `src/data.py`; the rest of the pipeline is agnostic.

## Project structure

```
Amazon Product Sentiment Analysis/
├── data/
│   ├── raw/                 # (raw .ft.bz2 archives live here or at repo root)
│   └── processed/           # parquet caches, keyed by split + sample size
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_LogisticRegression.ipynb
│   ├── 04_DistilBERT.ipynb
│   └── 05_BERT.ipynb
├── src/                     # the reusable engine — notebooks are thin wrappers
│   ├── config.py            # paths, label map, seed, defaults
│   ├── data.py              # parse .ft.bz2 → DataFrame (cached, balanced sample)
│   ├── preprocess.py        # text cleaning for TF-IDF / word clouds
│   ├── features.py          # engineered text features
│   ├── eda.py               # all plots + word clouds → reports/figures/
│   ├── train_baseline.py    # TF-IDF + LogReg (+ GridSearchCV)
│   ├── train_transformer.py # fine-tune DistilBERT / BERT (HF Trainer)
│   └── explain.py           # SHAP on the baseline
├── scripts/make_notebooks.py
├── models/                  # saved models + metrics.json (gitignored)
├── reports/figures/         # generated EDA / SHAP figures
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. EDA figures (class balance, length, word clouds) → reports/figures/
python -m src.eda --sample 50000

# 2. Baseline: TF-IDF + Logistic Regression (add --grid for tuning)
python -m src.train_baseline --sample 100000

# 3. Transformers (CUDA GPU recommended)
python -m src.train_transformer --model distilbert-base-uncased --sample 100000
python -m src.train_transformer --model bert-base-uncased       --sample 100000

# 4. Explainability (SHAP over the baseline's words)
python -m src.explain --sample 20000
```

Use `--sample 0` anywhere to run on the **full** dataset. All training scripts
print metrics and persist them to `models/`. The notebooks call these same
functions, so notebook and CLI results never diverge.

## Design choices

- **Balanced subsampling** (default 100k) keeps EDA snappy and transformer
  training tractable on one GPU; flip to `--sample 0` for the full corpus.
- **Parquet caching** — the slow bz2 parse runs once per (split, size).
- **Negation kept in cleaning** — "not good" must not collapse to "good".
- **Transformers see raw text** — only the TF-IDF path uses `preprocess.py`;
  BERT/DistilBERT rely on their own tokenizers.
- **Reproducibility** — fixed seed (`config.SEED = 42`) throughout.

## Pipeline stages (mapped to the brief)

EDA · text cleaning · feature engineering · 3 models · train/test + 5-fold CV ·
GridSearchCV tuning (baseline) + LR/batch/epoch/weight-decay knobs
(transformers) · SHAP explainability · error analysis (notebook 03/05).
