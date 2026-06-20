# AI-Powered Customer Review Sentiment Analysis (BERT / DistilBERT)

End-to-end, business-focused sentiment analysis on **Amazon Reviews 2023** (5 product
categories, ~235k reviews after cleaning). Classifies reviews into **negative /
neutral / positive** and surfaces *why* customers are dissatisfied — for automated
review monitoring, defect detection, and product-quality prioritisation.

## Business questions
- What % of reviews are positive / negative?
- Which categories & products generate the most negative sentiment?
- What words/phrases drive negative reviews?
- Can we predict sentiment accurately enough to automate monitoring?

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 0. Download | `src/download_data.py` | `data/raw.parquet` (5 cats × 50k, streamed from HF) |
| 1+3. Prep & label | `src/data_prep.py` | `data/clean.parquet`, `data_summary.json` |
| 2. EDA | `src/eda.py` | rating/sentiment/length/category figures, word clouds, n-grams |
| 4. Features | `src/features.py` | `data/features.parquet` (clean_text, VADER, lengths…) |
| 5,7,8. Baseline | `src/train_baseline.py` | TF-IDF + LogReg, `split.npz`, `baseline_metrics.json`, `models/baseline.joblib` |
| 6,7,8. Transformers | `src/train_transformer.py` | fine-tuned DistilBERT + BERT, `transformer_metrics.json`, `models/` |
| 9,10. Explain & errors | `src/explain.py` | token importance, SHAP examples, `error_analysis.json` |
| 11. Insights | `src/insights.py` | `business_insights.json`, model-comparison figure |
| 12. Dashboard | `src/dashboard.py` | `streamlit run src/dashboard.py` |

All models share **one** stratified test split (`data/split.npz`) for a fair comparison.

## Run order
```powershell
python src\download_data.py
python src\data_prep.py
python src\eda.py
python src\features.py
python src\train_baseline.py
python src\train_transformer.py   # needs CUDA GPU
python src\explain.py
python src\insights.py
streamlit run src\dashboard.py
```

## Labeling
- **3-class** (default): 1–2★ → negative, 3★ → neutral, 4–5★ → positive
- **Binary**: drop 3★, 1–2★ → negative, 4–5★ → positive (`sentiment_bin` column)

## Notes
- Data is heavily positive-skewed (~83% positive), so the baseline tunes
  `class_weight` and we report **macro-F1** + **ROC-AUC** alongside accuracy.
- `download_data.py` streams only the first ~70k lines per category (shuffled),
  so it never downloads the multi-GB full files.
- Transformers were trained on an NVIDIA RTX GPU (fp16). `TRAIN_CAP` in
  `train_transformer.py` subsamples the train split for a tractable run.
