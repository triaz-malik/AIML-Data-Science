# Network Intrusion Detection System (IDS) — ML & DL

Binary classifier that labels network connections as **normal** or **anomaly (attack)**
on the NSL-KDD dataset, with hyperparameter tuning, SMOTE/ADASYN, SHAP explainability,
a Streamlit dashboard, and an auto-generated PDF report.

## Results (held-out 20% test split)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|---------:|----------:|-------:|----:|--------:|
| **XGBoost** (best) | 0.9976 | 0.9987 | 0.9962 | **0.9974** | 1.0000 |
| Random Forest | 0.9962 | 0.9983 | 0.9936 | 0.9959 | 0.9999 |
| Neural Net (MLP) | 0.9931 | 0.9966 | 0.9885 | 0.9925 | 0.9988 |
| Logistic Regression | 0.9734 | 0.9772 | 0.9655 | 0.9713 | 0.9944 |

Top SHAP drivers: `src_bytes`, `count`, `dst_bytes`, `dst_host_same_src_port_rate`, `service`, `protocol_type`.

## Important data notes
- The provided **`Test_data.csv` is unlabelled** — all metrics above come from a stratified
  80/20 split of `Train_data.csv` plus 5-fold cross-validation. `Test_data.csv` is used only
  to generate `outputs/test_predictions.csv`.
- These files contain **binary labels only** (`normal` / `anomaly`). The multi-class attack
  subtypes (DoS / Probe / U2R / R2L) from the full NSL-KDD release are not present, so this
  project is binary. The pipeline would extend to multi-class by swapping the label mapping
  in `src/preprocessing.py`.
- Data is roughly balanced (~47% attack), so SMOTE/ADASYN give only a marginal lift (see the
  comparison in `models/metrics.json` and the report) — included for completeness.

## Project layout
```
src/
  config.py          paths, schema, constants
  preprocessing.py   load / clean / one-hot + scale / stratified split
  eda.py             EDA figures + printed findings
  train_models.py    tune + train 4 models, SMOTE/ADASYN compare, K-fold, save .pkl
  evaluate.py        confusion matrices, ROC, PR curves, comparison table
  explain_shap.py    SHAP summary / bar / waterfall
  predict_test.py    score the unlabelled Test_data.csv
  make_report.py     assemble reports/IDS_Report.pdf
app/streamlit_app.py Streamlit dashboard (single connection + batch CSV)
run_pipeline.py      run everything end-to-end
models/              saved models + preprocessor + metrics.json
reports/             figures/, model_comparison.csv, IDS_Report.pdf
outputs/             test_predictions.csv
```

## Setup
```powershell
pip install -r requirements.txt
```
(Uses scikit-learn's MLPClassifier for the neural net — no TensorFlow needed.)

## Run
```powershell
python run_pipeline.py                 # full pipeline: EDA -> train -> eval -> SHAP -> predict -> report
# or step-by-step:
python src/eda.py
python src/train_models.py
python src/evaluate.py
python src/explain_shap.py
python src/predict_test.py
python src/make_report.py

streamlit run app/streamlit_app.py     # interactive dashboard
```

## Dashboard
- **Single connection** tab: edit key features (protocol, service, bytes, counts, rates) →
  Normal/Attack verdict, attack probability, and a 0–100 risk score.
- **Batch CSV** tab: upload connections → scored table + downloadable predictions.

Reproducibility: fixed `random_state=42`, deterministic split shared across all scripts.
