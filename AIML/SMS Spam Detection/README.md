# AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

An end-to-end machine-learning system that classifies inbound SMS traffic into
five risk categories — **Normal, Promotion, Spam, Phishing, Fraud** — so telecom
operators can detect and block risky messages *before* customers become victims.

> Built with classic ML (TF-IDF + Logistic Regression) **and** transformer
> models (DistilBERT, BERT), with SHAP explainability, error analysis, and
> Power BI–ready exports.

---

## 1. Business Problem

Telecom operators route **millions of SMS messages daily**. Hidden in that
traffic are banking scams, fake OTP requests, investment/lottery fraud, fake
delivery notices, and aggressive promotional spam. Each one erodes customer
trust and drives complaints, churn, and fraud-related losses.

**Goal:** automatically detect risky messages in real time and route them for
blocking, warning, or review.

### Business Questions
1. What percentage of messages are fraudulent / phishing / promotional?
2. Which fraud type is most common?
3. What keywords are most associated with scams?
4. Can AI accurately identify phishing attempts?
5. Which messages should be blocked?

---

## 2. Data

| Source | Role | Notes |
|--------|------|-------|
| **UCI SMS Spam Collection** | Primary, clean ground truth | 5,574 messages, `ham`/`spam` |
| **Mishra & Soni Smishing corpus** | Phishing/fraud seed | filtered to de-noised smishing examples |

After de-duplication: **8,103 unique messages** (≈ 4,518 ham / 3,585 spam).

### From 2 classes to 5 — weak supervision
Public datasets are only labelled `ham`/`spam`. The finer
`Normal / Promotion / Spam / Phishing / Fraud` scheme is derived with
**transparent, priority-ordered rules** (most-harmful-first:
Fraud → Phishing → Promotion → Spam; `ham` → Normal) using curated keyword sets,
URL detection, and the smishing-corpus prior.

> ⚠️ **Honesty note:** the fine-grained labels are *heuristically derived*, not
> human-annotated. The binary `ham`/`spam` label remains true ground truth and
> is preserved alongside every record. This is documented as a known limitation.

---

## 3. Pipeline

```
01_acquire_data.py        → data/raw/sms_raw.csv
02_label_5class.py        → data/processed/sms_labeled.csv      (weak 5-class labels)
03_eda.py                 → outputs/figures/01..04              (EDA visuals)
04_feature_engineering.py → data/processed/sms_features.csv     (+ engineered features)
05_model_logreg.py        → outputs/models/logreg.joblib        (baseline + tuning)
06_model_transformers.py  → reports/metrics_{distilbert,bert}.json
07_explain_errors.py      → SHAP + error analysis
08_business_powerbi.py    → outputs/powerbi/*.csv               (dashboard data)
predict.py                → inference CLI (score new messages)
```

Run everything:
```bash
python run_all.py            # full pipeline (GPU recommended)
python run_all.py --fast     # skip transformer fine-tuning
```

### Try it on a new message
```bash
python src/predict.py "Your account is suspended. Verify at http://bit.ly/x"
python src/predict.py        # built-in demo set
```
Example output:
```
MESSAGE : URGENT! You have WON a £1000 cash prize! Call 09061701461 to claim now.
PREDICT : Fraud  (confidence 1.00)  ->  BLOCK
MESSAGE : Your bank account has been suspended. Verify at http://secure-bank.ng/login
PREDICT : Phishing  (confidence 0.98)  ->  BLOCK
MESSAGE : 50% OFF everything this weekend! Reply STOP to unsubscribe.
PREDICT : Promotion  (confidence 0.99)  ->  REVIEW
```
Actions map risk → operator response: **ALLOW / REVIEW / WARN / BLOCK**.

---

## 4. Exploratory Data Analysis

Figures in `outputs/figures/`:
- **01_class_distribution** — the fraud/phishing/spam landscape
- **02_message_length** — Normal messages are short (~53 chars); scam/promo
  messages are ~3× longer
- **03_wordclouds** — per-class vocabulary (Spam: *free, win, prize*; Phishing:
  *account, verify, bank*; Fraud: *won, claim, cash*)
- **04_top_ngrams** — top unigrams/bigrams/trigrams across malicious messages

---

## 5. Feature Engineering

Hand-crafted signals (`src/features.py`) combined with TF-IDF:
`msg_length, char_count, word_count, n_urls, n_digits, digit_ratio,
n_currency, n_uppercase, uppercase_ratio, n_exclaim, n_special, has_phone`.

Strongest correlations with "is malicious": **char_count (0.62), msg_length
(0.59), n_digits (0.56), n_urls (0.52)**. Phishing leads on URLs; Fraud leads on
currency symbols.

---

## 6. Models & Results

Three models, identical stratified 80/20 split (6,482 train / 1,621 test),
class-weighted for imbalance.

| Model | 5-class Accuracy | 5-class Macro-F1 | Binary Accuracy | Binary F1 |
|-------|:---------------:|:----------------:|:---------------:|:---------:|
| TF-IDF + Logistic Regression | 95.1% | 0.877 | 98.2% | 0.979 |
| DistilBERT (fine-tuned) | 95.7% | 0.869 | **99.1%** | **0.990** |
| **BERT (fine-tuned)** | **96.2%** | **0.898** | 99.1% | 0.990 |

> **BERT** is the best overall (highest 5-class accuracy and macro-F1, with the
> biggest gains on the hard minority classes Promotion & Spam). DistilBERT
> matches it on binary ham/spam at a fraction of the size. The classic
> TF-IDF + LogReg baseline is within ~1 point while being far cheaper and fully
> interpretable.

Per-class F1 (BERT): Normal 0.99 · Phishing 0.95 · Fraud 0.94 · Spam 0.81 ·
Promotion 0.80.

- **Model 1 — TF-IDF + Logistic Regression**: word (1–2gram) + char (3–4gram)
  TF-IDF **plus** engineered numeric features; 5-fold CV grid search over `C`
  and `l1_ratio` (best: `C=5.0, l1_ratio=1.0` → pure L1). Fast & interpretable.
- **Model 2 — DistilBERT** (fine-tuned, GPU, ~80 s on RTX 5080).
- **Model 3 — BERT** (fine-tuned, GPU).

Confusion matrices: `outputs/figures/06_logreg_confusion.png`,
`07_distilbert_confusion.png`, `07_bert_confusion.png`.

---

## 7. Explainability (SHAP) & Error Analysis

- **Global drivers** (`08_global_top_words.png`): the words that most push each
  class decision.
- **Local explanation** (`09_shap_example.png`): e.g. *"Your account is
  suspended. Verify immediately."* → flagged via **verify / account / suspended**.
- **Error analysis** (`reports/error_analysis.md`): false positives (legit
  messages that look risky, e.g. *"Your bank statement is ready"*) and false
  negatives (risky messages that look normal).

---

## 8. Business Insights & Recommendations

Generated into `outputs/powerbi/business_findings.csv`. Examples:
- A large share of phishing messages contain a **URL** → increase URL screening.
- Most fraud/phishing messages use **banking keywords** → deploy banking-specific
  filters and step-up verification.
- `verify` / `account` / `OTP` strongly predict fraud → real-time keyword alerts.

### Power BI dashboards (data in `outputs/powerbi/`)
| Dashboard | Pages | Source CSV |
|-----------|-------|------------|
| **Executive** | Total messages, Spam %, Fraud %, Phishing % | `class_summary.csv` |
| **Fraud** | Top scam types, risk tiers, keyword analysis | `fraud_keywords.csv`, `feature_by_class.csv` |
| **AI** | Model accuracy, predictions, confidence scores | `model_comparison.csv`, `messages_scored.csv` |

---

## 9. Business Value (for telecom operators)
- Reduce subscriber fraud and fraud-related losses
- Protect customers and improve trust
- Reduce complaints and churn
- Real-time, explainable, auditable risk scoring

---

## 10. Project Structure
```
SMS Spam Detection/
├── data/{raw,processed}/
├── src/                  # 01..08 pipeline + features.py, utils.py
├── outputs/{figures,models,powerbi}/
├── reports/              # metrics_*.json, eda_summary.md, error_analysis.md
├── run_all.py
└── requirements.txt
```

## 11. Limitations
- Fine-grained labels are weak (rule-derived), not human-annotated.
- The smishing corpus is Nigerian-telecom heavy and noisy; only de-noised
  smishing seeds were kept.
- Class imbalance (Promotion/Spam are small) limits per-class recall on the
  smallest classes — mitigated with class weights and macro-F1 reporting.
