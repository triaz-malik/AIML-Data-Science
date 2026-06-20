# AI-Powered Customer Recommendation & Segmentation Engine

End-to-end retail analytics on the **Online Retail II** dataset (~1M transactions,
a UK online retailer, Dec 2009 – Dec 2011). The project turns raw transactions
into targeted, revenue-driving action: it segments customers, recommends products,
mines basket associations, predicts customer lifetime value, and explains the
"why" behind every score — then ships the results as a Power BI star schema.

> **Business framing:** this is positioned as a retail analytics *solution*, not a
> single-model notebook. Every phase ends in a concrete business action.

## Results at a glance
| Phase | Output | Headline result |
|---|---|---|
| Cleaning | 776,579 clean sale rows from 1.07M | transparent data-quality funnel |
| EDA | revenue/customer/time charts | top 20% of customers ≈ 77% of revenue; UK ≈ 84% |
| Segmentation | KMeans (K=4) + DBSCAN | VIP segment = ~1,200 customers ≈ 76% of revenue |
| **Recommender** | **User-based KNN CF** | **Hit Rate@10 ≈ 0.63, Precision@10 ≈ 0.13** (cosine, K=11) |
| Market basket | Apriori rules | matching teacup pairs at **lift ≈ 27** |
| CLV | XGBoost vs RF vs LightGBM | XGBoost **R² ≈ 0.60** on 6-month forward revenue |
| Explainability | SHAP | Monetary / Recency / Frequency drive value (RFM rediscovered) |

*(Recommendation metrics use a leakage-free held-out evaluation — see
`src/recommendation.py`.)*

## Architecture
```
online_retail_II.csv
  → Phase 2  Data Cleaning          src/data_cleaning.py
  → Phase 3  EDA                    src/eda.py
  → Phase 4  RFM Features           src/features.py
  → Phase 5  Segmentation           src/segmentation.py   (KMeans + DBSCAN)
  → Phase 6  KNN Recommender        src/recommendation.py (core)
  → Phase 7  Market Basket          src/market_basket.py  (Apriori)
  → Phase 8  CLV Prediction         src/clv.py            (RF/XGB/LGBM)
  → Phase 9  SHAP Explainability    src/explainability.py
  → Phase 10 Power BI + Report      src/export_powerbi.py, src/report.py
```

## Quick start
```bash
pip install -r requirements.txt

# run the whole pipeline
python run_pipeline.py

# or run phases individually
python src/data_cleaning.py
python src/recommendation.py

# run a slice
python run_pipeline.py --from features --to clv
python run_pipeline.py --only market_basket
```

All paths are configured in `config.py`. Outputs land in:
- `data/processed/` — cleaned parquet, customer features, recommendations
- `data/powerbi/` — star-schema CSVs (`fact_sales`, `dim_customer`, `dim_product`,
  `dim_date`, `fact_recommendations`, `association_rules`)
- `outputs/figures/` — all charts (EDA, clusters, KNN tuning, SHAP)
- `outputs/models/` — saved `joblib` models
- `outputs/reports/` — metrics CSVs + `business_report.md`

## Power BI dashboard
Load the CSVs from `data/powerbi/` and relate them as a star schema:
- `fact_sales[CustomerID] → dim_customer[CustomerID]`
- `fact_sales[StockCode] → dim_product[StockCode]`
- `fact_sales[DateKey] → dim_date[DateKey]`
- `fact_recommendations[CustomerID] → dim_customer[CustomerID]`

Suggested pages: **Executive** (revenue/orders/customers KPIs), **Segmentation**
(segment sizes & revenue share, PredictedCLV_6m), **Recommendations** (per-customer
top-10), **Market Basket** (association rules by lift).

## Tech
Python · pandas · scikit-learn · XGBoost · LightGBM · mlxtend (Apriori) · SHAP ·
matplotlib/seaborn · Power BI.

## Notes & honesty
- Returns/cancellations and non-product stock codes (postage, fees, manual
  adjustments) are removed before modelling; the funnel reports every dropped row.
- The recommender is evaluated with the target customer's own row excluded, so the
  reported metrics are not inflated by leakage.
- CLV uses a calibration (12-month) / holdout (6-month) split to avoid look-ahead.
