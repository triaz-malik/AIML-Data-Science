# AI-Powered Customer Recommendation & Segmentation Engine
## Business Recommendation Report

**Dataset:** Online Retail II — 776,579 clean transactions,
5,852 customers, 4,620
products, **£17,068,583** revenue (2009-12-01 →
2011-12-09).

---

### 1. The business problem
The retailer has thousands of customers and products but generic marketing, low
personalisation, and no systematic view of who its valuable customers are or what
to recommend. This engine turns raw transactions into targeted, revenue-driving
action.

### 2. Customer economics
- Revenue is highly concentrated — a small share of customers drives the majority
  of revenue (see `figures/02_customer_pareto.png`).
- The UK dominates the customer base; international markets are a growth lever
  (`figures/03_country_revenue.png`).

### 3. Customer segments (KMeans) & the playbook
| Segment | Customers | Avg Recency (d) | Avg Frequency | Avg Monetary (£) | Revenue share |
|---|---|---|---|---|---|
| VIP | 1,197 | 44 | 18.9 | 10,831 | 76.0% |
| Loyal | 1,606 | 58 | 5.0 | 1,280 | 12.0% |
| Occasional | 1,654 | 334 | 2.3 | 1,061 | 10.3% |
| At-Risk | 1,395 | 339 | 1.6 | 210 | 1.7% |

**Actions**
- **VIP** — protect and reward: early access, loyalty perks, dedicated service.
  Highest-ROI group to retain.
- **Loyal** — grow share of wallet with personalised cross-sell (see engine below).
- **Occasional** — reactivate with win-back offers timed to their purchase cadence.
- **At-Risk** — low-cost automated reminders before they churn entirely.

### 4. Recommendation engine (KNN collaborative filtering)
Tuned over distance metric × K with a held-out evaluation (20% of each customer's
basket hidden, then recovered):

- **Best config:** `cosine` distance, **K = 11**
- **Hit Rate@10 = 62.8%**, Precision@10 =
  12.8%, MAP@10 = 0.106
- Cosine similarity clearly outperforms Euclidean/Manhattan on sparse purchase
  data (`figures/09_knn_tuning.png`).

A top-10 recommendation list has been generated for **every** customer
(`data/powerbi/fact_recommendations.csv`) ready for CRM / email activation.

### 5. Products that sell together (Apriori)
| If basket contains | Also recommend | Confidence | Lift |
|---|---|---|---|
| PINK REGENCY TEACUP AND SAUCER | GREEN REGENCY TEACUP AND SAUCER | 84% | 26.9 |
| GREEN REGENCY TEACUP AND SAUCER | PINK REGENCY TEACUP AND SAUCER | 67% | 26.9 |
| ROSES REGENCY TEACUP AND SAUCER | GREEN REGENCY TEACUP AND SAUCER | 72% | 23.2 |
| GREEN REGENCY TEACUP AND SAUCER | ROSES REGENCY TEACUP AND SAUCER | 80% | 23.2 |
| TOILET METAL SIGN | BATHROOM METAL SIGN | 78% | 20.1 |
| BATHROOM METAL SIGN | TOILET METAL SIGN | 54% | 20.1 |
| ALARM CLOCK BAKELIKE RED | ALARM CLOCK BAKELIKE GREEN | 61% | 17.8 |
| ALARM CLOCK BAKELIKE GREEN | ALARM CLOCK BAKELIKE RED | 67% | 17.8 |

**Action:** bundle and co-merchandise these pairs; lift ≫ 1 means the
association is far stronger than chance — high-confidence upsell at the basket.

### 6. Customer Lifetime Value (next 6 months)
Best model: **XGBoost** — RMSE £1,511,
MAE £471, R² 0.60
(`figures/10_clv_model_comparison.png`). A forward CLV score is attached to every
customer in `dim_customer.csv`.

**SHAP — what drives future value:**
1. **Monetary** (mean |SHAP| = 0.699)
2. **Recency** (mean |SHAP| = 0.464)
3. **Frequency** (mean |SHAP| = 0.424)
4. **Tenure** (mean |SHAP| = 0.363)
5. **AvgInterpurchase** (mean |SHAP| = 0.257)

Monetary, Recency and Frequency dominate — the model independently rediscovers
RFM, validating the segmentation. **Action:** focus retention budget on customers
with high predicted CLV but rising Recency (slipping away).

### 7. Expected business impact
| Lever | Mechanism | Expected impact |
|---|---|---|
| Cross-sell | KNN + basket recommendations | +5–15% basket revenue |
| Retention | Segment-targeted win-back | Reduced VIP/Loyal churn |
| Marketing efficiency | CLV-prioritised spend | Higher ROI per £ |
| Inventory | Demand from top-product trends | Fewer stockouts/overstock |

### 8. Deliverables
- Models: `outputs/models/` (segmentation, recommender, CLV)
- Customer scores & recommendations: `data/powerbi/`
- Figures: `outputs/figures/` · Reports: `outputs/reports/`
- Power BI star schema ready for an executive dashboard.
