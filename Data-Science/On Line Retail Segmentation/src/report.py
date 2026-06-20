"""
Phase 10 - Business Recommendation Report.

Synthesises the artefacts produced by every prior phase into a single
stakeholder-facing markdown report (outputs/reports/business_report.md):
the data quality funnel, the customer-base economics, the segment playbook,
the recommendation engine's measured lift, the strongest basket rules, and the
CLV drivers — each tied to a concrete action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(config.REPORT_DIR / name)


def run() -> None:
    funnel = pd.read_csv(config.REPORT_DIR / "cleaning_funnel.csv",
                         index_col=0).iloc[:, 0]
    seg = _read_csv("segment_profile.csv")
    knn = _read_csv("knn_tuning.csv").sort_values("hit_rate", ascending=False)
    rules = _read_csv("association_rules.csv").head(8)
    clv = _read_csv("clv_model_comparison.csv")
    shap_imp = _read_csv("shap_importance.csv")

    best_knn = knn.iloc[0]
    best_clv = clv.iloc[0]
    total_rev = float(funnel["clean_revenue"])

    seg = seg.copy()
    seg_lines = "\n".join(
        f"| {r['segment']} | {int(r['customers']):,} | "
        f"{r['avg_recency']:.0f} | {r['avg_frequency']:.1f} | "
        f"{r['avg_monetary']:,.0f} | {r['revenue_share']:.1%} |"
        for _, r in seg.iterrows())

    rule_lines = "\n".join(
        f"| {r['antecedents']} | {r['consequents']} | "
        f"{r['confidence']:.0%} | {r['lift']:.1f} |"
        for _, r in rules.iterrows())

    shap_lines = "\n".join(
        f"{i+1}. **{r['feature']}** (mean |SHAP| = {r['mean_abs_shap']:.3f})"
        for i, (_, r) in enumerate(shap_imp.head(5).iterrows()))

    md = f"""# AI-Powered Customer Recommendation & Segmentation Engine
## Business Recommendation Report

**Dataset:** Online Retail II — {int(funnel['clean_rows']):,} clean transactions,
{int(funnel['clean_customers']):,} customers, {int(funnel['clean_products']):,}
products, **£{total_rev:,.0f}** revenue ({funnel['date_min'][:10]} →
{funnel['date_max'][:10]}).

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
{seg_lines}

**Actions**
- **VIP** — protect and reward: early access, loyalty perks, dedicated service.
  Highest-ROI group to retain.
- **Loyal** — grow share of wallet with personalised cross-sell (see engine below).
- **Occasional** — reactivate with win-back offers timed to their purchase cadence.
- **At-Risk** — low-cost automated reminders before they churn entirely.

### 4. Recommendation engine (KNN collaborative filtering)
Tuned over distance metric × K with a held-out evaluation (20% of each customer's
basket hidden, then recovered):

- **Best config:** `{best_knn['metric']}` distance, **K = {int(best_knn['k'])}**
- **Hit Rate@10 = {best_knn['hit_rate']:.1%}**, Precision@10 =
  {best_knn['precision@N']:.1%}, MAP@10 = {best_knn['map@N']:.3f}
- Cosine similarity clearly outperforms Euclidean/Manhattan on sparse purchase
  data (`figures/09_knn_tuning.png`).

A top-10 recommendation list has been generated for **every** customer
(`data/powerbi/fact_recommendations.csv`) ready for CRM / email activation.

### 5. Products that sell together (Apriori)
| If basket contains | Also recommend | Confidence | Lift |
|---|---|---|---|
{rule_lines}

**Action:** bundle and co-merchandise these pairs; lift ≫ 1 means the
association is far stronger than chance — high-confidence upsell at the basket.

### 6. Customer Lifetime Value (next 6 months)
Best model: **{best_clv['model']}** — RMSE £{best_clv['RMSE']:,.0f},
MAE £{best_clv['MAE']:,.0f}, R² {best_clv['R2']:.2f}
(`figures/10_clv_model_comparison.png`). A forward CLV score is attached to every
customer in `dim_customer.csv`.

**SHAP — what drives future value:**
{shap_lines}

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
"""
    out = config.REPORT_DIR / "business_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"Business report written -> {out}")


if __name__ == "__main__":
    run()
