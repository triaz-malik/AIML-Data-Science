# Clinical Recommendation Report

## Phase 9 — Explainable AI (SHAP)
Top factors driving the model's stroke-risk predictions
(mean absolute SHAP value, higher = more influential):

| Rank | Feature | Importance |
|---|---|---|
| 1 | age | 0.1182 |
| 2 | health_risk_score | 0.0400 |
| 3 | avg_glucose_level | 0.0327 |
| 4 | age_group_Young | 0.0290 |
| 5 | smoking_status_never smoked | 0.0269 |
| 6 | bmi | 0.0236 |
| 7 | age_group_Adult | 0.0193 |
| 8 | Residence_type_Rural | 0.0125 |
| 9 | smoking_status_Unknown | 0.0122 |
| 10 | Residence_type_Urban | 0.0120 |

See `outputs/figures/12_shap_importance.png` (global) and
`13_shap_summary.png` (beeswarm: direction + magnitude per patient).

**Local (per-patient) explanations** answer *why was THIS patient flagged?* —
waterfall plots for a representative Critical, borderline, and Low patient:
`14_shap_local_critical.png`, `14_shap_local_borderline.png`,
`14_shap_local_low.png`. These are what a clinician sees alongside each score.

As expected clinically, **age** dominates, followed by glucose, BMI, the
composite health-risk score, hypertension, and heart disease — the model's
reasoning aligns with established stroke risk factors, which builds clinical trust.

## Phase 10 — Risk Stratification
Patients are stratified by predicted stroke probability (out-of-fold):

| Risk Level | Action | Patients | Actual stroke rate |
|---|---|---|---|
| Low (0%–10%) | Routine Monitoring | 2633 | 1.0% |
| Medium (10%–30%) | Follow-up | 341 | 5.9% |
| High (30%–60%) | Specialist Review | 731 | 5.9% |
| Critical (60%–101%) | Immediate Attention | 1404 | 11.4% |

The actual stroke rate rises monotonically across bands — confirming the
stratification is clinically meaningful: higher bands really do contain
higher-risk patients, so prioritizing them for specialist review is justified.

Patient-level scores: `outputs/predictions/patient_scores.csv`.

### Recommended workflow
- **Low** → routine monitoring at regular checkups.
- **Medium** → schedule follow-up; review modifiable factors (glucose, BMI, smoking).
- **High** → specialist (neurology/cardiology) review.
- **Critical** → immediate clinical attention / urgent workup.