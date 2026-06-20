"""Generate a professional multi-page PDF report for the Ames House Prices
(Advanced Regression) project.

All metrics are recovered from House_Prices_Report.docx (the project's own
narrative report) and verified against house_prices.py. No numbers are
fabricated. Every figure embed is guarded by an existence check.

Run:  python make_report.py
Output:  House_Prices_Report.pdf  (next to House_Prices_Report.docx)
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

# BASE derived from this script's location -> top-level project root.
BASE = Path(__file__).resolve().parent
FIGURE_DIR = BASE / "figures"
REPORT_PATH = BASE / "House_Prices_Report.pdf"

# --- Brand palette ---------------------------------------------------------
NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#2e6da4")
LIGHT = colors.HexColor("#eaf1f8")
GREY = colors.HexColor("#555555")
WIN = colors.HexColor("#d4edda")

# --- Styles ----------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=30,
                          textColor=NAVY, leading=36, spaceAfter=12))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=14,
                          textColor=GREY, alignment=TA_CENTER, leading=20))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17,
                          textColor=NAVY, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                          textColor=BLUE, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5,
                          leading=15, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle("Bul", parent=styles["Body"], leftIndent=14,
                          bulletIndent=4, spaceAfter=2))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("KPI", parent=styles["Normal"], fontSize=22,
                          textColor=NAVY, alignment=TA_CENTER, leading=24))
styles.add(ParagraphStyle("KPILabel", parent=styles["Normal"], fontSize=9,
                          textColor=GREY, alignment=TA_CENTER))


def p(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", styles["Bul"]) for t in items]


def fig(name, width=15 * cm, caption=None):
    """Return flowables for a figure if it exists, else an empty list."""
    path = FIGURE_DIR / name
    if not path.exists():
        return []
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    out = [Spacer(1, 4), img]
    if caption:
        out.append(p(caption, "Caption"))
    else:
        out.append(Spacer(1, 8))
    return out


def hr():
    return HRFlowable(width="100%", thickness=1, color=LIGHT,
                      spaceBefore=6, spaceAfter=10)


# --- Recovered leaderboard (docx Table 5, verified against house_prices.py) -
# Model, OOF RMSLE, vs Ridge baseline
LEADERBOARD = [
    ("Stacker (Ridge meta-learner)", "0.10740", "+5.4%"),
    ("Lasso", "0.11179", "+1.5%"),
    ("ElasticNet", "0.11182", "+1.5%"),
    ("CatBoost (tuned)", "0.11318", "+0.3%"),
    ("Ridge", "0.11351", "baseline"),
    ("GBM", "0.11395", "-0.4%"),
    ("XGBoost (tuned)", "0.11556", "-1.8%"),
    ("LightGBM (tuned)", "0.11602", "-2.2%"),
]
BEST_RMSLE = "0.10740"          # Stacker OOF RMSLE
BEST_SINGLE = ("CatBoost", "0.11318")


def kpi_row():
    """Headline KPI cards built only from recovered numbers."""
    cards = [
        (BEST_RMSLE, "Stacker 5-fold OOF RMSLE"),
        ("+5.4%", "vs Ridge baseline"),
        ("80 &rarr; 251", "raw &rarr; engineered features"),
    ]
    cells = []
    for value, label in cards:
        inner = Table([[p(value, "KPI")], [p(label, "KPILabel")]],
                      colWidths=[5.2 * cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        cells.append(inner)
    outer = Table([cells], colWidths=[5.6 * cm] * 3)
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return outer


def leaderboard_table():
    header = ["Model", "OOF RMSLE", "vs Ridge baseline"]
    data = [header] + [list(r) for r in LEADERBOARD]
    t = Table(data, colWidths=[7.5 * cm, 3.5 * cm, 4 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        # Highlight the winning (stacker) row.
        ("BACKGROUND", (0, 1), (-1, 1), WIN),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ]
    t.setStyle(TableStyle(style))
    return t


def info_table(rows, col0=NAVY, widths=(4 * cm, 11 * cm)):
    t = Table(rows, colWidths=list(widths))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), col0),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build():
    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="House Prices: Advanced Regression Report",
        author="Data Science Portfolio",
    )
    S = []

    # ---------------- Cover ------------------------------------------------
    S.append(Spacer(1, 3.5 * cm))
    S.append(p("House Prices", "CoverTitle"))
    S.append(p("Advanced Regression on the Ames, Iowa Housing Dataset",
              "CoverSub"))
    S.append(Spacer(1, 0.5 * cm))
    S.append(hr())
    S.append(Spacer(1, 0.3 * cm))
    S.append(p("Stacked Ridge Meta-Learner over 7 Base Models &mdash; "
               "EDA, Cleaning, Feature Engineering, Tuning &amp; Explainability",
               "CoverSub"))
    S.append(Spacer(1, 1.4 * cm))
    S.append(kpi_row())
    S.append(PageBreak())

    # ---------------- 1. Business Problem ---------------------------------
    S.append(p("1. Business Problem", "H1"))
    S.append(hr())
    S.append(p("A real-estate company wants to estimate the selling price of a "
               "house <b>before</b> it is listed. A reliable pre-listing "
               "valuation model lets the business set competitive prices, sell "
               "faster, and avoid the cost of over- or under-pricing inventory."))
    S.append(p("Business benefits", "H2"))
    S.extend(bullets([
        "<b>Better pricing strategy</b> &mdash; data-driven list prices instead of guesswork.",
        "<b>Faster sales</b> &mdash; correctly priced homes spend less time on market.",
        "<b>Reduced over/under-pricing risk</b> &mdash; fewer mark-downs and missed margin.",
        "<b>Acquisition screening</b> &mdash; flag homes priced below model value.",
        "<b>Renovation ROI</b> &mdash; quantify which upgrades move price most.",
    ]))
    S.append(p("<b>Target variable:</b> SalePrice, modeled on the "
               "<b>log1p</b> scale. The competition is scored on RMSLE (Root "
               "Mean Squared Log Error), which is mathematically equivalent to "
               "RMSE on log(SalePrice) &mdash; so predicting in log space "
               "aligns the loss with the metric.", "Body"))

    # ---------------- 2. Dataset ------------------------------------------
    S.append(p("2. Dataset and Target", "H1"))
    S.append(hr())
    S.append(p("The Kaggle House Prices dataset describes residential homes in "
               "<b>Ames, Iowa</b> using <b>80 explanatory features</b> "
               "(1,460 train / 1,459 test rows). After removing two documented "
               "outliers, <b>1,458 rows</b> are used for training. Features "
               "span every dimension a buyer cares about:"))
    S.append(info_table([
        ["Category", "Example features"],
        ["Area", "LotArea, GrLivArea, TotalBsmtSF, 1stFlrSF"],
        ["Quality", "OverallQual, OverallCond, KitchenQual"],
        ["Location", "Neighborhood, Condition1"],
        ["Age", "YearBuilt, YearRemodAdd, YrSold"],
        ["Garage", "GarageArea, GarageCars, GarageType"],
        ["Basement", "TotalBsmtSF, BsmtFinSF1, BsmtQual"],
    ]))
    S.append(p("2.1 Why log-transform the target?", "H2"))
    S.append(p("Raw SalePrice has a strong right skew (<b>1.88</b>) &mdash; a "
               "long tail of luxury homes pulls the mean above the median. "
               "After <b>log1p</b>, skewness drops to <b>0.12</b> and the QQ "
               "plot is nearly linear, satisfying linear-model assumptions and "
               "weighting proportional errors equally across price ranges."))
    S.extend(fig("01_target_distribution.png", caption=
                 "Fig 1. Raw SalePrice (left), log1p-transformed (middle), and "
                 "QQ plot vs Normal (right)."))
    S.append(PageBreak())

    # ---------------- 3. EDA ----------------------------------------------
    S.append(p("3. Exploratory Data Analysis", "H1"))
    S.append(hr())
    S.append(p("3.1 Missing values: most are not actually missing", "H2"))
    S.append(p("19 columns have missing values, but the data dictionary reveals "
               "that most NaN entries encode the <b>absence</b> of a feature, "
               "not unknown data. PoolQC is 99.5% NaN simply because almost no "
               "home has a pool. Filling these with a median would invent pools "
               "where there are none; the pipeline fills them with the literal "
               "string &quot;None&quot; so encoding treats them as a category."))
    S.extend(fig("02_missing.png", width=14 * cm, caption=
                 "Fig 2. Top missing features in train (left) and test (right). "
                 "Leading entries are all absence-encodings."))
    S.append(p("3.2 Correlations with target", "H2"))
    S.append(p("<b>OverallQual</b> correlates <b>0.79</b> with SalePrice "
               "&mdash; by far the strongest single predictor. Five features "
               "clear the 0.6 mark, covering the obvious drivers: overall "
               "quality, living area, garage capacity, basement size and first "
               "floor size. GarageCars and GarageArea are nearly redundant."))
    S.extend(fig("03_correlation.png", width=14 * cm, caption=
                 "Fig 3. Top numeric features by absolute correlation (left); "
                 "pairwise correlation heatmap (right)."))
    S.append(PageBreak())
    S.append(p("3.3 Key price drivers", "H2"))
    S.append(p("Plotting the six strongest predictors against SalePrice "
               "confirms quality and size dominate. Two GrLivArea points beyond "
               "4,000 sqft sit far below trend &mdash; documented partial sales "
               "removed before training."))
    S.extend(fig("04_key_features.png", caption=
                 "Fig 4. Six strongest predictors plotted against SalePrice."))
    S.append(p("3.4 Neighborhood premium", "H2"))
    S.append(p("Median price varies <b>3.6&times;</b> between the cheapest "
               "(MeadowV) and most expensive (NridgHt) neighborhoods. This "
               "single categorical encodes a large fraction of total variance "
               "&mdash; exploited twice: as a target-encoded numeric feature "
               "and via KMeans clustering."))
    S.extend(fig("05_neighborhood.png", caption=
                 "Fig 5. Neighborhood-level median sale prices, sorted "
                 "descending (sample sizes shown per bar)."))
    S.append(PageBreak())

    # ---------------- 4. Cleaning & Missing Values ------------------------
    S.append(p("4. Data Cleaning &amp; Missing Values", "H1"))
    S.append(hr())
    S.append(p("Missing values were imputed with <b>semantic, domain-aware</b> "
               "rules rather than blind defaults, and two documented outliers "
               "(homes &gt; 4,000 sqft that sold for &lt; $300K) were removed."))
    S.append(info_table([
        ["Strategy", "Rationale"],
        ['Fill with "None"', "NaN encodes feature absence (PoolQC, Alley, Fence, "
         "FireplaceQu, Garage*, Bsmt*, MasVnrType)"],
        ["Fill with 0", "No feature -> zero size/count (GarageArea/Cars, "
         "TotalBsmtSF, BsmtFinSF1/2, MasVnrArea)"],
        ["Neighborhood median", "LotFrontage &mdash; houses on the same street "
         "share a frontage profile"],
        ["Mode", "Single-NaN columns (MSZoning, Electrical, KitchenQual, "
         "Functional, Utilities)"],
    ], col0=BLUE, widths=(4.2 * cm, 10.8 * cm)))
    S.append(p("Of the missing columns, only <b>LotFrontage</b> (17.7% NaN) is "
               "truly missing data; the rest are absence-encodings.", "Body"))

    # ---------------- 5. Feature Engineering ------------------------------
    S.append(p("5. Feature Engineering", "H1"))
    S.append(hr())
    S.append(p("Buyers think holistically &mdash; total space, total bathrooms, "
               "recency. The pipeline adds <b>17 engineered features</b>, then "
               "applies log1p to 42 skewed numerics and one-hot encoding, "
               "expanding the matrix to <b>251 model-ready features</b>."))
    S.append(info_table([
        ["Feature", "Definition"],
        ["TotalSF", "TotalBsmtSF + 1stFlrSF + 2ndFlrSF"],
        ["TotalBathrooms", "FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath"],
        ["HouseAge / YearsSinceRemod", "Temporal recency features"],
        ["Has* flags", "HasPool, HasGarage, Has2ndFloor, HasBsmt, HasFireplace"],
        ["QualArea / QualTotalSF", "OverallQual x area &mdash; quality-weighted size"],
        ["NeighborhoodPrice", "Target encoding (median price per nbhd, train-only)"],
        ["NeighborhoodCluster", "KMeans (k=5) on aggregated neighborhood stats"],
    ], col0=BLUE, widths=(5 * cm, 10 * cm)))
    S.append(p("<b>Leakage guard:</b> target encoding and KMeans clustering are "
               "fit on <b>train data only</b>, then mapped onto test. Fitting on "
               "the train+test concatenation would leak the test distribution "
               "and give optimistic CV scores.", "Body"))
    S.append(PageBreak())

    # ---------------- 6. Modeling & Tuning --------------------------------
    S.append(p("6. Modeling &amp; Hyperparameter Tuning", "H1"))
    S.append(hr())
    S.append(p("Seven base models were validated with <b>5-fold KFold</b> "
               "(shuffle, seed 42) on the log target:"))
    S.append(info_table([
        ["Model", "Role"],
        ["Ridge / Lasso / ElasticNet", "Linear baselines (L2 / L1 / hybrid)"],
        ["GBM (sklearn)", "Tree boosting with Huber loss"],
        ["XGBoost", "Gradient boosting, strong on interactions"],
        ["LightGBM", "Fast histogram-based boosting"],
        ["CatBoost", "Native categorical handling"],
    ], col0=BLUE, widths=(5.5 * cm, 9.5 * cm)))
    S.append(p("Tuning &amp; stacking", "H2"))
    S.extend(bullets([
        "<b>XGBoost, LightGBM, CatBoost</b> tuned with <b>Optuna</b> "
        "(TPE sampler, 30&ndash;100 trials), optimizing 5-fold CV RMSLE.",
        "<b>Ridge meta-learner</b> (RidgeCV) fit on the 5-fold OOF predictions "
        "of all 7 base models &mdash; true stacking, not fixed-weight blending.",
        "The meta-learner can assign negative coefficients and selects "
        "regularization via cross-validated alpha search.",
    ]))

    # ---------------- 7. Results ------------------------------------------
    S.append(p("7. Results &amp; Model Comparison", "H1"))
    S.append(hr())
    S.append(p("Cross-validation scorecard (30 Optuna trials per booster). "
               "Lower OOF RMSLE is better."))
    S.append(leaderboard_table())
    S.append(Spacer(1, 8))
    S.append(p(f"<b>Winner: the Ridge stacker</b> at OOF RMSLE "
               f"<b>{BEST_RMSLE}</b> &mdash; a <b>5.4% improvement</b> over the "
               f"Ridge baseline. Notably, the best <i>single</i> model was "
               f"<b>{BEST_SINGLE[0]} ({BEST_SINGLE[1]})</b>; even it lost to "
               f"the stacker. Lasso and ElasticNet outscored all three tuned "
               f"tree boosters &mdash; on a small, clean dataset, simple often "
               f"wins."))
    S.extend(fig("06_model_comparison.png", width=15 * cm, caption=
                 "Fig 6. 5-fold CV RMSLE for all 7 base models with 1-sigma "
                 "error bars (left) and percent improvement vs Ridge (right)."))
    S.append(PageBreak())

    # ---------------- 8. Diagnostics --------------------------------------
    S.append(p("8. Stacker Diagnostics", "H1"))
    S.append(hr())
    S.append(p("Residuals are centered on zero with no strong "
               "heteroscedasticity. The predicted-vs-actual scatter hugs the "
               "y=x line tightly, with slight under-prediction at the upper "
               "extreme &mdash; expected for a model trained on a long-tailed "
               "target. Mean OOF RMSLE: <b>0.10740</b>."))
    S.extend(fig("08_diagnostics.png", width=14 * cm, caption=
                 "Fig 7. Stacker diagnostics &mdash; residual plot (left) and "
                 "OOF predicted vs actual (right)."))
    S.append(p("Feature importance", "H2"))
    S.append(p("Engineered features dominate every tree model's rankings. "
               "<b>QualTotalSF</b>, <b>QualArea</b> and <b>NeighborhoodPrice</b> "
               "consistently appear in the top 5 &mdash; validation that the "
               "engineering work paid off."))
    S.extend(fig("07_feature_importance.png", width=14 * cm, caption=
                 "Fig 8. Top 20 features by importance for the tuned tree "
                 "models (engineered features highlighted)."))
    S.append(PageBreak())

    # ---------------- 9. Explainability -----------------------------------
    S.append(p("9. Explainability (SHAP)", "H1"))
    S.append(hr())
    S.append(p("SHAP values decompose each prediction into per-feature "
               "contributions, accounting for interactions &mdash; a more "
               "faithful picture than split-count importances. "
               "<b>QualTotalSF</b> (quality &times; total area) dominates as the "
               "single most informative composite feature."))
    S.extend(fig("10_shap_summary.png", width=14 * cm, caption=
                 "Fig 9. SHAP summary for the tuned XGBoost model. Color encodes "
                 "feature value (red=high, blue=low); x-axis shows impact on "
                 "predicted log price."))
    S.extend(fig("11_shap_top20.png", width=14 * cm, caption=
                 "Fig 10. Top 20 features by mean |SHAP value|."))
    S.append(p("Prediction sanity check", "H2"))
    S.append(p("Predicted test prices closely match the actual training-price "
               "distribution &mdash; no obvious extrapolation pathology."))
    S.extend(fig("09_submission_check.png", width=13 * cm, caption=
                 "Fig 11. Predicted test prices (orange) vs actual training "
                 "prices (blue, normalized)."))
    S.append(PageBreak())

    # ---------------- 10. Business Value & Future Work --------------------
    S.append(p("10. Business Value &amp; Future Work", "H1"))
    S.append(hr())
    S.append(p("The stacked model converts raw property attributes into a "
               "defensible, auditable price estimate the business can act on:"))
    S.extend(bullets([
        "<b>Pricing engine:</b> an instant pre-listing valuation, deployed via "
        "the Streamlit estimator over the serialized artifacts/model.pkl.",
        "<b>Faster turnover:</b> accurate prices reduce time-on-market and "
        "costly mark-downs.",
        "<b>Acquisition screening:</b> flag homes priced below model value as "
        "investment candidates.",
        "<b>Renovation ROI:</b> SHAP shows quality-weighted area moves price "
        "most &mdash; guiding where upgrade spend pays back.",
        "<b>Transparency:</b> SHAP explanations make every valuation auditable.",
    ]))
    S.append(p("Key takeaways", "H2"))
    S.extend(bullets([
        "Log-transforming the target aligned the loss with RMSLE and tamed luxury outliers.",
        "Most NaN values encoded absence, not unknown data &mdash; semantic imputation mattered.",
        "Engineered features (QualTotalSF, QualArea) outranked every raw column.",
        "Stacking beat the best single model: CatBoost 0.113 lost to the Ridge stacker 0.107.",
    ]))
    S.append(p("Future improvements", "H2"))
    S.extend(bullets([
        "<b>More Optuna trials</b> (--trials 100) typically pushes the stacker below 0.107.",
        "<b>External data:</b> school ratings, crime rate, distance to city center.",
        "<b>Deployment hardening:</b> wrap the saved model behind a monitored API.",
        "<b>BI dashboard</b> for neighborhood ranking and predicted-vs-actual tracking.",
    ]))
    S.append(Spacer(1, 0.5 * cm))
    S.append(hr())
    S.append(p("Generated with reportlab from project figures. Metrics recovered "
               "from House_Prices_Report.docx and verified against "
               "house_prices.py. Reproduce the pipeline with "
               "<font name='Courier'>python house_prices.py</font>.", "Caption"))

    doc.build(S)
    n_figs = sum(1 for f in [
        "01_target_distribution", "02_missing", "03_correlation",
        "04_key_features", "05_neighborhood", "06_model_comparison",
        "07_feature_importance", "08_diagnostics", "09_submission_check",
        "10_shap_summary", "11_shap_top20",
    ] if (FIGURE_DIR / f"{f}.png").exists())
    print(f"Saved report -> {REPORT_PATH}")
    print(f"Embedded {n_figs}/11 figures.")


if __name__ == "__main__":
    build()
