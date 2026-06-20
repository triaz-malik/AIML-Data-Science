"""Central configuration: paths, constants, column groups, thresholds."""

from pathlib import Path

# ------------------------------------------------------------------ paths --- #
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "healthcare-dataset-stroke-data.csv"

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
MODEL_DIR = OUTPUTS / "models"
REPORT_DIR = OUTPUTS / "reports"
PRED_DIR = OUTPUTS / "predictions"
PBI_DIR = OUTPUTS / "powerbi"

for _d in (OUTPUTS, FIG_DIR, MODEL_DIR, REPORT_DIR, PRED_DIR, PBI_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "stroke"

# Model deployed/served downstream (SHAP, risk scoring, Power BI). KNN is the
# featured model for portfolio consistency; RF/XGBoost remain as benchmarks in
# the comparison table. See model_comparison_report.md for the AUC tradeoff.
SERVED_MODEL = "knn"

# ----------------------------------------------------- feature groupings --- #
# Raw numeric / categorical features present in the source data.
RAW_NUMERIC = ["age", "avg_glucose_level", "bmi"]
RAW_CATEGORICAL = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# Engineered features added in data_prep.add_features().
ENGINEERED_NUMERIC = ["health_risk_score"]
ENGINEERED_CATEGORICAL = ["age_group", "bmi_category", "glucose_category"]

# --------------------------------------------- risk-stratification bands --- #
# Predicted stroke probability -> clinical action. Tunable.
RISK_BANDS = [
    (0.00, 0.10, "Low", "Routine Monitoring"),
    (0.10, 0.30, "Medium", "Follow-up"),
    (0.30, 0.60, "High", "Specialist Review"),
    (0.60, 1.01, "Critical", "Immediate Attention"),
]
