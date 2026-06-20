"""
Central configuration for the Telecom Customer Segmentation &
Package Recommendation System.

All paths are resolved relative to the project root so the pipeline
runs the same way regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data (Orange / BigML telecom churn dataset, 80/20 split)
RAW_TRAIN_CSV = PROJECT_ROOT / "churn-bigml-80.csv"
RAW_TEST_CSV = PROJECT_ROOT / "churn-bigml-20.csv"

# Output folders
OUTPUTS = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
DATA_DIR = OUTPUTS / "data"
MODEL_DIR = OUTPUTS / "models"
REPORT_DIR = OUTPUTS / "reports"

for _d in (FIG_DIR, DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Intermediate artifacts shared between phases
FEATURES_CSV = DATA_DIR / "telecom_features.csv"          # Phase 2 output
SEGMENTED_CSV = DATA_DIR / "telecom_segmented.csv"        # Phase 3 output
RECOMMENDATIONS_CSV = DATA_DIR / "recommendations.csv"    # Phase 4 output
SCORED_CSV = DATA_DIR / "telecom_scored.csv"             # Phase 5 output (Power BI ready)

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TARGET = "Churn"

CATEGORICAL_RAW = ["International plan", "Voice mail plan"]
DROP_FOR_MODEL = ["State", "Area code"]  # high-cardinality / non-predictive id-like

# Raw usage columns
DAY_COLS = ["Total day minutes", "Total day calls", "Total day charge"]
EVE_COLS = ["Total eve minutes", "Total eve calls", "Total eve charge"]
NIGHT_COLS = ["Total night minutes", "Total night calls", "Total night charge"]
INTL_COLS = ["Total intl minutes", "Total intl calls", "Total intl charge"]

MINUTE_COLS = [
    "Total day minutes",
    "Total eve minutes",
    "Total night minutes",
    "Total intl minutes",
]
CHARGE_COLS = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]

# Features used for clustering / segmentation (engineered + key raw)
SEGMENTATION_FEATURES = [
    "Total Usage Minutes",
    "Total Charges",
    "International Usage Ratio",
    "Total day minutes",
    "Total eve minutes",
    "Total night minutes",
    "Total intl minutes",
    "Number vmail messages",
    "Customer service calls",
    "Customer Value Score",
]

# Features used for the KNN recommendation engine
RECO_FEATURES = [
    "Total Usage Minutes",
    "Total Charges",
    "Total intl minutes",
    "Total intl calls",
    "Account length",
    "Customer service calls",
]

# Reproducibility
RANDOM_STATE = 42
