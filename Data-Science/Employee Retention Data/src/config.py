"""
Central configuration: paths, business assumptions, and column definitions.

Keeping every "magic number" and path in one place makes the pipeline
reproducible and easy to audit -- an interviewer can read this file and
understand every assumption behind the ROI calculation.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
MODEL_DIR = OUTPUTS / "models"
METRIC_DIR = OUTPUTS / "metrics"
REPORT_DIR = ROOT / "reports"

for _d in (FIG_DIR, MODEL_DIR, METRIC_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Business assumptions (drive the ROI / business-value section of the report)
# --------------------------------------------------------------------------- #
COMPANY_HEADCOUNT = 10_000
BASELINE_ATTRITION_RATE = 0.15           # 15% leave per year

COST_RECRUITMENT = 5_000
COST_TRAINING = 3_000
COST_PRODUCTIVITY = 7_000
COST_PER_LEAVER = COST_RECRUITMENT + COST_TRAINING + COST_PRODUCTIVITY  # $15,000

# Retention-programme assumptions (how effective the HR intervention is)
SHARE_LEAVERS_DETECTED = 0.80            # recall we expect to act on in production
SHARE_DETECTED_RETAINED = 0.25           # of those flagged, share we successfully keep

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
TARGET = "Attrition"

# Constant / identifier columns carry no signal -> dropped during prep.
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

# Categorical columns that need one-hot encoding (after feature engineering).
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "SalaryBand",
    "ExperienceGroup",
    "CommuteCategory",
]

# Decision threshold used when converting probabilities to a "will leave" label.
# Lowered below 0.5 because catching leavers (recall) is worth more to the
# business than avoiding the occasional false alarm.
DECISION_THRESHOLD = 0.35
