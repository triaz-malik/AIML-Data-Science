"""Central configuration: paths, constants, and feature groups."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"

for _d in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
TARGET = "SalePrice"
ID_COL = "Id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# n_iter for RandomizedSearchCV. Lower = faster, higher = more thorough.
N_ITER_SEARCH = 25

# ---------------------------------------------------------------------------
# Feature groups (derived from data_description.txt)
# ---------------------------------------------------------------------------
# Categorical columns where a missing value genuinely means "feature absent".
NONE_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Numeric columns where a missing value means zero (no garage/basement/etc.).
ZERO_NUMERIC = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# A handful of numeric-looking columns that are really categorical codes.
NUMERIC_AS_CATEGORICAL = ["MSSubClass"]
