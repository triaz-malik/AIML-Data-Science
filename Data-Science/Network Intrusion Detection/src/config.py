"""Central configuration: paths, column groups, and constants."""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "Train_data.csv"   # labelled (column: class)
TEST_CSV = DATA_DIR / "Test_data.csv"     # unlabelled prediction set

# --- Schema ------------------------------------------------------------------
TARGET = "class"                 # values: normal / anomaly
POSITIVE_LABEL = "anomaly"       # encoded as 1 (the "attack" / positive class)

CATEGORICAL = ["protocol_type", "service", "flag"]
# Columns known to be constant in NSL-KDD (no signal); dropped during cleaning.
CONSTANT_CANDIDATES = ["num_outbound_cmds"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Saved artefact names
PREPROCESSOR_PKL = MODELS_DIR / "preprocessor.pkl"
METRICS_JSON = MODELS_DIR / "metrics.json"
MODEL_FILES = {
    "Logistic Regression": MODELS_DIR / "logistic_regression_ids.pkl",
    "Random Forest": MODELS_DIR / "random_forest_ids.pkl",
    "XGBoost": MODELS_DIR / "xgb_ids_model.pkl",
    "Neural Network (MLP)": MODELS_DIR / "mlp_ids.pkl",
}
BEST_MODEL_PKL = MODELS_DIR / "best_model.pkl"
