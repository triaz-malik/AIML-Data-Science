"""
Central configuration for the Customer Recommendation & Segmentation Engine.

All paths are resolved relative to this file so the pipeline runs the same way
regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

RAW_CSV = ROOT / "online_retail_II.csv"

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
POWERBI_DIR = DATA_DIR / "powerbi"

OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

# Cleaned, analysis-ready transaction table (one row per line item)
CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
# Customer-level feature table (one row per customer, incl. RFM + segments)
CUSTOMER_PARQUET = PROCESSED_DIR / "customer_features.parquet"

for _d in (PROCESSED_DIR, POWERBI_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Analysis parameters
# --------------------------------------------------------------------------- #
# The dataset ends in Dec 2011. We snapshot "today" as one day after the last
# invoice so Recency is well-defined for every customer.
SNAPSHOT_OFFSET_DAYS = 1

RANDOM_STATE = 42

# Segmentation
KMEANS_K_RANGE = range(2, 11)        # search range for the elbow / silhouette
KMEANS_N_SEGMENTS = 4                # final K used for business segments
DBSCAN_EPS = 1.2                     # in standardized RFM space
DBSCAN_MIN_SAMPLES = 30

# KNN recommendation engine
KNN_NEIGHBORS = 11                   # default K; tuned in recommendation.py
KNN_METRIC = "cosine"
TOP_N_RECOMMENDATIONS = 10

# Market basket (Apriori)
BASKET_MIN_SUPPORT = 0.02
BASKET_MIN_LIFT = 1.0
BASKET_COUNTRY = "United Kingdom"    # focus market to keep the basket dense

# Customer Lifetime Value
CLV_CALIBRATION_MONTHS = 12          # features built on first N months ...
CLV_HOLDOUT_MONTHS = 6               # ... target = spend in following M months

# Plot style
FIG_DPI = 120
PALETTE = "viridis"
