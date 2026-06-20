"""
Run the full Telecom Customer Segmentation & Package Recommendation
pipeline, Phase 1 -> Phase 6, then print the Phase 7 business summary.

Usage:
    python src/run_all.py
"""
from __future__ import annotations

import time

import phase1_eda
import phase2_features
import phase3_segmentation
import phase4_recommendation
import phase5_churn
import phase6_explainability
from utils import section


def main() -> None:
    t0 = time.time()
    phase1_eda.run()
    phase2_features.run()
    phase3_segmentation.run()
    phase4_recommendation.run()
    phase5_churn.run()
    phase6_explainability.run()

    section("PHASE 7 - BUSINESS VALUE")
    print(
        "The pipeline lets a telecom operator:\n"
        "  * Identify high-value customers via the Customer Value Score & segments.\n"
        "  * Recommend better packages with the KNN engine (cross-sell / upsell flags).\n"
        "  * Reduce churn by scoring every subscriber's churn probability & risk band.\n"
        "  * Increase ARPU by moving Low-Revenue users toward fitting paid packages.\n"
        "  * Improve retention campaigns by targeting High-Risk + High-Value customers.\n\n"
        "All artifacts are in outputs/ (figures, models, reports, data).\n"
        "outputs/data/telecom_scored.csv is ready to load into Power BI."
    )
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
