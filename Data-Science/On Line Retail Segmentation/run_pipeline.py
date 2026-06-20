"""
End-to-end pipeline runner for the Customer Recommendation & Segmentation Engine.

Runs every phase in dependency order. Use --from / --to to run a slice, e.g.:

    python run_pipeline.py                 # full pipeline
    python run_pipeline.py --from features # resume from RFM onward
    python run_pipeline.py --only eda      # single phase
"""
from __future__ import annotations

import argparse
import time

from src import (clv, data_cleaning, eda, explainability, export_powerbi,
                 features, market_basket, recommendation, segmentation)

PHASES = [
    ("clean", "Phase 2  Data Cleaning", data_cleaning.run),
    ("eda", "Phase 3  Exploratory Data Analysis", eda.run),
    ("features", "Phase 4  RFM Feature Engineering", features.run),
    ("segmentation", "Phase 5  Customer Segmentation", segmentation.run),
    ("recommendation", "Phase 6  KNN Recommendation Engine", recommendation.run),
    ("market_basket", "Phase 7  Market Basket Analysis", market_basket.run),
    ("clv", "Phase 8  Customer Lifetime Value", clv.run),
    ("explainability", "Phase 9  SHAP Explainability", explainability.run),
    ("powerbi", "Phase 10 Power BI Export", export_powerbi.export),
    ("report", "Phase 10 Business Report", None),  # filled below
]


def _run_report():
    from src import report
    report.run()


PHASES[-1] = ("report", "Phase 10 Business Report", _run_report)


def main():
    keys = [k for k, _, _ in PHASES]
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=keys)
    ap.add_argument("--to", dest="end", choices=keys)
    ap.add_argument("--only", dest="only", choices=keys)
    args = ap.parse_args()

    if args.only:
        selected = [p for p in PHASES if p[0] == args.only]
    else:
        s = keys.index(args.start) if args.start else 0
        e = keys.index(args.end) + 1 if args.end else len(PHASES)
        selected = PHASES[s:e]

    t0 = time.time()
    for key, title, fn in selected:
        print("\n" + "#" * 70)
        print(f"# {title}")
        print("#" * 70)
        ts = time.time()
        fn()
        print(f"[{key}] done in {time.time() - ts:.1f}s")
    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
