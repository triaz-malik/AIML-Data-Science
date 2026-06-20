"""
Data preparation & category consolidation for the CFPB Consumer Complaint project.

- Loads the raw CFPB CSV.
- Keeps only complaints that contain a consumer narrative (the text we classify on).
- Consolidates the 18 raw `Product` labels into 8 canonical business categories
  (CFPB renamed/merged product taxonomies over the years; we undo that drift).
- Saves a cleaned parquet file + a JSON summary of the label space.
"""
import json
import os
import pandas as pd

BASE = r"C:\Working\AI ML Projetcs\Consumer Complaints"
RAW = os.path.join(BASE, "Consumer_Complaints.csv")
DATA_DIR = os.path.join(BASE, "data")
METRICS_DIR = os.path.join(BASE, "outputs", "metrics")

# Map the 18 raw products -> 8 canonical categories. None => drop (tiny/ambiguous).
CATEGORY_MAP = {
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Credit reporting": "Credit reporting & repair",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting & repair",
    "Credit card": "Credit card or prepaid card",
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Bank account or service": "Bank account or service",
    "Checking or savings account": "Bank account or service",
    "Student loan": "Student loan",
    "Consumer Loan": "Consumer & personal loan",
    "Vehicle loan or lease": "Consumer & personal loan",
    "Payday loan": "Consumer & personal loan",
    "Payday loan, title loan, or personal loan": "Consumer & personal loan",
    "Money transfers": "Money transfer & virtual currency",
    "Money transfer, virtual currency, or money service": "Money transfer & virtual currency",
    "Virtual currency": "Money transfer & virtual currency",
    "Other financial service": None,  # 292 rows, ambiguous -> drop
}

USE_COLS = [
    "Date received", "Product", "Sub-product", "Issue",
    "Consumer complaint narrative", "Company", "State",
    "Submitted via", "Company response to consumer",
    "Timely response?", "Consumer disputed?",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading raw CSV...")
    df = pd.read_csv(RAW, usecols=USE_COLS, dtype=str)
    n_total = len(df)

    # Keep rows with a real narrative.
    df = df[df["Consumer complaint narrative"].notna()].copy()
    df["Consumer complaint narrative"] = df["Consumer complaint narrative"].str.strip()
    df = df[df["Consumer complaint narrative"].str.len() > 0]
    n_with_text = len(df)

    # Consolidate categories.
    df["category"] = df["Product"].map(CATEGORY_MAP)
    df = df[df["category"].notna()].copy()
    n_final = len(df)

    # Parse dates.
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df = df[df["Date received"].notna()].copy()

    df = df.rename(columns={
        "Consumer complaint narrative": "narrative",
        "Date received": "date",
        "State": "state",
        "Company": "company",
        "Submitted via": "submitted_via",
        "Timely response?": "timely_response",
        "Consumer disputed?": "consumer_disputed",
        "Company response to consumer": "company_response",
    })

    out = os.path.join(DATA_DIR, "clean.parquet")
    df.to_parquet(out, index=False)

    summary = {
        "rows_total_raw": int(n_total),
        "rows_with_narrative": int(n_with_text),
        "pct_with_narrative": round(n_with_text / n_total * 100, 2),
        "rows_after_consolidation": int(n_final),
        "n_categories": int(df["category"].nunique()),
        "category_counts": df["category"].value_counts().to_dict(),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_companies": int(df["company"].nunique()),
        "n_states": int(df["state"].nunique()),
    }
    with open(os.path.join(METRICS_DIR, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved cleaned dataset -> {out}")


if __name__ == "__main__":
    main()
