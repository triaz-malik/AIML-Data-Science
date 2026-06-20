"""
Phase 7 - Market Basket Analysis (Apriori association rules).

Where the KNN engine answers "what should THIS customer buy next?", market
basket analysis answers "which products sell TOGETHER?" — the basis for product
bundling, store layout and cross-sell promotions.

We build one basket per invoice (focused on the dense UK market and the most
popular products so Apriori stays tractable), mine frequent itemsets, and derive
association rules ranked by lift:

  support    - how often the itemset appears
  confidence - P(consequent | antecedent)
  lift       - how much more likely the consequent is, given the antecedent,
               versus its baseline rate (lift > 1 => positive association)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TOP_N_PRODUCTS = 300   # restrict to popular products to keep Apriori tractable


def build_baskets(df: pd.DataFrame) -> pd.DataFrame:
    market = df[df["Country"] == config.BASKET_COUNTRY]
    # Keep only the most popular products by number of invoices they appear in.
    top = (market.groupby("Description")["Invoice"].nunique()
                 .sort_values(ascending=False).head(TOP_N_PRODUCTS).index)
    market = market[market["Description"].isin(top)]

    basket = (market.groupby(["Invoice", "Description"])["Quantity"].sum()
                    .unstack(fill_value=0))
    basket = (basket > 0)                 # one-hot: product present in invoice
    # Drop single-item invoices — they carry no co-occurrence signal.
    basket = basket[basket.sum(axis=1) >= 2]
    return basket


def mine_rules(basket: pd.DataFrame) -> pd.DataFrame:
    itemsets = apriori(basket, min_support=config.BASKET_MIN_SUPPORT,
                       use_colnames=True, low_memory=True)
    if itemsets.empty:
        raise RuntimeError("No frequent itemsets — lower BASKET_MIN_SUPPORT.")
    rules = association_rules(itemsets, metric="lift",
                              min_threshold=config.BASKET_MIN_LIFT)
    # Readable string columns for export / dashboard.
    rules["antecedents_s"] = rules["antecedents"].apply(
        lambda s: ", ".join(sorted(s)))
    rules["consequents_s"] = rules["consequents"].apply(
        lambda s: ", ".join(sorted(s)))
    rules = rules.sort_values("lift", ascending=False)
    return rules


def run() -> None:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    print(f"Building baskets for {config.BASKET_COUNTRY} "
          f"(top {TOP_N_PRODUCTS} products) ...")
    basket = build_baskets(df)
    print(f"  {basket.shape[0]:,} multi-item invoices x {basket.shape[1]} products")

    print("Mining association rules (Apriori) ...")
    rules = mine_rules(basket)

    cols = ["antecedents_s", "consequents_s", "support", "confidence", "lift"]
    export = rules[cols].rename(columns={"antecedents_s": "antecedents",
                                         "consequents_s": "consequents"})
    export.to_csv(config.REPORT_DIR / "association_rules.csv", index=False)

    print(f"\nFound {len(rules):,} rules (lift >= {config.BASKET_MIN_LIFT}). "
          f"Top 12 by lift:\n")
    show = export.head(12).copy()
    show["support"] = show["support"].round(3)
    show["confidence"] = show["confidence"].round(3)
    show["lift"] = show["lift"].round(2)
    with pd.option_context("display.max_colwidth", 38, "display.width", 160):
        print(show.to_string(index=False))
    print(f"\nSaved -> {config.REPORT_DIR / 'association_rules.csv'}")


if __name__ == "__main__":
    run()
