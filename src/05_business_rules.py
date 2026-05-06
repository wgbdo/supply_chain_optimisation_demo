"""
Step 5: Business Rules Engine (Dave's Rules)
=============================================
Applies SME-captured business rules on top of statistical forecasts.

CONCEPT — Why Layer Rules on Top of ML Forecasts?
  ML models are great at learning patterns from historical data, but they can miss:
    1. Rare events the model hasn't seen enough of (e.g. once-a-year Easter spike)
    2. Operational knowledge (e.g. "perishables bruise in transit — always order extra")
    3. External context the features don't capture (e.g. new product launch)

  Rather than trying to encode everything into the ML model, we use a two-layer
  approach:
    Layer 1: ML model produces a base probabilistic forecast
    Layer 2: Business rules adjust the forecast based on SME knowledge

  This is pragmatic and auditable — you can see exactly which rules fired and
  what they did. Over time, as the ML model learns from more data, some of Dave's
  rules may become redundant and can be retired.

Key output: data/processed/adjusted_forecasts.parquet
  Same as forecasts.parquet but with additional columns:
    - adjusted_q50, adjusted_q90 (post-rule forecasts)
    - rules_applied (list of rule IDs that fired)
    - explanations (human-readable text)

Usage:
    python src/05_business_rules.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, RULES_PATH
from src.business_rules_utils import apply_rules, check_condition, check_sku_pattern  # noqa: F401


def load_rules() -> list[dict]:
    """Load Dave's rules from the JSON file."""
    with open(RULES_PATH) as f:
        rules = json.load(f)
    print(f"  Loaded {len(rules)} business rules from {RULES_PATH}")
    return rules



def main():
    print("=" * 60)
    print("Step 5: Business Rules Engine (Dave's Rules)")
    print("=" * 60)

    # Load forecasts from Step 4
    forecasts = pd.read_parquet(DATA_PROCESSED / "forecasts.parquet")
    print(f"\nLoaded {len(forecasts):,} forecast rows")

    # Load rules
    rules = load_rules()

    # Apply rules to each row
    print("\nApplying business rules...")
    adjustments = forecasts.apply(lambda row: apply_rules(row, rules), axis=1)
    adj_df = pd.DataFrame(adjustments.tolist())

    # Merge adjustments back
    result = pd.concat([forecasts.reset_index(drop=True), adj_df], axis=1)

    # Summary: how many rows had rules applied?
    n_with_rules = (result["rules_applied"] != "").sum()
    print(f"\n  Rows with rules applied: {n_with_rules:,} / {len(result):,} ({n_with_rules/len(result)*100:.1f}%)")

    # Show which rules fired and how often
    all_rules = result["rules_applied"].str.split("|").explode()
    all_rules = all_rules[all_rules != ""]
    if len(all_rules) > 0:
        print("\n  Rule frequency:")
        for rule_id, count in all_rules.value_counts().items():
            print(f"    {rule_id}: {count:,} times")

    # Show a few examples of adjusted forecasts
    adjusted_rows = result[result["rules_applied"] != ""].head(5)
    if len(adjusted_rows) > 0:
        print("\n  Example adjusted forecasts:")
        for _, row in adjusted_rows.iterrows():
            print(f"    Store {row['store_nbr']}, Item {row['item_nbr']}, Week {row['week_start'].date()}")
            print(f"      Base q50: {row['forecast_q50']:.0f} → Adjusted: {row['adjusted_q50']:.0f} (×{row['forecast_multiplier']:.2f})")
            print(f"      Rules: {row['rules_applied']}")

    # Save
    output_path = DATA_PROCESSED / "adjusted_forecasts.parquet"
    result.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    print("\nDone! Next step: python src/06_inventory_optimisation.py")


if __name__ == "__main__":
    main()
