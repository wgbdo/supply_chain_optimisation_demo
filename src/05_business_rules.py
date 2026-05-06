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


def load_rules() -> list[dict]:
    """Load Dave's rules from the JSON file."""
    with open(RULES_PATH) as f:
        rules = json.load(f)
    print(f"  Loaded {len(rules)} business rules from {RULES_PATH}")
    return rules


def check_condition(rule: dict, row: pd.Series) -> bool:
    """
    Evaluate whether a rule's condition is met for a given data row.

    In a production system, you'd use a proper rule engine (e.g. business-rules,
    Drools, or a simple DSL). For this PoC, we interpret the JSON conditions
    directly.
    """
    condition = rule["condition"]
    ctype = condition["type"]

    if ctype == "always":
        return True

    elif ctype == "holiday_upcoming":
        # Check if the week's holiday_name contains the target holiday
        target = condition["holiday_name"].lower()
        return target in row.get("holiday_name", "").lower()

    elif ctype == "temperature_above":
        # For this demo, we don't have temperature data, so we skip this rule.
        # In production, you'd join weather forecast data.
        return False

    elif ctype == "item_age_weeks_below":
        # We don't track item introduction dates in this demo.
        # In production, you'd compute weeks since first sale.
        return False

    elif ctype == "week_contains_date":
        # Check if the week start's day-of-month falls in the target list
        dom = row["week_start"].day
        return dom in condition.get("day_of_month", [])

    return False


def check_sku_pattern(rule: dict, row: pd.Series) -> bool:
    """
    Check if a rule applies to this item based on sku_pattern.

    Patterns:
      "*"           → applies to all items
      "perishable"  → only perishable items (perishable == 1)
    """
    pattern = rule["sku_pattern"]

    if pattern == "*":
        return True
    elif pattern == "perishable":
        return row.get("perishable", 0) == 1

    return False


def apply_rules(row: pd.Series, rules: list[dict]) -> dict:
    """
    Apply all matching rules to a single forecast row.

    Returns adjusted forecasts and metadata about which rules fired.

    CONCEPT — Rule Stacking:
      Multiple rules can fire for the same row. Their effects stack
      multiplicatively. For example:
        - Easter uplift: ×1.35
        - Perishable bruising buffer: ×1.15
        → Combined effect: ×1.35 × 1.15 = ×1.55

      This matches how Dave would think: "It's Easter AND it's perishable,
      so I need even more buffer."
    """
    forecast_multiplier = 1.0
    order_qty_multiplier = 1.0
    safety_stock_multiplier = 1.0
    rules_applied = []
    explanations = []

    for rule in rules:
        # Check if this rule matches the SKU and condition
        if not check_sku_pattern(rule, row):
            continue
        if not check_condition(rule, row):
            continue

        # Apply the action
        action = rule["action"]
        atype = action["type"]

        if atype == "multiply_forecast":
            forecast_multiplier *= action["factor"]
        elif atype == "multiply_order_qty":
            order_qty_multiplier *= action["factor"]
        elif atype == "multiply_safety_stock":
            safety_stock_multiplier *= action["factor"]

        rules_applied.append(rule["rule_id"])
        explanations.append(f"[{rule['rule_id']}] {rule['name']}: {rule['rationale']}")

    # Apply adjustments to the forecast values
    adjusted_q50 = row["forecast_q50"] * forecast_multiplier
    adjusted_q90 = row["forecast_q90"] * forecast_multiplier

    return {
        "adjusted_q50": adjusted_q50,
        "adjusted_q90": adjusted_q90,
        "order_qty_multiplier": order_qty_multiplier,
        "safety_stock_multiplier": safety_stock_multiplier,
        "forecast_multiplier": forecast_multiplier,
        "rules_applied": "|".join(rules_applied) if rules_applied else "",
        "explanations": " // ".join(explanations) if explanations else "No rules applied — using base ML forecast.",
    }


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
