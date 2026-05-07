"""
Business rules utility functions — pure logic with no I/O.

Extracted from 05_business_rules.py so they can be unit-tested without
importing a module whose name starts with a digit (which Python's normal
import machinery cannot handle).
"""

import pandas as pd


def check_condition(rule: dict, row: pd.Series) -> bool:
    """Evaluate whether a rule's condition is met for a given data row."""
    condition = rule["condition"]
    ctype = condition["type"]

    if ctype == "always":
        return True

    elif ctype == "holiday_upcoming":
        target = condition["holiday_name"].lower()
        return target in row.get("holiday_name", "").lower()

    elif ctype == "temperature_above":
        # No temperature data in this demo — skip.
        return False

    elif ctype == "item_age_weeks_below":
        # Item introduction dates not tracked in this demo — skip.
        return False

    elif ctype == "week_contains_date":
        dom = row["week_start"].day
        return dom in condition.get("day_of_month", [])

    elif ctype == "month_in":
        # e.g. Australian summer: months [10, 11, 12, 1, 2, 3, 4]
        month = row["week_start"].month
        return month in condition.get("months", [])

    elif ctype == "low_rolling_demand":
        # Fire when the item's rolling average demand is below the threshold.
        # Uses a pre-computed rolling column already present in forecasts.parquet
        # (e.g. rolling_4w_mean, rolling_12w_mean).
        col = condition.get("rolling_column", "rolling_4w_mean")
        threshold = condition.get("threshold", 5.0)
        val = row.get(col, None)
        if val is None:
            return False
        return float(val) < threshold

    return False


def check_sku_pattern(rule: dict, row: pd.Series) -> bool:
    """Check if a rule applies to this item based on sku_pattern."""
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
    """
    forecast_multiplier = 1.0
    order_qty_multiplier = 1.0
    safety_stock_multiplier = 1.0
    rules_applied = []
    explanations = []

    for rule in rules:
        if not check_sku_pattern(rule, row):
            continue
        if not check_condition(rule, row):
            continue

        action = rule["action"]
        atype = action["type"]

        if atype == "multiply_forecast":
            forecast_multiplier *= action["factor"]
        elif atype == "multiply_order_qty":
            order_qty_multiplier *= action["factor"]
        elif atype == "multiply_safety_stock":
            safety_stock_multiplier *= action["factor"]
        elif atype == "cap_forecast":
            # Cap the forecast at rolling_mean * cap_multiplier.
            # Stored as a special marker — applied after the multiplier loop.
            cap_col = rule["condition"].get("rolling_column", "rolling_4w_mean")
            cap_val = float(row.get(cap_col, row["forecast_q50"])) * action.get("cap_multiplier", 1.0)
            forecast_multiplier = min(forecast_multiplier, cap_val / max(row["forecast_q50"], 1e-6))

        rules_applied.append(rule["rule_id"])
        explanations.append(f"[{rule['rule_id']}] {rule['name']}: {rule['rationale']}")

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
