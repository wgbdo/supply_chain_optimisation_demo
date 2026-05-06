"""
Unit tests for the business rules engine (src/05_business_rules.py).

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.business_rules_utils import check_condition, check_sku_pattern, apply_rules

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_row(**kwargs) -> pd.Series:
    """Build a minimal forecast row for testing."""
    defaults = {
        "store_nbr": 1,
        "item_nbr": 101,
        "week_start": pd.Timestamp("2017-01-02"),  # January = Australian summer
        "perishable": 0,
        "forecast_q50": 300.0,
        "forecast_q90": 420.0,
        "holiday_name": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ── check_sku_pattern ─────────────────────────────────────────────────────────

def test_wildcard_pattern_matches_all():
    rule = {"sku_pattern": "*"}
    assert check_sku_pattern(rule, make_row(perishable=0)) is True
    assert check_sku_pattern(rule, make_row(perishable=1)) is True


def test_perishable_pattern_matches_only_perishable():
    rule = {"sku_pattern": "perishable"}
    assert check_sku_pattern(rule, make_row(perishable=1)) is True
    assert check_sku_pattern(rule, make_row(perishable=0)) is False


# ── check_condition ───────────────────────────────────────────────────────────

def test_always_condition_fires():
    rule = {"condition": {"type": "always"}}
    assert check_condition(rule, make_row()) is True


def test_holiday_condition_fires_on_matching_name():
    rule = {"condition": {"type": "holiday_upcoming", "holiday_name": "Easter"}}
    assert check_condition(rule, make_row(holiday_name="Easter Sunday")) is True
    assert check_condition(rule, make_row(holiday_name="Christmas")) is False
    assert check_condition(rule, make_row(holiday_name="")) is False


def test_temperature_condition_never_fires_without_data():
    rule = {"condition": {"type": "temperature_above", "threshold_celsius": 35}}
    assert check_condition(rule, make_row()) is False


def test_month_in_condition_fires_in_australian_summer():
    rule = {"condition": {"type": "month_in", "months": [10, 11, 12, 1, 2, 3, 4]}}
    # January — in summer
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-01-09"))) is True
    # November — in summer
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-11-06"))) is True
    # July — Australian winter, should NOT fire
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-07-03"))) is False
    # August — Australian winter, should NOT fire
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-08-07"))) is False


def test_week_contains_date_condition():
    rule = {"condition": {"type": "week_contains_date", "day_of_month": [15, 28, 29, 30, 31]}}
    # Week starting on the 15th
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-05-15"))) is True
    # Week starting on the 28th
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-01-28"))) is True
    # Week starting on the 3rd — not in list
    assert check_condition(rule, make_row(week_start=pd.Timestamp("2017-07-03"))) is False


# ── apply_rules ───────────────────────────────────────────────────────────────

def test_no_rules_returns_base_forecast():
    row = make_row(forecast_q50=300.0, forecast_q90=420.0)
    result = apply_rules(row, rules=[])
    assert result["adjusted_q50"] == 300.0
    assert result["adjusted_q90"] == 420.0
    assert result["rules_applied"] == ""


def test_forecast_multiplier_applied_correctly():
    rule = {
        "rule_id": "TEST_001",
        "name": "Test uplift",
        "sku_pattern": "*",
        "condition": {"type": "always"},
        "action": {"type": "multiply_forecast", "factor": 1.35},
        "rationale": "Test",
    }
    row = make_row(forecast_q50=300.0, forecast_q90=400.0)
    result = apply_rules(row, rules=[rule])
    assert abs(result["adjusted_q50"] - 300.0 * 1.35) < 0.01
    assert abs(result["adjusted_q90"] - 400.0 * 1.35) < 0.01
    assert result["rules_applied"] == "TEST_001"


def test_order_qty_multiplier_applied_separately():
    rule = {
        "rule_id": "TEST_002",
        "name": "Bruising buffer",
        "sku_pattern": "perishable",
        "condition": {"type": "always"},
        "action": {"type": "multiply_order_qty", "factor": 1.05},
        "rationale": "Test",
    }
    # Should apply to perishable
    row_perishable = make_row(perishable=1, forecast_q50=300.0, forecast_q90=400.0)
    result = apply_rules(row_perishable, rules=[rule])
    assert abs(result["order_qty_multiplier"] - 1.05) < 0.001
    # forecast itself should be unchanged
    assert result["adjusted_q50"] == 300.0

    # Should NOT apply to non-perishable
    row_non = make_row(perishable=0, forecast_q50=300.0, forecast_q90=400.0)
    result_non = apply_rules(row_non, rules=[rule])
    assert result_non["order_qty_multiplier"] == 1.0


def test_stacking_multipliers():
    rules = [
        {
            "rule_id": "R1",
            "name": "Uplift A",
            "sku_pattern": "*",
            "condition": {"type": "always"},
            "action": {"type": "multiply_forecast", "factor": 1.35},
            "rationale": "",
        },
        {
            "rule_id": "R2",
            "name": "Uplift B",
            "sku_pattern": "*",
            "condition": {"type": "always"},
            "action": {"type": "multiply_forecast", "factor": 1.10},
            "rationale": "",
        },
    ]
    row = make_row(forecast_q50=100.0, forecast_q90=130.0)
    result = apply_rules(row, rules=rules)
    # 100 × 1.35 × 1.10 = 148.5
    assert abs(result["adjusted_q50"] - 148.5) < 0.01
    assert "R1" in result["rules_applied"]
    assert "R2" in result["rules_applied"]


def test_dave_003_fires_in_summer_only(tmp_path):
    """DAVE_003 should only fire for perishable items in Australian summer months."""
    import json, importlib
    rules_data = [
        {
            "rule_id": "DAVE_003",
            "name": "Perishable bruising buffer",
            "sku_pattern": "perishable",
            "condition": {"type": "month_in", "months": [10, 11, 12, 1, 2, 3, 4]},
            "action": {"type": "multiply_order_qty", "factor": 1.01},
            "rationale": "Test",
        }
    ]
    summer_row = make_row(perishable=1, week_start=pd.Timestamp("2017-01-09"))
    winter_row = make_row(perishable=1, week_start=pd.Timestamp("2017-07-03"))

    summer_result = apply_rules(summer_row, rules_data)
    winter_result = apply_rules(winter_row, rules_data)

    assert "DAVE_003" in summer_result["rules_applied"]
    assert "DAVE_003" not in winter_result["rules_applied"]
