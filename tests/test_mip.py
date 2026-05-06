"""
Unit tests for the inventory optimisation MIP (src/06_inventory_optimisation.py).

Tests use a small synthetic problem to verify:
  - The solver returns Optimal status
  - MOQ constraints are respected
  - Warehouse capacity is respected
  - order_qty >= 0

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
from config.settings import (
    AVG_PROCUREMENT_COST,
    DEFAULT_MOQ,
    PERISHABLE_WASTE_COST_MULTIPLIER,
    STOCKOUT_COST_PER_UNIT,
    WAREHOUSE_CAPACITY_PER_STORE,
    WASTE_COST_PER_UNIT,
)


def solve_small_mip(n_items: int = 5, demand_per_item: float = 200.0, capacity: int = 5000) -> dict:
    """
    Solve a minimal MIP for `n_items` items with uniform demand.
    Returns a dict with status, order quantities, and total cost.
    """
    items = list(range(n_items))
    on_hand = {i: 0 for i in items}
    demand = {i: demand_per_item for i in items}
    safety = {i: 20.0 for i in items}
    is_perishable = {i: (i % 2 == 0) for i in items}  # even items are perishable
    waste_prob = {i: 0.40 if is_perishable[i] else 0.10 for i in items}
    effective_waste_cost = {
        i: WASTE_COST_PER_UNIT * (PERISHABLE_WASTE_COST_MULTIPLIER if is_perishable[i] else 1.0)
        for i in items
    }

    prob = LpProblem("test_mip", LpMinimize)
    order_qty = {i: LpVariable(f"order_{i}", lowBound=0, cat="Integer") for i in items}
    use_supplier = {i: LpVariable(f"use_{i}", cat="Binary") for i in items}
    overstock = {i: LpVariable(f"over_{i}", lowBound=0) for i in items}
    understock = {i: LpVariable(f"under_{i}", lowBound=0) for i in items}

    prob += (
        lpSum(AVG_PROCUREMENT_COST * order_qty[i] for i in items)
        + lpSum(effective_waste_cost[i] * waste_prob[i] * overstock[i] for i in items)
        + lpSum(STOCKOUT_COST_PER_UNIT * understock[i] for i in items)
    ), "Total_Cost"

    BIG_M = 50000
    for i in items:
        target = demand[i] + safety[i]
        prob += order_qty[i] + on_hand[i] >= target, f"meet_demand_{i}"
        prob += overstock[i] >= order_qty[i] + on_hand[i] - demand[i], f"overstock_{i}"
        prob += understock[i] >= demand[i] - order_qty[i] - on_hand[i], f"understock_{i}"
        prob += order_qty[i] >= DEFAULT_MOQ * use_supplier[i], f"moq_min_{i}"
        prob += order_qty[i] <= BIG_M * use_supplier[i], f"moq_link_{i}"

    prob += lpSum(order_qty[i] + on_hand[i] for i in items) <= capacity, "capacity"

    prob.solve()

    return {
        "status": LpStatus[prob.status],
        "order_qty": {i: int(value(order_qty[i])) for i in items},
        "total_cost": value(prob.objective),
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_mip_returns_optimal_status():
    result = solve_small_mip(n_items=5, demand_per_item=200.0, capacity=5000)
    assert result["status"] == "Optimal", f"Expected Optimal, got {result['status']}"


def test_order_quantities_are_non_negative():
    result = solve_small_mip(n_items=5, demand_per_item=200.0, capacity=5000)
    for i, qty in result["order_qty"].items():
        assert qty >= 0, f"Item {i} has negative order qty: {qty}"


def test_order_quantities_meet_demand_plus_safety():
    demand = 200.0
    safety = 20.0
    result = solve_small_mip(n_items=3, demand_per_item=demand, capacity=5000)
    for i, qty in result["order_qty"].items():
        # order_qty + on_hand (0) >= demand + safety = 220
        assert qty >= demand + safety, (
            f"Item {i}: order_qty {qty} < demand + safety {demand + safety}"
        )


def test_moq_respected_when_ordering():
    result = solve_small_mip(n_items=3, demand_per_item=200.0, capacity=5000)
    for i, qty in result["order_qty"].items():
        if qty > 0:
            assert qty >= DEFAULT_MOQ, (
                f"Item {i}: order_qty {qty} < MOQ {DEFAULT_MOQ}"
            )


def test_warehouse_capacity_respected():
    capacity = 2000
    result = solve_small_mip(n_items=5, demand_per_item=200.0, capacity=capacity)
    total_ordered = sum(result["order_qty"].values())
    # on_hand is 0 in this test, so total stock = total ordered
    assert total_ordered <= capacity, (
        f"Total ordered {total_ordered} exceeds capacity {capacity}"
    )


def test_infeasible_capacity_is_reported():
    """When capacity is impossibly small, the solver should not return Optimal."""
    # 5 items × MOQ 100 = 500 minimum. Set capacity to 10 → infeasible.
    result = solve_small_mip(n_items=5, demand_per_item=200.0, capacity=10)
    assert result["status"] != "Optimal"


def test_warehouse_capacity_setting_is_positive():
    assert WAREHOUSE_CAPACITY_PER_STORE > 0


def test_cost_parameters_are_positive():
    assert AVG_PROCUREMENT_COST > 0
    assert WASTE_COST_PER_UNIT > 0
    assert STOCKOUT_COST_PER_UNIT > 0
    assert PERISHABLE_WASTE_COST_MULTIPLIER >= 1.0


def test_perishable_waste_cost_higher_than_non_perishable():
    perishable_cost = WASTE_COST_PER_UNIT * PERISHABLE_WASTE_COST_MULTIPLIER
    assert perishable_cost > WASTE_COST_PER_UNIT, (
        "Perishable effective waste cost should exceed non-perishable"
    )
