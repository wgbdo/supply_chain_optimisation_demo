"""
Step 6: Inventory Optimisation
================================
Uses mathematical optimisation (Mixed-Integer Programming / MIP) to determine
optimal order quantities for each store×item, given:
  - Probabilistic demand forecasts (q50, q90) from Step 4
  - Business rule adjustments from Step 5
  - Cost parameters (procurement, waste, stockout)
  - Operational constraints (MOQ, shelf life, warehouse capacity)

CONCEPT — What is Mathematical Optimisation?
  Optimisation finds the "best" decision (e.g. how much to order) subject to
  constraints (e.g. warehouse capacity). It has three components:

  1. DECISION VARIABLES: the things you're choosing
     → order_qty[store, item]: how many units to order for each store×item

  2. OBJECTIVE FUNCTION: what you're trying to minimise (or maximise)
     → minimise: procurement_cost + expected_waste_cost + expected_stockout_cost

  3. CONSTRAINTS: rules the solution must obey
     → order_qty >= MOQ (if ordering at all)
     → total units in warehouse <= capacity
     → order_qty must be a non-negative integer

CONCEPT — Why MIP (Mixed-Integer Programming)?
  "Mixed-Integer" means some variables are continuous (e.g. cost) and some are
  integers (e.g. order quantities — you can't order 500.7 units). MIP solvers
  like CBC (open-source, bundled with PuLP) and HiGHS efficiently search the
  space of feasible integer solutions.

CONCEPT — The Newsvendor Trade-Off:
  The core trade-off in inventory management is:
    - Order too much → waste (especially perishables that expire)
    - Order too little → stockouts (lost sales, unhappy customers)

  The optimal order quantity depends on the ratio of these costs:
    Critical Ratio = stockout_cost / (stockout_cost + waste_cost)

  If stockout_cost = $10 and waste_cost = $3.50:
    Critical Ratio = 10 / (10 + 3.5) = 0.74

  This means you should target a service level of ~74% — i.e. order enough to
  cover the 74th percentile of demand. Since stockouts are more expensive than
  waste, you lean towards ordering more.

Key output: data/processed/order_plan.parquet
  Columns: store_nbr, item_nbr, week_start, order_qty, order_cost,
           expected_waste, expected_stockout, total_cost, ...

Usage:
    python src/06_inventory_optimisation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    AVG_PROCUREMENT_COST,
    DATA_PROCESSED,
    DEFAULT_MOQ,
    NON_PERISHABLE_SHELF_LIFE_DAYS,
    PERISHABLE_SHELF_LIFE_DAYS,
    STOCKOUT_COST_PER_UNIT,
    WAREHOUSE_CAPACITY_PER_STORE,
    WASTE_COST_PER_UNIT,
)


def compute_on_hand_inventory(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate current on-hand inventory for each store×item.

    In a real system, this comes from SAP table MARD (stock per storage location).
    For the demo, we simulate it as a fraction of recent demand — this is roughly
    what you'd see mid-week before the next delivery arrives.
    """
    rng = np.random.default_rng(42)

    # Simulate on-hand as 20-50% of last week's demand (mid-cycle stock)
    forecasts["on_hand"] = (
        forecasts["demand_lag_1"] * rng.uniform(0.2, 0.5, size=len(forecasts))
    ).clip(lower=0).round().astype(int)

    return forecasts


def compute_safety_stock(row: pd.Series) -> float:
    """
    Compute safety stock for a single store×item×week.

    CONCEPT — Safety Stock:
      Safety stock is the buffer inventory held to protect against demand
      variability. It's derived from the gap between the median forecast (q50)
      and the high-scenario forecast (q90):

        safety_stock = (q90 - q50) × safety_stock_multiplier

      The safety_stock_multiplier comes from Dave's business rules (e.g.
      ×1.5 for new products). Default is 1.0.

      For perishable items, we cap safety stock to avoid ordering so much that
      it expires before we can sell it.
    """
    base_safety = max(0, row["adjusted_q90"] - row["adjusted_q50"])
    multiplier = row.get("safety_stock_multiplier", 1.0)
    safety = base_safety * multiplier

    # Cap safety stock for perishables: no point holding more than ~2 days of
    # extra demand if the item expires in 7 days
    if row.get("perishable", 0) == 1:
        max_safety = row["adjusted_q50"] * (2 / 7)  # ~2 days worth
        safety = min(safety, max_safety)

    return round(safety)


def optimise_single_store(
    store_data: pd.DataFrame, store_nbr: int
) -> pd.DataFrame:
    """
    Solve the inventory optimisation problem for a single store.

    We solve one week at a time. For each week, the decision is:
    "How much of each item should we order?"

    FORMULATION:
      Minimise:
        Σ_i [ procurement_cost × order_qty[i]
              + waste_cost × overstock[i] × waste_probability
              + stockout_cost × understock[i] ]

      Subject to:
        order_qty[i] + on_hand[i] >= demand_q50[i] + safety_stock[i]   (meet demand)
        overstock[i] >= order_qty[i] + on_hand[i] - demand_q50[i]      (define overstock)
        understock[i] >= demand_q50[i] - order_qty[i] - on_hand[i]     (define understock)
        order_qty[i] >= MOQ × use_supplier[i]                          (MOQ if ordering)
        order_qty[i] <= BIG_M × use_supplier[i]                        (link binary)
        Σ_i (order_qty[i] + on_hand[i]) <= warehouse_capacity          (capacity)
        order_qty[i] >= 0, integer
    """
    weeks = store_data["week_start"].unique()
    results = []

    for week in sorted(weeks):
        week_data = store_data[store_data["week_start"] == week].copy()

        if len(week_data) == 0:
            continue

        # Compute safety stock for each item
        week_data["safety_stock"] = week_data.apply(compute_safety_stock, axis=1)

        items = week_data["item_nbr"].values
        n_items = len(items)

        # ── Build the optimisation model ──────────────────────────────────

        # Create the problem. "LpMinimize" tells PuLP we want to minimise cost.
        prob = LpProblem(f"Store{store_nbr}_Week{week.strftime('%Y%m%d')}", LpMinimize)

        # Decision variables:
        # order_qty[i]: how many units of item i to order (non-negative integer)
        order_qty = {
            i: LpVariable(f"order_{i}", lowBound=0, cat="Integer")
            for i in items
        }

        # use_supplier[i]: binary flag — are we ordering item i at all?
        # This is needed to enforce MOQ: "if you order, order at least MOQ"
        use_supplier = {
            i: LpVariable(f"use_{i}", cat="Binary")
            for i in items
        }

        # Auxiliary variables for overstock and understock (for cost computation)
        overstock = {
            i: LpVariable(f"over_{i}", lowBound=0) for i in items
        }
        understock = {
            i: LpVariable(f"under_{i}", lowBound=0) for i in items
        }

        # Look up data for each item (as dicts for fast access in the loop)
        on_hand = dict(zip(week_data["item_nbr"], week_data["on_hand"]))
        demand = dict(zip(week_data["item_nbr"], week_data["adjusted_q50"]))
        safety = dict(zip(week_data["item_nbr"], week_data["safety_stock"]))
        is_perishable = dict(zip(week_data["item_nbr"], week_data["perishable"]))

        # Waste probability: perishables have a higher chance of overstock turning
        # into waste (shorter shelf life). We use 40% for perishable, 10% for non.
        waste_prob = {
            i: 0.40 if is_perishable.get(i, 0) == 1 else 0.10
            for i in items
        }

        # Order qty multiplier from business rules (e.g. bruising buffer)
        oqm = dict(zip(week_data["item_nbr"], week_data.get("order_qty_multiplier", pd.Series(1.0, index=week_data.index))))

        # ── Objective function ────────────────────────────────────────────
        # Total cost = procurement + expected waste + expected stockout
        prob += (
            lpSum(AVG_PROCUREMENT_COST * order_qty[i] for i in items)
            + lpSum(WASTE_COST_PER_UNIT * waste_prob[i] * overstock[i] for i in items)
            + lpSum(STOCKOUT_COST_PER_UNIT * understock[i] for i in items)
        ), "Total_Cost"

        # ── Constraints ───────────────────────────────────────────────────

        BIG_M = 50000  # Upper bound on order quantity (must be large enough)

        for i in items:
            target = demand[i] + safety[i]

            # Meet demand + safety stock
            prob += (
                order_qty[i] + on_hand[i] >= target,
                f"meet_demand_{i}",
            )

            # Define overstock (excess above demand)
            prob += (
                overstock[i] >= order_qty[i] + on_hand[i] - demand[i],
                f"overstock_def_{i}",
            )

            # Define understock (shortfall below demand)
            prob += (
                understock[i] >= demand[i] - order_qty[i] - on_hand[i],
                f"understock_def_{i}",
            )

            # MOQ enforcement: if ordering, order at least MOQ
            prob += (
                order_qty[i] >= DEFAULT_MOQ * use_supplier[i],
                f"moq_min_{i}",
            )
            prob += (
                order_qty[i] <= BIG_M * use_supplier[i],
                f"moq_link_{i}",
            )

        # Warehouse capacity: total stock after delivery <= capacity
        prob += (
            lpSum(order_qty[i] + on_hand[i] for i in items) <= WAREHOUSE_CAPACITY_PER_STORE,
            "warehouse_capacity",
        )

        # ── Solve ─────────────────────────────────────────────────────────
        # CBC is the default open-source solver bundled with PuLP.
        # For larger problems, you'd switch to HiGHS or Gurobi.
        prob.solve()  # uses default CBC solver

        if LpStatus[prob.status] != "Optimal":
            print(f"    WARNING: Store {store_nbr}, Week {week.date()}: solver status = {LpStatus[prob.status]}")
            continue

        # ── Extract results ───────────────────────────────────────────────
        for _, row in week_data.iterrows():
            i = row["item_nbr"]
            oq = int(value(order_qty[i]))

            # Apply order quantity multiplier from business rules
            # (e.g. Dave's perishable bruising buffer: ×1.15)
            oq_adjusted = int(oq * oqm.get(i, 1.0))

            results.append(
                {
                    "store_nbr": store_nbr,
                    "item_nbr": i,
                    "week_start": week,
                    "family": row["family"],
                    "perishable": row["perishable"],
                    "on_hand": on_hand[i],
                    "forecast_q50": row["forecast_q50"],
                    "forecast_q90": row["forecast_q90"],
                    "adjusted_q50": row["adjusted_q50"],
                    "adjusted_q90": row["adjusted_q90"],
                    "safety_stock": safety[i],
                    "order_qty": oq_adjusted,
                    "order_cost": oq_adjusted * AVG_PROCUREMENT_COST,
                    "expected_overstock": max(0, oq_adjusted + on_hand[i] - demand[i]),
                    "expected_waste": max(0, oq_adjusted + on_hand[i] - demand[i]) * waste_prob[i],
                    "actual_demand": row["unit_sales"],  # for backtesting
                    "rules_applied": row.get("rules_applied", ""),
                    "explanations": row.get("explanations", ""),
                }
            )

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Step 6: Inventory Optimisation (PuLP MIP)")
    print("=" * 60)

    # Load adjusted forecasts from Step 5
    forecasts = pd.read_parquet(DATA_PROCESSED / "adjusted_forecasts.parquet")
    print(f"\nLoaded {len(forecasts):,} adjusted forecast rows")

    # Simulate on-hand inventory (in production this comes from SAP MARD)
    forecasts = compute_on_hand_inventory(forecasts)

    # Solve per store (each store is an independent optimisation problem)
    stores = sorted(forecasts["store_nbr"].unique())
    all_results = []

    for store_nbr in stores:
        store_data = forecasts[forecasts["store_nbr"] == store_nbr]
        n_weeks = store_data["week_start"].nunique()
        print(f"\n  Optimising Store {store_nbr} ({n_weeks} weeks, {len(store_data)} rows)...")
        result = optimise_single_store(store_data, store_nbr)
        all_results.append(result)
        print(f"    → {len(result)} order decisions, total cost: ${result['order_cost'].sum():,.0f}")

    order_plan = pd.concat(all_results, ignore_index=True)

    # ── Summary statistics ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Order Plan Summary")
    print("=" * 60)
    print(f"  Total order decisions:     {len(order_plan):,}")
    print(f"  Total procurement cost:    ${order_plan['order_cost'].sum():,.0f}")
    print(f"  Total expected waste:      {order_plan['expected_waste'].sum():,.0f} units")
    print(f"  Avg order qty per item:    {order_plan['order_qty'].mean():,.0f} units")
    print(f"  Items with zero order:     {(order_plan['order_qty'] == 0).sum():,}")

    # Waste rate: expected wasted units / total ordered units
    total_ordered = order_plan["order_qty"].sum()
    total_waste = order_plan["expected_waste"].sum()
    if total_ordered > 0:
        print(f"  Expected waste rate:       {total_waste / total_ordered * 100:.1f}%")

    # Save
    output_path = DATA_PROCESSED / "order_plan.parquet"
    order_plan.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    print("\nDone! Next step: python src/07_evaluation.py")


if __name__ == "__main__":
    main()
