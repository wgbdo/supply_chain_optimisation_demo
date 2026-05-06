"""
Step 7: Evaluation & Backtesting
==================================
Evaluates the full pipeline (forecast → rules → optimisation) against actual
historical demand to answer: "Would this system have performed better than Dave?"

CONCEPT — Backtesting:
  We held out the last HOLDOUT_WEEKS weeks from training (see Step 4). Now we
  compare what the model *recommended* ordering vs what *actually happened*.
  This is analogous to paper-trading in finance: test the strategy on historical
  data before deploying it with real money.

METRICS WE TRACK:

  Forecast quality:
    - MAPE (Mean Absolute Percentage Error): how accurate is the median forecast?
    - Prediction interval coverage: does the q10–q90 range actually contain ~80%
      of actuals? If much less, the model is overconfident.

  Inventory performance:
    - Fill rate: % of demand met from available stock. Target: >98%.
    - Stockout rate: % of store×item×weeks where order + on-hand < actual demand.
    - Waste rate: estimated wasted units as % of total ordered. Target: <3%.

  Financial:
    - Total procurement cost
    - Estimated waste cost
    - Estimated stockout cost
    - Total cost (the thing we're minimising)

Key output: data/processed/evaluation_report.parquet + console summary

Usage:
    python src/07_evaluation.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    AVG_PROCUREMENT_COST,
    DATA_PROCESSED,
    PLOTS_DIR,
    STOCKOUT_COST_PER_UNIT,
    WASTE_COST_PER_UNIT,
)


def compute_forecast_metrics(plan: pd.DataFrame) -> dict:
    """
    Evaluate forecast accuracy (before optimisation adjustments).

    These metrics assess the *ML model's* quality, independent of business rules
    and inventory decisions.
    """
    actuals = plan["actual_demand"]
    q50 = plan["forecast_q50"]
    q10 = plan.get("forecast_q10", pd.Series(dtype=float))
    q90 = plan["forecast_q90"]

    # MAPE: average percentage error. Exclude zeros to avoid division by zero.
    mask = actuals > 0
    mape = (np.abs(actuals[mask] - q50[mask]) / actuals[mask]).mean() * 100

    # Bias: positive = model over-forecasts, negative = under-forecasts
    bias = (q50 - actuals).mean()
    bias_pct = (bias / actuals.mean()) * 100

    # Coverage: what % of actuals fall within the q10–q90 interval?
    # For an 80% prediction interval, target coverage ≈ 80%.
    if "forecast_q10" in plan.columns:
        coverage = ((actuals >= plan["forecast_q10"]) & (actuals <= q90)).mean() * 100
    else:
        coverage = np.nan

    return {
        "MAPE (%)": round(mape, 2),
        "Bias (units)": round(bias, 1),
        "Bias (%)": round(bias_pct, 2),
        "80% Interval Coverage (%)": round(coverage, 2),
    }


def compute_inventory_metrics(plan: pd.DataFrame) -> dict:
    """
    Evaluate inventory decisions (the output of the optimisation).

    CONCEPT — Fill Rate vs Service Level:
      - Fill rate (volume-based): total units fulfilled / total units demanded.
        Example: supplied 9,800 out of 10,000 demanded → fill rate = 98%.
      - Stockout rate (incidence-based): % of store×item×weeks with ANY stockout.
        Example: 5 out of 200 decisions had a stockout → stockout rate = 2.5%.

      Both matter. Fill rate tells you the customer impact; stockout rate tells
      you how often the warehouse team has a problem to deal with.
    """
    # Available stock = what was ordered + what was already on hand
    plan["available"] = plan["order_qty"] + plan["on_hand"]
    plan["fulfilled"] = plan[["available", "actual_demand"]].min(axis=1)

    # Stockout: actual demand exceeded available stock
    plan["is_stockout"] = (plan["actual_demand"] > plan["available"]).astype(int)
    plan["stockout_units"] = (plan["actual_demand"] - plan["available"]).clip(lower=0)

    # Overstock: available exceeded actual demand (potential waste for perishables)
    plan["overstock_units"] = (plan["available"] - plan["actual_demand"]).clip(lower=0)

    # Waste estimate: for perishables, assume 40% of overstock becomes waste;
    # for non-perishables, 10% (due to damage, obsolescence, etc.)
    plan["waste_units"] = plan.apply(
        lambda r: r["overstock_units"] * (0.40 if r["perishable"] == 1 else 0.10),
        axis=1,
    )

    total_demand = plan["actual_demand"].sum()
    total_fulfilled = plan["fulfilled"].sum()
    total_ordered = plan["order_qty"].sum()
    total_stockout_units = plan["stockout_units"].sum()
    total_waste_units = plan["waste_units"].sum()

    return {
        "Fill Rate (%)": round(total_fulfilled / total_demand * 100, 2) if total_demand > 0 else 0,
        "Stockout Rate (%)": round(plan["is_stockout"].mean() * 100, 2),
        "Stockout Incidents": int(plan["is_stockout"].sum()),
        "Total Stockout Units": int(total_stockout_units),
        "Waste Rate (%)": round(total_waste_units / total_ordered * 100, 2) if total_ordered > 0 else 0,
        "Total Waste Units": int(total_waste_units),
    }


def compute_financial_metrics(plan: pd.DataFrame) -> dict:
    """
    Compute the financial impact of inventory decisions.

    This is the bottom line: how much does the whole thing cost?
    """
    total_procurement = plan["order_cost"].sum()
    total_waste_cost = plan["waste_units"].sum() * WASTE_COST_PER_UNIT
    total_stockout_cost = plan["stockout_units"].sum() * STOCKOUT_COST_PER_UNIT
    total_cost = total_procurement + total_waste_cost + total_stockout_cost

    return {
        "Total Procurement Cost ($)": round(total_procurement, 2),
        "Total Waste Cost ($)": round(total_waste_cost, 2),
        "Total Stockout Cost ($)": round(total_stockout_cost, 2),
        "Total Cost ($)": round(total_cost, 2),
        "Cost per Unit Demanded ($)": round(total_cost / plan["actual_demand"].sum(), 2) if plan["actual_demand"].sum() > 0 else 0,
    }


def compute_perishable_breakdown(plan: pd.DataFrame) -> None:
    """Print metrics split by perishable vs non-perishable."""
    for label, mask in [("Perishable", plan["perishable"] == 1), ("Non-Perishable", plan["perishable"] == 0)]:
        sub = plan[mask]
        if len(sub) == 0:
            continue

        total_demand = sub["actual_demand"].sum()
        fill = sub["fulfilled"].sum() / total_demand * 100 if total_demand > 0 else 0
        waste = sub["waste_units"].sum() / sub["order_qty"].sum() * 100 if sub["order_qty"].sum() > 0 else 0
        stockout = sub["is_stockout"].mean() * 100

        print(f"    {label:20s}: fill={fill:.1f}%, waste={waste:.1f}%, stockout_rate={stockout:.1f}%")


def plot_forecast_vs_actual(plan: pd.DataFrame):
    """
    Plot forecast vs actual demand for a sample item, showing prediction intervals.
    This is the classic "fan chart" that shows forecast uncertainty.
    """
    # Pick the item with the most data points for a nice plot
    item_counts = plan.groupby("item_nbr").size()
    sample_item = item_counts.idxmax()
    sample_store = plan[plan["item_nbr"] == sample_item]["store_nbr"].iloc[0]

    sub = plan[
        (plan["item_nbr"] == sample_item) & (plan["store_nbr"] == sample_store)
    ].sort_values("week_start")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(sub["week_start"], sub["actual_demand"], "k-o", label="Actual", markersize=4)
    ax.plot(sub["week_start"], sub["adjusted_q50"], "b--", label="Forecast (q50)", alpha=0.8)

    if "forecast_q10" in sub.columns:
        ax.fill_between(
            sub["week_start"],
            sub["forecast_q10"] if "forecast_q10" in sub.columns else sub["adjusted_q50"] * 0.8,
            sub["forecast_q90"],
            alpha=0.2,
            color="blue",
            label="80% prediction interval",
        )

    ax.set_title(f"Forecast vs Actual — Store {sample_store}, Item {sample_item}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Unit Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    print("  Saved: 08_forecast_vs_actual.png")


def plot_cost_breakdown(financial: dict):
    """Pie chart showing where the money goes."""
    labels = ["Procurement", "Waste", "Stockout"]
    values = [
        financial["Total Procurement Cost ($)"],
        financial["Total Waste Cost ($)"],
        financial["Total Stockout Cost ($)"],
    ]

    fig, ax = plt.subplots(figsize=(7, 7))
    colours = ["steelblue", "coral", "gold"]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colours, startangle=140)
    ax.set_title("Cost Breakdown")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_cost_breakdown.png", dpi=150)
    plt.close(fig)
    print("  Saved: 09_cost_breakdown.png")


def main():
    print("=" * 60)
    print("Step 7: Evaluation & Backtesting")
    print("=" * 60)

    # Load the order plan from Step 6
    plan = pd.read_parquet(DATA_PROCESSED / "order_plan.parquet")
    print(f"\nLoaded {len(plan):,} order decisions")

    # ── Forecast Metrics ──────────────────────────────────────────────
    print("\n── Forecast Quality ──")
    forecast_metrics = compute_forecast_metrics(plan)
    for name, val in forecast_metrics.items():
        print(f"  {name:35s}: {val}")

    # ── Inventory Metrics ──────────────────────────────────────────────
    print("\n── Inventory Performance ──")
    inventory_metrics = compute_inventory_metrics(plan)
    for name, val in inventory_metrics.items():
        print(f"  {name:35s}: {val}")

    print("\n  Breakdown by perishability:")
    compute_perishable_breakdown(plan)

    # ── Financial Metrics ──────────────────────────────────────────────
    print("\n── Financial Impact ──")
    financial_metrics = compute_financial_metrics(plan)
    for name, val in financial_metrics.items():
        print(f"  {name:35s}: {val}")

    # ── Plots ──────────────────────────────────────────────────────────
    print("\nGenerating evaluation plots...")
    plot_forecast_vs_actual(plan)
    plot_cost_breakdown(financial_metrics)

    # ── Save full evaluation results ───────────────────────────────────
    plan.to_parquet(DATA_PROCESSED / "evaluation_report.parquet", index=False)

    # Also save a summary as CSV for easy viewing
    summary = {**forecast_metrics, **inventory_metrics, **financial_metrics}
    summary_df = pd.DataFrame([summary])
    summary_path = DATA_PROCESSED / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # ── Interpret results ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Interpretation Guide")
    print("=" * 60)
    print("""
  MAPE < 20%:        Good for weekly SKU-level grocery forecasting.
  MAPE 20-35%:       Acceptable for volatile/promotional items.
  MAPE > 35%:        Investigate — may need more features or different model.

  Fill Rate > 98%:   Excellent service level.
  Fill Rate 95-98%:  Acceptable — minor stockouts.
  Fill Rate < 95%:   Needs attention — too many stockouts.

  Waste Rate < 3%:   Good for perishable food operations.
  Waste Rate 3-5%:   Typical for food distribution.
  Waste Rate > 5%:   Over-ordering — tighten safety stock or improve forecasts.

  The cost breakdown shows where to focus improvement efforts:
    - High waste cost → improve perishable forecasting or reduce safety stock
    - High stockout cost → increase safety stock or improve forecast coverage
    - High procurement cost → negotiate with suppliers or optimise MOQ
""")

    print("Done! Next step: streamlit run src/08_dashboard.py")


if __name__ == "__main__":
    main()
