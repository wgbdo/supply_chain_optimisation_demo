"""
Step 2: Exploratory Data Analysis (EDA)
========================================
Generates visualisations to understand demand patterns before modelling.

This step answers questions like:
  - What does weekly demand look like over time?
  - Is there seasonality? (weekly, monthly, yearly)
  - How do perishable vs non-perishable items differ?
  - What's the effect of promotions?
  - How much variance is there across stores?

Plots are saved to data/processed/plots/ for reference.

Usage:
    python src/02_eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, PLOTS_DIR


def plot_total_demand_over_time(weekly: pd.DataFrame):
    """
    Plot total weekly demand across all stores and items.
    Look for: trend (up/down), seasonality (repeating pattern), outliers.
    """
    total = weekly.groupby("week_start")["unit_sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(total["week_start"], total["unit_sales"], linewidth=0.8)
    ax.set_title("Total Weekly Demand (All Stores × All Items)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Unit Sales")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_total_demand.png", dpi=150)
    plt.close(fig)
    print("  Saved: 01_total_demand.png")


def plot_perishable_vs_nonperishable(weekly: pd.DataFrame):
    """
    Compare demand patterns for perishable vs non-perishable items.

    Why this matters for supply chain:
    - Perishable items have shorter shelf life → less room for error
    - Waste cost is higher for perishables
    - You need tighter forecasts and faster replenishment cycles
    """
    grouped = (
        weekly.groupby(["week_start", "perishable"])["unit_sales"]
        .sum()
        .reset_index()
    )
    grouped["category"] = grouped["perishable"].map(
        {0: "Non-Perishable", 1: "Perishable"}
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    for cat, sub in grouped.groupby("category"):
        ax.plot(sub["week_start"], sub["unit_sales"], label=cat, linewidth=0.8)

    ax.set_title("Weekly Demand: Perishable vs Non-Perishable")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Unit Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_perishable_vs_nonperishable.png", dpi=150)
    plt.close(fig)
    print("  Saved: 02_perishable_vs_nonperishable.png")


def plot_promotion_effect(weekly: pd.DataFrame):
    """
    Visualise how promotions affect demand.

    We bucket promo_intensity into "No Promo" (<0.1) and "On Promo" (>=0.5)
    and compare the distribution of unit_sales.
    """
    df = weekly.copy()
    df["promo_bucket"] = "Partial/None"
    df.loc[df["promo_intensity"] < 0.1, "promo_bucket"] = "No Promo"
    df.loc[df["promo_intensity"] >= 0.5, "promo_bucket"] = "On Promo"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df[df["promo_bucket"].isin(["No Promo", "On Promo"])],
        x="promo_bucket",
        y="unit_sales",
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Demand Distribution: Promo vs No Promo")
    ax.set_ylabel("Weekly Unit Sales per Store×Item")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_promotion_effect.png", dpi=150)
    plt.close(fig)
    print("  Saved: 03_promotion_effect.png")


def plot_demand_by_family(weekly: pd.DataFrame):
    """
    Compare demand across product families.
    Helps prioritise which families need the most forecasting attention.
    """
    family_totals = (
        weekly.groupby("family")["unit_sales"]
        .sum()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    family_totals.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Total Demand by Product Family")
    ax.set_xlabel("Total Unit Sales")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_demand_by_family.png", dpi=150)
    plt.close(fig)
    print("  Saved: 04_demand_by_family.png")


def plot_demand_by_store(weekly: pd.DataFrame):
    """
    Compare demand across stores. Large variance between stores might
    suggest store-level features (size, location) matter for forecasting.
    """
    store_weekly = (
        weekly.groupby(["week_start", "store_nbr"])["unit_sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    for store, sub in store_weekly.groupby("store_nbr"):
        ax.plot(sub["week_start"], sub["unit_sales"], label=f"Store {store}", linewidth=0.7, alpha=0.8)

    ax.set_title("Weekly Demand by Store")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Unit Sales")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_demand_by_store.png", dpi=150)
    plt.close(fig)
    print("  Saved: 05_demand_by_store.png")


def plot_holiday_effect(weekly: pd.DataFrame):
    """
    Compare demand in holiday weeks vs normal weeks.
    This validates whether Dave's holiday uplift rules make sense.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    weekly_total = weekly.groupby(["week_start", "is_holiday_week"])["unit_sales"].sum().reset_index()
    weekly_total["week_type"] = weekly_total["is_holiday_week"].map(
        {0: "Normal Week", 1: "Holiday Week"}
    )

    sns.boxplot(data=weekly_total, x="week_type", y="unit_sales", ax=ax, showfliers=False)
    ax.set_title("Total Demand: Holiday Weeks vs Normal Weeks")
    ax.set_ylabel("Total Weekly Unit Sales")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_holiday_effect.png", dpi=150)
    plt.close(fig)
    print("  Saved: 06_holiday_effect.png")


def print_summary_stats(weekly: pd.DataFrame):
    """Print high-level statistics about the dataset."""
    print("\n  Dataset Summary:")
    print(f"    Date range:    {weekly['week_start'].min().date()} → {weekly['week_start'].max().date()}")
    print(f"    Num weeks:     {weekly['week_start'].nunique()}")
    print(f"    Num stores:    {weekly['store_nbr'].nunique()}")
    print(f"    Num items:     {weekly['item_nbr'].nunique()}")
    print(f"    Num families:  {weekly['family'].nunique()}")
    print(f"    Total rows:    {len(weekly):,}")
    print(f"    Perishable %:  {weekly['perishable'].mean()*100:.1f}%")
    print(f"    Avg weekly demand per store×item: {weekly['unit_sales'].mean():.1f}")
    print(f"    Median promo intensity: {weekly['promo_intensity'].median():.2f}")


def main():
    print("=" * 60)
    print("Step 2: Exploratory Data Analysis")
    print("=" * 60)

    weekly = pd.read_parquet(DATA_PROCESSED / "weekly_demand.parquet")
    print_summary_stats(weekly)

    print("\nGenerating plots...")
    plot_total_demand_over_time(weekly)
    plot_perishable_vs_nonperishable(weekly)
    plot_promotion_effect(weekly)
    plot_demand_by_family(weekly)
    plot_demand_by_store(weekly)
    plot_holiday_effect(weekly)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
    print("Done! Next step: python src/03_feature_engineering.py")


if __name__ == "__main__":
    main()
