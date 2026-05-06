"""
Step 3: Feature Engineering
=============================
Transforms the weekly demand data into a feature matrix suitable for ML models.

CONCEPT — Feature Engineering for Time Series:
  Unlike standard tabular ML where rows are independent, time series data has
  temporal dependencies. We capture these through:

  1. **Lag features**: what was demand N weeks ago? This lets the model learn
     autoregressive patterns (e.g. "if demand was high last week, it'll likely
     be high this week too").

  2. **Rolling statistics**: moving averages and standard deviations over recent
     windows. These smooth out noise and capture trends.

  3. **Calendar features**: week-of-year, month, day-of-year. These capture
     seasonality (e.g. demand always rises in December).

  4. **External features**: promotions, holidays, and any other signals that
     affect demand but aren't part of the historical sales data.

Key output: data/processed/features.parquet
  A "flat" table where each row is one (store, item, week) with:
    - target: unit_sales (what we're predicting)
    - features: lags, rolling stats, calendar, promo, holiday, etc.

Usage:
    python src/03_feature_engineering.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features: demand from N weeks ago.

    LAG FEATURES EXPLAINED:
      - lag_1: last week's demand → most predictive for short-term
      - lag_2: two weeks ago → captures fortnightly patterns
      - lag_4: four weeks ago → captures monthly patterns
      - lag_52: same week last year → captures yearly seasonality

    We compute these per (store, item) group so that store 1's lag doesn't
    leak into store 2's features.
    """
    print("  Adding lag features...")

    group_cols = ["store_nbr", "item_nbr"]

    for lag in [1, 2, 4, 8, 52]:
        col_name = f"demand_lag_{lag}"
        df[col_name] = df.groupby(group_cols)["unit_sales"].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window statistics.

    ROLLING FEATURES EXPLAINED:
      - rolling_4w_mean: average demand over the last 4 weeks.
        This smooths out weekly noise and captures the recent trend.
      - rolling_4w_std: standard deviation over 4 weeks.
        High std = volatile demand → need more safety stock.
      - rolling_12w_mean: 3-month average → captures medium-term trend.

    The `shift(1)` is critical: it ensures we only use information available
    *before* the current week. Without it, we'd have data leakage (using
    the current week's actual sales to predict the current week).
    """
    print("  Adding rolling features...")

    group_cols = ["store_nbr", "item_nbr"]

    for window in [4, 12]:
        # shift(1) to avoid leakage: only use past data
        rolled = df.groupby(group_cols)["unit_sales"].shift(1).rolling(window)

        df[f"rolling_{window}w_mean"] = (
            df.groupby(group_cols)["unit_sales"]
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df[f"rolling_{window}w_std"] = (
            df.groupby(group_cols)["unit_sales"]
            .transform(lambda x: x.shift(1).rolling(window).std())
        )

    # Coefficient of variation (CV) = std/mean
    # CV is a normalised measure of demand volatility:
    #   - CV < 0.5: stable demand → easier to forecast
    #   - CV > 1.0: highly volatile → need larger safety stock
    df["demand_cv_4w"] = df["rolling_4w_std"] / df["rolling_4w_mean"].replace(0, np.nan)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features.

    CALENDAR FEATURES EXPLAINED:
      These capture seasonality — repeating patterns tied to the calendar.
      - week_of_year (1-52): captures yearly seasonality (Christmas, summer, etc.)
      - month (1-12): coarser version of weekly seasonality
      - is_month_start/end: payroll effects (Dave's Rule #006: payday bump)
      - quarter (1-4): captures quarterly business cycles
    """
    print("  Adding calendar features...")

    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["year"] = df["week_start"].dt.year

    # Is this the first or last week of the month? (payday effect)
    df["day_of_month"] = df["week_start"].dt.day
    df["is_month_start"] = (df["day_of_month"] <= 7).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 24).astype(int)

    # Cyclical encoding for week_of_year and month
    # Why cyclical? Week 52 and week 1 are adjacent, but numerically far apart.
    # Sin/cos encoding fixes this: sin(2π × 52/52) ≈ sin(2π × 0/52) ≈ 0.
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add promotion-derived features.

    Beyond the raw promo_intensity (what fraction of days had a promo this week),
    we add:
      - was_on_promo_last_week: sometimes demand *drops* after a promo ends
        because customers stocked up (the "post-promo dip")
      - promo_change: did promo status change? Transitions often have bigger
        effects than steady-state promo.
    """
    print("  Adding promotion features...")

    group_cols = ["store_nbr", "item_nbr"]

    df["promo_last_week"] = df.groupby(group_cols)["promo_intensity"].shift(1)
    df["promo_change"] = df["promo_intensity"] - df["promo_last_week"]

    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where lag/rolling features are NaN (the "warm-up" period).

    The first 52 weeks of each store×item series will have NaN for lag_52.
    Rather than imputing (which adds noise), we drop them. This is acceptable
    because we have years of data — losing the first year still leaves plenty.
    """
    before = len(df)
    df = df.dropna(subset=["demand_lag_52", "rolling_12w_mean"]).copy()
    after = len(df)
    print(f"  Dropped {before - after:,} warm-up rows ({(before-after)/before*100:.1f}%)")
    return df


def main():
    print("=" * 60)
    print("Step 3: Feature Engineering")
    print("=" * 60)

    # Load weekly demand from Step 1
    weekly = pd.read_parquet(DATA_PROCESSED / "weekly_demand.parquet")

    # IMPORTANT: sort by store, item, date before computing lags/rolling
    # Otherwise shift() will mix up different time series
    weekly = weekly.sort_values(["store_nbr", "item_nbr", "week_start"]).reset_index(drop=True)

    # Build features
    weekly = add_lag_features(weekly)
    weekly = add_rolling_features(weekly)
    weekly = add_calendar_features(weekly)
    weekly = add_promo_features(weekly)

    # Drop warm-up rows
    weekly = drop_warmup_rows(weekly)

    # Save
    output_path = DATA_PROCESSED / "features.parquet"
    weekly.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"  Shape: {weekly.shape}")
    print(f"  Feature columns: {sorted([c for c in weekly.columns if c not in ['store_nbr', 'item_nbr', 'week_start', 'unit_sales', 'family', 'holiday_name']])}")
    print("\nDone! Next step: python src/04_demand_forecasting.py")


if __name__ == "__main__":
    main()
