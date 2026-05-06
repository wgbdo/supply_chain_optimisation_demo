"""
Step 1: Data Preparation
=========================
Loads raw CSV files (either Kaggle Favorita data or synthetic data),
cleans them, and aggregates daily sales into weekly demand per store×item.

This is analogous to extracting data from SAP and preparing it for modelling:
  - SAP table VBAK/VBAP (sales orders)  →  train.csv
  - SAP table MARA/MARC (material master) →  items.csv
  - SAP table LFA1 (vendor master)       →  stores.csv (used as location proxy)

Key output: data/processed/weekly_demand.parquet
  Columns: store_nbr, item_nbr, week_start, unit_sales, family, perishable, ...

Usage:
    python src/01_data_prep.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import AGG_FREQ, DATA_PROCESSED, MIN_MEDIAN_WEEKLY_DEMAND, N_ITEMS, N_STORES, RAW_DATA_SOURCE


def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Load all raw CSVs into a dictionary of DataFrames.

    In a real SAP integration, this is where you'd connect to SAP via
    CDS Views / OData / Fivetran and pull the equivalent tables.
    """
    print("Loading raw data...")
    print(f"  Source: {RAW_DATA_SOURCE}")

    # The main sales file can be large (~125M rows for real Favorita data).
    # parse_dates on 125M rows causes a 957 MB intermediate allocation; instead
    # we load date as str and convert AFTER filtering down to the subset we need.
    train = pd.read_csv(
        RAW_DATA_SOURCE / "train.csv",
        dtype={
            "date": "str",
            "store_nbr": "int16",
            "item_nbr": "int32",
            "onpromotion": "object",  # may have NaNs in real data
        },
    )

    stores = pd.read_csv(RAW_DATA_SOURCE / "stores.csv")
    items = pd.read_csv(RAW_DATA_SOURCE / "items.csv")

    holidays = pd.read_csv(
        RAW_DATA_SOURCE / "holidays_events.csv", parse_dates=["date"]
    )

    print(f"  train.csv:    {len(train):>12,} rows")
    print(f"  stores.csv:   {len(stores):>12,} rows")
    print(f"  items.csv:    {len(items):>12,} rows")
    print(f"  holidays:     {len(holidays):>12,} rows")

    return {
        "train": train,
        "stores": stores,
        "items": items,
        "holidays": holidays,
    }


def filter_top_items(train: pd.DataFrame, items: pd.DataFrame) -> tuple:
    """
    Keep only the top N_ITEMS by total sales volume and top N_STORES.

    Why: The full Favorita dataset has ~4,000 items × 54 stores.
    For a PoC, we focus on a subset to keep runtimes reasonable.
    In production, you'd run this on the full dataset.
    """
    print(f"\nFiltering to top {N_ITEMS} items and top {N_STORES} stores...")

    # Top stores by transaction volume
    top_stores = (
        train.groupby("store_nbr")["unit_sales"]
        .sum()
        .nlargest(N_STORES)
        .index.tolist()
    )

    # Exclude intermittent items before ranking.
    # Items with low average weekly demand are unsuitable for LightGBM quantile
    # regression (insufficient signal) and are better handled by Croston's method.
    # ISO date strings (YYYY-MM-DD) sort correctly, so string min/max is accurate.
    date_min = pd.to_datetime(train["date"].min())
    date_max = pd.to_datetime(train["date"].max())
    date_range = date_max - date_min
    n_weeks = max(1, date_range.days / 7)
    item_avg_weekly = train.groupby("item_nbr")["unit_sales"].sum() / n_weeks
    non_intermittent_items = item_avg_weekly[item_avg_weekly >= MIN_MEDIAN_WEEKLY_DEMAND].index
    n_intermittent = len(item_avg_weekly) - len(non_intermittent_items)
    print(f"  Excluded {n_intermittent:,} intermittent items (avg weekly demand < {MIN_MEDIAN_WEEKLY_DEMAND} units)")

    # Top items by sales volume — ranked only among non-intermittent items
    top_items = (
        train[train["item_nbr"].isin(non_intermittent_items)]
        .groupby("item_nbr")["unit_sales"]
        .sum()
        .nlargest(N_ITEMS)
        .index.tolist()
    )

    train_filtered = train[
        train["store_nbr"].isin(top_stores) & train["item_nbr"].isin(top_items)
    ].copy()

    items_filtered = items[items["item_nbr"].isin(top_items)].copy()

    print(f"  Kept {len(train_filtered):,} rows ({len(train_filtered)/len(train)*100:.1f}% of original)")

    return train_filtered, items_filtered, top_stores


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the sales data:
      - Handle negative unit_sales (returns) by clipping to 0
      - Fill NaN in onpromotion (the real Favorita data has ~16% NaN here)
      - Ensure consistent dtypes
    """
    print("\nCleaning sales data...")

    # Negative sales = returns. For demand forecasting we want gross demand,
    # so we clip negatives to 0. A more sophisticated approach would model
    # returns separately.
    n_negative = (df["unit_sales"] < 0).sum()
    if n_negative > 0:
        print(f"  Clipped {n_negative:,} negative sales (returns) to 0")
    df["unit_sales"] = df["unit_sales"].clip(lower=0)

    # Fill promotion NaNs: assume no promotion if unknown
    df["onpromotion"] = (
        df["onpromotion"]
        .fillna(False)
        .replace({"True": True, "False": False, "1": True, "0": False})
        .astype(bool)
    )

    return df


def aggregate_to_weekly(
    daily: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate daily sales to weekly totals per store×item.

    Why weekly?
    - Food box companies typically make ordering decisions weekly
    - Weekly data is smoother and easier to forecast than daily
    - Reduces data volume significantly (~7x fewer rows)

    The `AGG_FREQ = "W-MON"` means weeks start on Monday.
    Pandas' resample/Grouper snaps each date to the Monday of its week.
    """
    print(f"\nAggregating daily → weekly ({AGG_FREQ})...")

    weekly = (
        daily.groupby(
            [
                pd.Grouper(key="date", freq=AGG_FREQ),
                "store_nbr",
                "item_nbr",
            ]
        )
        .agg(
            unit_sales=("unit_sales", "sum"),
            # onpromotion: fraction of days in the week with a promo active
            promo_days=("onpromotion", "sum"),
            n_days=("onpromotion", "count"),
        )
        .reset_index()
    )

    weekly.rename(columns={"date": "week_start"}, inplace=True)

    # promo_intensity: 0.0 (no promo) to 1.0 (promo every day that week)
    weekly["promo_intensity"] = weekly["promo_days"] / weekly["n_days"]
    weekly.drop(columns=["promo_days", "n_days"], inplace=True)

    # Join item metadata (family, perishable flag)
    weekly = weekly.merge(items[["item_nbr", "family", "perishable"]], on="item_nbr", how="left")

    print(f"  Result: {len(weekly):,} weekly rows")
    print(f"  Date range: {weekly['week_start'].min()} → {weekly['week_start'].max()}")

    return weekly


def create_holiday_features(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Create a weekly holiday indicator.

    Returns a DataFrame with columns: week_start, is_holiday_week, holiday_name
    """
    holidays_weekly = holidays.copy()
    holidays_weekly["week_start"] = holidays_weekly["date"].dt.to_period("W-MON").dt.start_time

    holiday_weeks = (
        holidays_weekly.groupby("week_start")
        .agg(
            is_holiday_week=("date", "count"),
            holiday_name=("description", lambda x: ", ".join(x.unique())),
        )
        .reset_index()
    )
    holiday_weeks["is_holiday_week"] = 1

    return holiday_weeks


def main():
    print("=" * 60)
    print("Step 1: Data Preparation")
    print("=" * 60)

    # Load raw data
    data = load_raw_data()

    # Filter to top items/stores for PoC scope
    train_filtered, items_filtered, top_stores = filter_top_items(
        data["train"], data["items"]
    )

    # Convert date column now that we're on the small filtered subset.
    # Doing it before filtering would allocate ~957 MB for 125M rows.
    train_filtered["date"] = pd.to_datetime(train_filtered["date"])

    # Clean
    train_clean = clean_sales(train_filtered)

    # Aggregate to weekly
    weekly = aggregate_to_weekly(train_clean, items_filtered)

    # Create holiday features
    holiday_features = create_holiday_features(data["holidays"])

    # Merge holiday features onto weekly demand
    weekly = weekly.merge(holiday_features, on="week_start", how="left")
    weekly["is_holiday_week"] = weekly["is_holiday_week"].fillna(0).astype(int)
    weekly["holiday_name"] = weekly["holiday_name"].fillna("")

    # Save processed data
    output_path = DATA_PROCESSED / "weekly_demand.parquet"
    weekly.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Shape: {weekly.shape}")
    print(f"  Columns: {list(weekly.columns)}")

    # Also save store and item metadata for later use
    stores_filtered = data["stores"][data["stores"]["store_nbr"].isin(top_stores)]
    stores_filtered.to_parquet(DATA_PROCESSED / "stores.parquet", index=False)
    items_filtered.to_parquet(DATA_PROCESSED / "items.parquet", index=False)
    holiday_features.to_parquet(DATA_PROCESSED / "holidays.parquet", index=False)

    print("\nDone! Next step: python src/02_eda.py")


if __name__ == "__main__":
    main()
