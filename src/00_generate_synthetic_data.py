"""
Step 0: Generate Synthetic Data
================================
Creates synthetic grocery sales data that mimics the structure of the
Corporación Favorita dataset. Use this if you don't have a Kaggle account.

The generated data includes:
  - Daily unit sales for multiple stores × items
  - Seasonal patterns (weekly, monthly, yearly)
  - Promotional effects
  - Holiday effects
  - Perishable vs non-perishable items
  - Store metadata
  - Holiday calendar

Usage:
    python src/00_generate_synthetic_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_RAW,
    SYNTHETIC_END_DATE,
    SYNTHETIC_N_ITEMS,
    SYNTHETIC_N_STORES,
    SYNTHETIC_PERISHABLE_FRACTION,
    SYNTHETIC_START_DATE,
)


def generate_stores(n_stores: int) -> pd.DataFrame:
    """Generate store metadata similar to Favorita's stores.csv."""
    cities = ["Melbourne", "Sydney", "Brisbane", "Adelaide", "Perth"]
    states = ["VIC", "NSW", "QLD", "SA", "WA"]
    store_types = ["A", "B", "C", "D"]

    stores = pd.DataFrame(
        {
            "store_nbr": range(1, n_stores + 1),
            "city": [cities[i % len(cities)] for i in range(n_stores)],
            "state": [states[i % len(states)] for i in range(n_stores)],
            "type": [store_types[i % len(store_types)] for i in range(n_stores)],
            "cluster": [i % 3 + 1 for i in range(n_stores)],
        }
    )
    return stores


def generate_items(n_items: int, perishable_fraction: float) -> pd.DataFrame:
    """Generate item metadata similar to Favorita's items.csv."""
    families = [
        "BREAD/BAKERY",
        "DAIRY",
        "DELI",
        "FROZEN FOODS",
        "GROCERY I",
        "GROCERY II",
        "MEATS",
        "PRODUCE",
        "BEVERAGES",
        "CLEANING",
    ]

    # Items in these families are perishable
    perishable_families = {"BREAD/BAKERY", "DAIRY", "DELI", "MEATS", "PRODUCE"}

    rng = np.random.default_rng(42)
    n_perishable = int(n_items * perishable_fraction)

    items = []
    for i in range(1, n_items + 1):
        if i <= n_perishable:
            family = rng.choice([f for f in families if f in perishable_families])
            perishable = 1
        else:
            family = rng.choice([f for f in families if f not in perishable_families])
            perishable = 0

        items.append(
            {
                "item_nbr": i,
                "family": family,
                "class": rng.integers(1000, 9999),
                "perishable": perishable,
            }
        )

    return pd.DataFrame(items)


def generate_holidays() -> pd.DataFrame:
    """Generate an Australian-style holiday calendar."""
    holidays = []
    for year in range(2015, 2018):
        holidays.extend(
            [
                {
                    "date": f"{year}-01-01",
                    "type": "Holiday",
                    "locale": "National",
                    "description": "New Year's Day",
                    "transferred": False,
                },
                {
                    "date": f"{year}-01-26",
                    "type": "Holiday",
                    "locale": "National",
                    "description": "Australia Day",
                    "transferred": False,
                },
                {
                    "date": f"{year}-04-25",
                    "type": "Holiday",
                    "locale": "National",
                    "description": "ANZAC Day",
                    "transferred": False,
                },
                {
                    "date": f"{year}-12-25",
                    "type": "Holiday",
                    "locale": "National",
                    "description": "Christmas",
                    "transferred": False,
                },
                {
                    "date": f"{year}-12-26",
                    "type": "Holiday",
                    "locale": "National",
                    "description": "Boxing Day",
                    "transferred": False,
                },
            ]
        )

        # Easter (approximate — varies by year)
        easter_dates = {2015: "04-03", 2016: "03-25", 2017: "04-14"}
        if year in easter_dates:
            holidays.append(
                {
                    "date": f"{year}-{easter_dates[year]}",
                    "type": "Holiday",
                    "locale": "National",
                    "description": "Easter",
                    "transferred": False,
                }
            )

    df = pd.DataFrame(holidays)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_sales(
    stores: pd.DataFrame,
    items: pd.DataFrame,
    holidays: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Generate synthetic daily sales data with realistic patterns.

    Patterns injected:
      - Base demand varies by item (some items sell more than others)
      - Day-of-week effect (weekends higher)
      - Monthly seasonality (Dec spike for Christmas)
      - Yearly trend (slight growth)
      - Random promotions (~15% of days)
      - Promotional uplift (+30-60% when on promo)
      - Holiday effects (+20-40% near holidays)
      - Noise
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Pre-compute holiday proximity (1 if within 7 days of a holiday, else 0)
    holiday_dates = set(holidays["date"].dt.date)

    records = []
    total = len(stores) * len(items)
    print(f"Generating sales for {len(stores)} stores × {len(items)} items × {len(dates)} days...")

    for _, store in stores.iterrows():
        for _, item in items.iterrows():
            # Base demand: random level per store-item pair (2 to 80 units/day)
            base = rng.uniform(2, 80)

            # Generate daily sales
            for date in dates:
                demand = base

                # Day-of-week effect: weekends are ~20% higher
                dow = date.dayofweek
                if dow >= 5:  # Saturday, Sunday
                    demand *= 1.20

                # Monthly seasonality: December is 40% higher
                month = date.month
                if month == 12:
                    demand *= 1.40
                elif month in [1, 7]:  # Jan and Jul slightly lower
                    demand *= 0.90

                # Yearly growth trend: +5% per year from start
                years_elapsed = (date - dates[0]).days / 365.25
                demand *= 1 + 0.05 * years_elapsed

                # Promotion: ~15% of days, gives 30-60% uplift
                on_promo = rng.random() < 0.15
                if on_promo:
                    demand *= rng.uniform(1.30, 1.60)

                # Holiday proximity: if within 7 days of a holiday, +25%
                near_holiday = any(
                    abs((date.date() - h).days) <= 7 for h in holiday_dates
                )
                if near_holiday:
                    demand *= 1.25

                # Add noise (standard deviation = 20% of demand)
                demand = max(0, demand + rng.normal(0, demand * 0.20))

                records.append(
                    {
                        "date": date,
                        "store_nbr": store["store_nbr"],
                        "item_nbr": item["item_nbr"],
                        "unit_sales": round(demand, 1),
                        "onpromotion": on_promo,
                    }
                )

    print(f"Generated {len(records):,} rows")
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Generating synthetic supply chain data")
    print("=" * 60)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Generate each component
    stores = generate_stores(SYNTHETIC_N_STORES)
    items = generate_items(SYNTHETIC_N_ITEMS, SYNTHETIC_PERISHABLE_FRACTION)
    holidays = generate_holidays()
    sales = generate_sales(
        stores, items, holidays, SYNTHETIC_START_DATE, SYNTHETIC_END_DATE
    )

    # Generate transactions (daily aggregated count per store)
    transactions = (
        sales.groupby(["date", "store_nbr"])
        .size()
        .reset_index(name="transactions")
    )

    # Save all files
    stores.to_csv(DATA_RAW / "stores.csv", index=False)
    items.to_csv(DATA_RAW / "items.csv", index=False)
    holidays.to_csv(DATA_RAW / "holidays_events.csv", index=False)
    sales.to_csv(DATA_RAW / "train.csv", index=False)
    transactions.to_csv(DATA_RAW / "transactions.csv", index=False)

    # Generate a simple oil.csv (not used heavily but keeps structure consistent)
    oil_dates = pd.date_range(SYNTHETIC_START_DATE, SYNTHETIC_END_DATE, freq="D")
    rng = np.random.default_rng(99)
    oil = pd.DataFrame(
        {
            "date": oil_dates,
            "dcoilwtico": 50 + np.cumsum(rng.normal(0, 0.5, len(oil_dates))),
        }
    )
    oil.to_csv(DATA_RAW / "oil.csv", index=False)

    print(f"\nAll files saved to {DATA_RAW}/")
    print(f"  stores.csv:           {len(stores)} rows")
    print(f"  items.csv:            {len(items)} rows")
    print(f"  holidays_events.csv:  {len(holidays)} rows")
    print(f"  train.csv:            {len(sales):,} rows")
    print(f"  transactions.csv:     {len(transactions):,} rows")
    print(f"  oil.csv:              {len(oil):,} rows")
    print("\nDone! You can now run: python src/01_data_prep.py")


if __name__ == "__main__":
    main()
