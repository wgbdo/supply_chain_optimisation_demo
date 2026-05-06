# Data Directory

## Generating the Data

This demo uses synthetic data. Run from the project root:

```bash
python src/00_generate_synthetic_data.py
```

This creates the following CSVs in `data/raw/` using a fixed random seed (`numpy.random.default_rng(42)`),
mimicking the Corporación Favorita grocery dataset structure:

```
data/raw/
├── train.csv           # daily unit sales: date, store_nbr, item_nbr, unit_sales, onpromotion
├── stores.csv          # store metadata (city, state, type, cluster)
├── items.csv           # item metadata (family, class, perishable)
├── transactions.csv    # daily transaction counts per store
├── oil.csv             # daily oil price (economic indicator)
└── holidays_events.csv # holidays with type and locale
```

> **Optional — Real Kaggle dataset:** If you want to run on the original Corporación Favorita
> data (~125M rows), download it from
> https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data,
> extract the CSVs into `data/raw/`, and skip step 0.

## Processed Data

The `data/processed/` directory is auto-created by the pipeline scripts.
Do not put files there manually.
