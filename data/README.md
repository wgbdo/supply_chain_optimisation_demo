# Data Directory

## Option A: Kaggle Dataset (Recommended)

Download from: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data

1. Create a free Kaggle account
2. Accept the competition rules
3. Download all files (~480 MB compressed)
4. Extract the CSVs into `data/raw/`

You should end up with:
```
data/raw/
├── train.csv
├── test.csv
├── stores.csv
├── items.csv
├── transactions.csv
├── oil.csv
├── holidays_events.csv
└── sample_submission.csv
```

## Option B: Synthetic Data

If you don't have a Kaggle account, run from the project root:

```bash
python src/00_generate_synthetic_data.py
```

This creates synthetic CSVs in `data/raw/` that mimic the Favorita structure.

## Processed Data

The `data/processed/` directory is auto-created by the pipeline scripts.
Do not put files there manually.
