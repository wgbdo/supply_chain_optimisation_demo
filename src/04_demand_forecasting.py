"""
Step 4: Demand Forecasting
============================
Trains probabilistic demand forecast models using LightGBM quantile regression.

CONCEPT — Why Probabilistic Forecasts?
  A single-point forecast ("we'll sell 3,200 units") is not enough for inventory
  decisions. You need to know the *range* of likely outcomes:
    - q10 (10th percentile): "there's a 90% chance we'll sell at least this much"
    - q50 (50th percentile / median): best single-point estimate
    - q90 (90th percentile): "there's only a 10% chance we'll sell more than this"

  The gap between q50 and q90 directly determines safety stock:
    safety_stock ≈ q90 - q50
  Wide gap = high uncertainty = more safety stock needed.

CONCEPT — Why LightGBM?
  Gradient-boosted decision trees (GBDT) are the workhorse of tabular ML:
    - Handle mixed feature types (numeric, categorical) natively
    - Robust to outliers and missing values
    - Fast to train
    - Often beat deep learning on structured/tabular data
    - LightGBM's `objective="quantile"` trains a model for any quantile

CONCEPT — Train/Test Split for Time Series:
  We can't do random splits for time series (that would leak future data).
  Instead, we use a temporal split: train on all data before a cutoff date,
  test on data after. This simulates real-world deployment where you only
  have past data to predict the future.

Key outputs:
  - data/processed/models/lgbm_q10.txt, lgbm_q50.txt, lgbm_q90.txt
  - data/processed/forecasts.parquet (predictions on the test set)

Usage:
    python src/04_demand_forecasting.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROCESSED,
    FORECAST_HORIZON,
    HOLDOUT_WEEKS,
    MODELS_DIR,
    QUANTILES,
)

# Features used by the model. These are all the columns from Step 3 that are
# legitimate inputs (i.e. known at prediction time, no leakage).
FEATURE_COLS = [
    # Lag features (past demand — known at prediction time)
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_4",
    "demand_lag_8",
    "demand_lag_52",
    # Rolling statistics (computed from past demand)
    "rolling_4w_mean",
    "rolling_4w_std",
    "rolling_12w_mean",
    "rolling_12w_std",
    "demand_cv_4w",
    # Calendar features (known for any future date)
    "week_of_year",
    "month",
    "quarter",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "is_month_start",
    "is_month_end",
    # Promotion (assumed known ahead — promo calendars are planned in advance)
    "promo_intensity",
    "promo_last_week",
    "promo_change",
    # Holiday and item attributes
    "is_holiday_week",
    "perishable",
]

TARGET_COL = "unit_sales"


def temporal_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Split data temporally: last HOLDOUT_WEEKS weeks → test, rest → train.

    IMPORTANT: this is NOT a random split. In time series, you must always
    train on past data and test on future data, otherwise you're cheating.
    """
    cutoff = df["week_start"].max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)

    train = df[df["week_start"] <= cutoff].copy()
    test = df[df["week_start"] > cutoff].copy()

    print(f"  Train: {len(train):,} rows | {train['week_start'].min().date()} → {train['week_start'].max().date()}")
    print(f"  Test:  {len(test):,} rows  | {test['week_start'].min().date()} → {test['week_start'].max().date()}")

    return train, test


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM model for a specific quantile.

    CONCEPT — Quantile Regression:
      Normal regression minimises MSE (mean squared error), which gives you the
      *mean* prediction. Quantile regression minimises the "pinball loss" for a
      specific quantile, giving you the Nth percentile prediction.

      For q=0.90: the model is penalised 9× more for under-predicting than
      over-predicting, so it learns to predict high (covering 90% of outcomes).

      For q=0.10: the model is penalised 9× more for over-predicting, so it
      learns to predict low.
    """
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,  # the quantile to target
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,   # increased from 31 — more expressive model
        min_child_samples=10,  # lower regularisation — allows finer splits
        subsample=0.8,  # use 80% of data per tree (reduces overfitting)
        colsample_bytree=0.8,  # use 80% of features per tree
        n_jobs=-1,       # parallel training
        random_state=42,
        verbose=-1,  # suppress training logs
    )

    model.fit(X_train, y_train)
    return model


def evaluate_forecasts(test: pd.DataFrame) -> dict:
    """
    Compute forecast accuracy metrics.

    METRICS EXPLAINED:
      - MAE: Mean Absolute Error — average $ amount the forecast is off by
      - MAPE: Mean Absolute Percentage Error — average % the forecast is off
        (e.g. MAPE=15% means on average the forecast is 15% wrong)
      - Coverage: what fraction of actual values fall within the q10-q90 interval?
        Target: ~80% (since it's an 80% prediction interval). If coverage is much
        lower, the model is overconfident.
    """
    actuals = test[TARGET_COL]
    pred_q50 = test["forecast_q50"]
    pred_q10 = test["forecast_q10"]
    pred_q90 = test["forecast_q90"]

    mae = mean_absolute_error(actuals, pred_q50)
    mape = mean_absolute_percentage_error(actuals, pred_q50) * 100

    # sMAPE: symmetric MAPE — bounded [0, 200%], less dominated by low-volume items
    denom = (np.abs(actuals) + np.abs(pred_q50)).replace(0, np.nan)
    smape = (2 * np.abs(actuals - pred_q50) / denom).mean() * 100

    coverage = ((actuals >= pred_q10) & (actuals <= pred_q90)).mean() * 100

    # Bias: positive = over-forecasting on average
    bias = (pred_q50 - actuals).mean()

    return {
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "Coverage_80pct": coverage,
        "Bias": bias,
    }


def plot_feature_importance(model: lgb.LGBMRegressor, feature_names: list):
    """Save a feature importance plot for the q50 model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 8))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance (q50 model)")
    ax.set_xlabel("Importance (split count)")
    fig.tight_layout()

    from config.settings import PLOTS_DIR
    fig.savefig(PLOTS_DIR / "07_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: 07_feature_importance.png")


def main():
    print("=" * 60)
    print("Step 4: Demand Forecasting (LightGBM Quantile Regression)")
    print("=" * 60)

    # Load features from Step 3
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet")

    # Split temporally
    print("\nSplitting data...")
    train, test = temporal_train_test_split(df)

    # Reserve last 10% of training weeks as calibration set for split-conformal
    # prediction intervals. Models are trained only on the remaining 90%.
    train_weeks_sorted = sorted(train["week_start"].unique())
    n_calib_weeks = max(4, int(len(train_weeks_sorted) * 0.10))
    calib_start_week = train_weeks_sorted[-n_calib_weeks]
    calib = train[train["week_start"] >= calib_start_week].copy()
    train_fit = train[train["week_start"] < calib_start_week].copy()
    print(f"  Calib: {len(calib):,} rows ({n_calib_weeks} weeks held out for conformal intervals)")

    X_train = train_fit[FEATURE_COLS]
    y_train = train_fit[TARGET_COL]
    X_test = test[FEATURE_COLS]

    # Train one model per quantile
    models = {}
    for q in QUANTILES:
        q_label = f"q{int(q*100)}"
        print(f"\nTraining {q_label} model (quantile={q})...")
        model = train_quantile_model(X_train, y_train, q)
        models[q_label] = model

        # Save model
        model_path = MODELS_DIR / f"lgbm_{q_label}.txt"
        model.booster_.save_model(str(model_path))
        print(f"  Saved model: {model_path}")

    # Generate predictions on test set
    print("\nGenerating forecasts on test set...")
    for q_label, model in models.items():
        preds = model.predict(X_test)
        # Clip to non-negative (demand can't be negative)
        test[f"forecast_{q_label}"] = np.clip(preds, 0, None)

    # ── Bias correction ──────────────────────────────────────────────────
    # The q50 model may systematically over- or under-forecast on the training
    # set. We measure this bias and subtract it from all quantile predictions
    # to re-centre the forecast. This is a simple but effective post-processing
    # step that does not require retraining.
    print("\nApplying bias correction...")
    train_preds_q50 = models["q50"].predict(X_train)
    bias_correction = float((train_preds_q50 - y_train.values).mean())
    print(f"  Training set bias: {bias_correction:+.2f} units (positive = over-forecasting)")
    for col in ["forecast_q10", "forecast_q50", "forecast_q90"]:
        test[col] = np.clip(test[col] - bias_correction, 0, None)
    print(f"  Bias correction applied to forecast_q10, q50, q90")
    # ── Conformal prediction interval adjustment ────────────────────────────────
    # Split-conformal regression: use the calibration set to find the smallest
    # symmetric widening of [q10, q90] that achieves 80% empirical coverage.
    #
    # Nonconformity score for each calibration point:
    #   s = max(q10(x) - y,  y - q90(x))
    # s > 0 means the actual fell outside the interval.
    # The (80th percentile) of {s_i} is the adjustment needed.
    print("\nApplying conformal prediction interval adjustment...")
    X_calib = calib[FEATURE_COLS]
    y_calib = calib[TARGET_COL].values
    calib_q10 = np.clip(models["q10"].predict(X_calib) - bias_correction, 0, None)
    calib_q90 = np.clip(models["q90"].predict(X_calib) - bias_correction, 0, None)
    scores = np.maximum(calib_q10 - y_calib, y_calib - calib_q90)
    n_calib = len(scores)
    idx = min(int(np.ceil((n_calib + 1) * 0.80)) - 1, n_calib - 1)
    adjustment = max(0.0, float(np.sort(scores)[idx]))
    print(f"  Calibration set size: {n_calib:,} rows")
    print(f"  Conformal adjustment: {adjustment:+.2f} units (widens q10 down / q90 up)")
    test["forecast_q10"] = np.clip(test["forecast_q10"] - adjustment, 0, None)
    test["forecast_q90"] = test["forecast_q90"] + adjustment
    # Evaluate
    metrics = evaluate_forecasts(test)
    print("\nForecast Accuracy Metrics:")
    for name, value in metrics.items():
        print(f"  {name:20s}: {value:.2f}")

    # Feature importance
    plot_feature_importance(models["q50"], FEATURE_COLS)

    # Save forecasts
    output_path = DATA_PROCESSED / "forecasts.parquet"
    test.to_parquet(output_path, index=False)
    print(f"\nSaved forecasts: {output_path}")
    print(f"  Shape: {test.shape}")

    # Quick sanity check: print a few example predictions
    sample = test.sample(5, random_state=42)[
        ["store_nbr", "item_nbr", "week_start", TARGET_COL, "forecast_q10", "forecast_q50", "forecast_q90"]
    ]
    print("\nSample predictions:")
    print(sample.to_string(index=False))

    print("\nDone! Next step: python src/05_business_rules.py")


if __name__ == "__main__":
    main()
