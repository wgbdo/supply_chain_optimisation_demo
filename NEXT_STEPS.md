# Next Steps — Supply Chain Optimisation Demo

## Pipeline Run Results

### Run 2 — Real Kaggle Favorita Data (125M rows, 49 items × 5 stores, 2013–2017)

| Metric | Result | Target | Status |
|---|---|---|---|
| Forecast MAPE | 154.34% | <30% | ❌ Skewed by low-volume items |
| Forecast Bias | +78.1 units (+13.6%) | ~0 | ⚠️ Over-forecasting |
| 80% Interval Coverage | 59.87% | ~80% | ⚠️ Below target |
| Fill Rate | 98.38% | >98% | ✅ |
| Stockout Rate | 7.49% (137 incidents) | <5% | ⚠️ Slightly above target |
| Waste Rate | 7.99% — perishables 12.9% | <5% | ❌ Above target |
| Total Cost | $4,310,512 | — | — |
| Cost per Unit | $4.12 | — | — |

### Run 1 — Synthetic Data (Baseline, now superseded)

| Metric | Result | Notes |
|---|---|---|
| MAPE | 89.31% | Synthetic noise; lower variance than real |
| Waste Rate | 6.54% | After bias correction + seasonal DAVE_003 |
| Total Cost | $2.34M | Smaller volumes than real data |

---

## Completed ✅

| Item | Change | Impact |
|---|---|---|
| Interval coverage NaN bug | Added `forecast_q10` to step 06 output | Coverage now 59.87% |
| LightGBM hyperparameter tuning | `num_leaves=63`, `n_estimators=1000`, `lr=0.03` | Improved model fit |
| Bias correction | Subtract mean training residual post-forecast | −10.36 unit bias corrected |
| DAVE_003 recalibration | Seasonal only (Oct–Apr) instead of always | Reduced off-season over-ordering |
| Perishability penalty in MIP | `PERISHABLE_WASTE_COST_MULTIPLIER=2.5` in objective | Cost-aware ordering for perishables |
| Unit tests | 21 tests across business rules + MIP solver | All passing |
| Real Kaggle data integration | Auto-detect `KAGGLE_DATA_PATH`; `RAW_DATA_SOURCE` fallback | Pipeline runs on 125M-row real dataset |

---

## Priority 1 — Reduce MAPE (154% → target <30%)

The high MAPE on real data is **not primarily a model quality problem** — it's a measurement problem. A handful of very-low-volume items (e.g. item with actual sales of 9 units, forecast 85) dominate the MAPE calculation.

### 1.1 — Filter out intermittent/low-volume items from top-N selection
- **File**: `config/settings.py`, `src/01_data_prep.py`
- **Change**: In `load_raw_data()`, after filtering to top N items, additionally drop items where the median weekly sales is below a threshold (e.g. `< 20 units`). These items are better handled by Croston's method, not LightGBM.
- **Expected impact**: MAPE drops significantly (low-volume items have unbounded % error).

### 1.2 — Switch MAPE metric to sMAPE or volume-weighted MAPE
- **File**: `src/04_demand_forecasting.py`, `src/07_evaluation.py`
- **Change**: Replace raw MAPE with symmetric MAPE (`2*|A-F|/(|A|+|F|)`) or weight errors by actual demand volume. This gives a fairer picture of model quality across the full range of item sizes.
- **Effort**: ~10 lines of code change.

### 1.3 — Per-family models
- The dataset has 8 product families with very different demand patterns (BEVERAGES vs DAIRY vs PRODUCE). A single shared LightGBM model cannot learn family-specific seasonality.
- **Change**: Loop over families in `src/04_demand_forecasting.py`, fit one LightGBM per family (or use the `family` column as a high-cardinality categorical feature with target encoding).

---

## Priority 2 — Reduce Waste Rate (8% → <5%)

Perishables at **12.9% waste** are the key driver. Root causes:
1. **Over-forecasting bias** (+13.6%) pumps the q90 upper bound used for safety stock.
2. **Safety stock is symmetric** — the MIP adds safety stock equally to perishable and non-perishable items. Perishables should use a tighter safety stock (lower quantile) because the cost of waste exceeds the cost of a minor stockout.
3. **`WAREHOUSE_CAPACITY_PER_STORE = 500,000`** is effectively unconstrained at current volumes. Reducing it to a realistic value reintroduces capacity as a natural waste brake.

### 2.1 — Use q40 instead of q50 as the base order for perishables
- **File**: `src/06_inventory_optimisation.py`
- **Change**: When building the demand constraint for perishable items, use `forecast_q40` (or interpolate between q10 and q50) rather than q50. This asymmetric ordering policy reduces expected waste at the cost of a slightly lower fill rate.
- **Note**: Requires adding a `q40` model in step 04 or interpolating.

### 2.2 — Calibrate `WAREHOUSE_CAPACITY_PER_STORE`
- Current value of 500,000 is a placeholder. Compute a realistic value: `avg_weekly_demand_per_store × max_weeks_stock_allowed`.
- With avg 529 units/item × 49 items × 2 weeks = ~52,000 units. Setting to `60_000` would meaningfully constrain over-ordering.
- **File**: `config/settings.py`

### 2.3 — Add `SAFETY_STOCK_WEEKS` as a per-family parameter
- **File**: `config/settings.py`, `src/06_inventory_optimisation.py`
- **Change**: Instead of a single `SAFETY_STOCK_WEEKS` scalar, use a dict keyed by family (e.g. `PRODUCE: 0.5`, `BEVERAGES: 1.5`). Perishable families get a smaller safety stock buffer.

---

## Priority 3 — Improve Interval Coverage (59.9% → ~80%)

The 80% prediction interval is only covering 59.9% of actual outcomes — the intervals are too narrow.

### 3.1 — Conformalized quantile regression
- Wrap the LightGBM quantile forecasts in a **conformal prediction** post-processing step using a calibration split (the last 10% of training data).
- The conformal adjustment inflates/deflates q10 and q90 so that empirical coverage on the calibration set hits exactly 80%.
- Library: `MAPIE` (already available via pip) or a 10-line manual implementation.

### 3.2 — Widen quantile gap
- Quick interim fix: train q05/q95 instead of q10/q90 — the 90% interval will cover more outcomes at the cost of slightly higher order quantities.
- **File**: `src/04_demand_forecasting.py` — change `QUANTILES = [0.1, 0.5, 0.9]` to `[0.05, 0.5, 0.95]`.

---

## Priority 4 — Pipeline Productionisation

### 4.1 — Scheduled pipeline execution
- Wrap steps 01–07 in **Prefect** or a simple **cron + PowerShell script**.
- Each step is already idempotent (reads parquet, writes parquet) — no script changes needed.
- Suggested schedule: weekly run on Monday morning before the ordering window.

### 4.2 — Dashboard hardening
- Add authentication (`streamlit-authenticator` or Streamlit Community Cloud OAuth).
- Replace local parquet reads with Azure Blob / S3 or SQLite for multi-user access.
- Add an operator override feedback loop: flag recommendations overridden → feeds back into rule calibration.

### 4.3 — Suppress CBC solver verbose output in production
- **File**: `src/06_inventory_optimisation.py`
- **Change**: Pass `msg=False` to `pulp.PULP_CBC_CMD(msg=False)` to silence the per-solve CBC log. Keeps pipeline output clean.

---

## Priority 5 — Strategic Enhancements (Longer Term)

| Enhancement | Why | Approach |
|---|---|---|
| **Decision-aware forecasting** | Current model minimises MAPE, not business cost. Asymmetric waste/stockout costs should shape the forecast. | SPO+ or cost-sensitive quantile loss |
| **Dynamic safety stock** | Static safety stock ignores supplier lead time variability. | Per-supplier lead time distribution → safety stock formula |
| **Multi-echelon optimisation** | Stores optimised independently; no cross-store rebalancing. | Add inter-store transfer variables to the MIP |
| **Intermittent demand models** | Low-volume SKUs need Croston's / ADIDA, not LightGBM | `statsforecast.AutoCES` or manual Croston |
| **Reinforcement learning policy** | Replace forecast+MIP with a direct state→action policy once a simulation environment exists. | Use synthetic generator as a Gym env; train with PPO |

---

## Immediate Action Checklist

- [x] Fix interval coverage NaN bug
- [x] Tune LightGBM hyperparameters
- [x] Add bias correction post-processing
- [x] Recalibrate DAVE_003 rule (seasonal)
- [x] Add perishability penalty to MIP
- [x] Add unit tests (21 passing)
- [x] Connect to real Kaggle data source
- [ ] Filter out intermittent/low-volume items before top-N selection (Priority 1.1)
- [ ] Switch to sMAPE or volume-weighted MAPE (Priority 1.2)
- [ ] Use asymmetric quantile for perishable orders (Priority 2.1)
- [ ] Calibrate `WAREHOUSE_CAPACITY_PER_STORE` to realistic value ~60,000 (Priority 2.2)
- [ ] Conformalize prediction intervals to hit 80% coverage (Priority 3.1)
- [ ] Silence CBC solver output with `msg=False` (Priority 4.3)
