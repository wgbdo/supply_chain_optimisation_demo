# Next Steps — Supply Chain Optimisation Demo

## Latest Pipeline Results

Dataset: Kaggle Favorita (125M rows), 49 items × 5 stores, 2013–2017 evaluation window.

| Metric | Result | Target | Status |
|---|---|---|---|
| sMAPE | **34.18%** | <35% | ✅ |
| Raw MAPE | 158.76% | — | ⚠️ Skewed by low-volume items (see Priority 1) |
| Forecast Bias | +76.3 units (+13.3%) | ~0 | ⚠️ Persistent over-forecast |
| 80% Interval Coverage | **70.97%** | ~80% | ⚠️ Below target |
| Fill Rate | **98.39%** | >98% | ✅ |
| Stockout Rate | 6.29% (115 incidents) | <5% | ⚠️ Slightly above target |
| Waste Rate | **7.64%** (perishables 12.1%) | <5% | ❌ Above target |
| Total Cost | $4,372,542 | — | — |
| Cost per Unit | $4.18 | — | — |
| **Saving vs 25% flat-buffer heuristic** | **$2,339,901 (34.9%)** | — | ✅ |

Business rules active: DAVE_001 (Easter), DAVE_002 (Christmas), DAVE_003 (perishable buffer), DAVE_004 (heat), DAVE_005 (new item), DAVE_006 (payday ×230), DAVE_007 (low-demand cap ×12).

---

## Priority 1 — Fix the MAPE Measurement Problem

**Root cause**: Raw MAPE is 158% because a handful of items with very low actual sales (e.g. 7 units/week actual vs 138 forecast) create unbounded percentage errors. The model is not broken — the metric is misleading.

### 1.1 — Filter intermittent items from top-N selection
- **What**: In `src/01_data_prep.py`, after selecting the top-N items by total sales, drop any item whose **median weekly sales < 20 units**. These items are genuinely intermittent and are better handled by a separate Croston/ADIDA model rather than LightGBM.
- **Why it matters**: Removing ~5–10 low-volume outliers would bring raw MAPE well below 50% without changing any model code. DAVE_007 suppresses the worst over-ordering from these items but doesn't fix the root cause.
- **File**: `config/settings.py` (add `MIN_MEDIAN_WEEKLY_SALES = 20`), `src/01_data_prep.py`

### 1.2 — Report sMAPE as the primary forecast metric
- **What**: sMAPE (`2*|A−F|/(|A|+|F|)`) is already computed at 34.18% — promote it as the headline metric in the dashboard and evaluation output. Demote raw MAPE to a footnote.
- **File**: `src/07_evaluation.py`, `src/08_dashboard.py`

### 1.3 — Per-family LightGBM models
- **What**: The dataset has 8 product families (BEVERAGES, DAIRY, PRODUCE, etc.) with very different seasonal patterns. A single shared model averages over these differences. Train one quantile LightGBM per family, then concatenate predictions.
- **Expected impact**: Lower bias, better interval coverage for families with distinct patterns (e.g. PRODUCE peaks differently to BEVERAGES).
- **File**: `src/04_demand_forecasting.py` — wrap the existing `fit`/`predict` loop in a `for family in df['family'].unique()` loop.

---

## Priority 2 — Reduce Waste Rate (7.6% → <5%)

Perishables at **12.1% waste** are the primary driver. Three independent levers:

### 2.1 — Tighten the base demand for perishable orders
- **What**: The MIP currently uses `forecast_q50` as the base demand for all items including perishables. Switch perishables to `forecast_q40` (interpolated between q10 and q50). Asymmetric ordering — order slightly less than the median forecast — reduces expected waste because the cost of waste ($8.75/unit) exceeds the cost of a minor stockout ($10/unit) only at the margin.
- **File**: `src/06_inventory_optimisation.py` — add `q40 = q10 + 0.4*(q50-q10)` and use it for perishable demand constraint.

### 2.2 — Reduce `WAREHOUSE_CAPACITY_PER_STORE` to a realistic value
- **What**: The current value of 60,000 units/store is still loose. A realistic two-week holding constraint at current volumes is ~52,000 units (avg 529 units/item × 49 items × 2 weeks). Setting the cap closer to this forces the MIP to trade off across items rather than order maximum for all.
- **File**: `config/settings.py` → `WAREHOUSE_CAPACITY_PER_STORE = 52_000`

### 2.3 — Per-family safety stock weeks
- **What**: Replace the single `SAFETY_STOCK_WEEKS` scalar with a dict keyed by perishability class: `{'perishable': 0.5, 'non_perishable': 1.5}`. Perishable items get a narrower safety buffer — a minor stockout is less expensive than accumulating waste across 49 SKUs per store.
- **File**: `config/settings.py`, `src/06_inventory_optimisation.py`

---

## Priority 3 — Close the Interval Coverage Gap (71% → ~80%)

The 80% prediction interval is covering only 71% of actual outcomes. The conformal adjustment (+23.81 units) partially fixes this, but the intervals are still too narrow on high-volatility items.

### 3.1 — Recalibrate the conformal adjustment per product family
- **What**: The current conformal correction is a single scalar applied uniformly. Compute a separate adjustment per family — high-variance families (PRODUCE) need a larger adjustment than stable categories (BEVERAGES). This is still a simple post-processing step with no model changes.
- **File**: `src/04_demand_forecasting.py` — group calibration residuals by family before computing the adjustment.

### 3.2 — Widen the base quantile gap
- **What**: Train `q05`/`q95` instead of `q10`/`q90`. This widens the interval by construction and is a one-line change. The 90% interval should cover ~80% of outcomes more reliably than the current 80% interval.
- **File**: `src/04_demand_forecasting.py` — `QUANTILES = [0.05, 0.5, 0.95]`

---

## Priority 4 — Reduce Stockout Rate (6.3% → <5%)

115 stockout incidents represent 16,912 lost units. The stockout is concentrated in a small number of item/store/week combinations.

### 4.1 — Identify stockout-prone items and add a targeted DAVE rule
- **What**: Run `evaluation_summary` grouped by item to find the 5 items responsible for the majority of stockout events. Add a `DAVE_008` rule that applies a `multiply_safety_stock: 1.25` factor specifically to those item IDs during their historical peak periods.
- **File**: `business_rules/rules.json`, `src/business_rules_utils.py`

### 4.2 — Use q90 as the order floor for high-stockout-risk items
- **What**: For items where the stockout rate exceeds 10% in historical evaluation, set the MIP minimum order to `forecast_q90` rather than the current safety-stock formula. This over-orders slightly but eliminates the tail stockout risk for the worst offenders.
- **File**: `src/06_inventory_optimisation.py`

---

## Priority 5 — Pipeline Productionisation

### 5.1 — Scheduled weekly run
- Wrap steps 01–07 in a PowerShell scheduled task or Prefect flow. Each step is already idempotent (reads parquet, writes parquet). Suggested schedule: Monday 06:00 before the weekly ordering window opens.

### 5.2 — Dashboard access control
- Add `streamlit-authenticator` or Streamlit Community Cloud OAuth. The current dashboard is publicly accessible with no login gate.

### 5.3 — Override feedback loop
- When a user overrides a recommendation in the dashboard, log the override (item, store, week, recommended_qty, override_qty) to a CSV. Feed this back into rule calibration — systematic overrides indicate a missing or miscalibrated rule.

---

## Longer-Term Strategic Improvements

| Enhancement | Why | Approach |
|---|---|---|
| **Decision-aware forecasting** | Current model minimises sMAPE, not business cost. Asymmetric waste/stockout costs should shape the forecast directly. | SPO+ (Smart Predict then Optimise) or cost-sensitive quantile loss |
| **Dynamic safety stock** | Static safety stock ignores supplier lead time variability — a delayed shipment turns a safe order into a stockout. | Model lead time as a distribution; use safety stock = z × σ_demand × √(lead_time) |
| **Multi-echelon optimisation** | Stores are optimised independently. Rebalancing slow-moving stock from overstocked stores reduces waste without additional procurement. | Add inter-store transfer variables to the MIP |
| **Intermittent demand models** | Low-volume SKUs (e.g. the 12 items DAVE_007 fires on) need Croston's method or ADIDA, not LightGBM. | `statsforecast.CrostonOptimized` or a manual Croston implementation |
| **Reinforcement learning policy** | Replace the forecast+MIP two-step with a direct state→order-quantity policy that learns the cost structure from experience. | Use the synthetic data generator as an OpenAI Gym env; train with PPO |
