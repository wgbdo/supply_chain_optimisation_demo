# Next Steps — Supply Chain Optimisation Demo

## Pipeline Run Results (Baseline)

| Metric | Result | Target | Status |
|---|---|---|---|
| Forecast MAPE | 89.31% | <30% | ❌ High |
| Forecast Bias | +38.6 units (+12.5%) | ~0 | ⚠️ Consistently over-forecasting |
| 80% Interval Coverage | N/A (bug) | ~80% | ❌ Needs fix |
| Fill Rate | 99.57% | >98% | ✅ |
| Stockout Rate | 7.1% (142 incidents) | <5% | ⚠️ Slightly above target |
| Waste Rate | 8.09% (46,330 units) | <5% | ❌ Above target |
| Total Cost | $2,478,765 | — | Baseline established |
| Cost per Unit | $4.01 | — | Baseline established |

---

## Priority 1 — Fix Known Bugs

### 1.1 — 80% Prediction Interval Coverage returns NaN
- **File**: `src/07_evaluation.py`
- **Cause**: Column name mismatch — the evaluation script looks for `forecast_q10` but the column produced by step 05 (after business rules) may be renamed or missing.
- **Fix**: Confirm column names in `data/processed/adjusted_forecasts.parquet` and align references in `compute_forecast_metrics()`.

### 1.2 — Demand forecasting MAPE (89%)
- **Cause**: High stochastic noise in the synthetic data combined with the small training set (5 stores × 50 items × ~2 years). The LightGBM models are likely under-fitting due to too few observations per series.
- **Fix** (short-term): Tune LightGBM hyperparameters — increase `num_leaves`, `n_estimators`; add `min_child_samples` tuning via cross-validation in `src/04_demand_forecasting.py`.
- **Fix** (medium-term): See Priority 2 below.

---

## Priority 2 — Improve Forecast Quality

### 2.1 — Probabilistic forecasting model upgrade
The current LightGBM quantile regression approach is a solid baseline, but produces noisy interval estimates. Consider:

| Option | Benefit | Effort |
|---|---|---|
| **Tune existing LightGBM** (num_leaves, learning_rate, early stopping) | Quick win; low risk | Low |
| **LightGBM + cross-validation feature selection** | Removes noisy lag features that harm generalisation | Low–Medium |
| **Add external regressors** (temperature, day-of-week interactions) | Addresses systematic patterns not captured by lags | Medium |
| **Replace with `statsforecast` MSTL or AutoETS** (already in `requirements.txt`) | Purpose-built for time series; better seasonal decomposition | Medium |
| **Conformal prediction wrapper** | Converts any point forecast to calibrated intervals; fixes coverage NaN issue as a side effect | Medium |

### 2.2 — Reduce positive bias (+12.5%)
The model consistently over-forecasts. This directly drives the 8.09% waste rate, since the MIP optimiser orders to cover the (inflated) q90 forecast.
- Add a bias-correction post-processing step after `src/04_demand_forecasting.py`: subtract the mean residual from the training set from all forecast quantiles.
- Alternatively, tune `DAVE_003` (perishable bruising multiplier ×1.15 in `business_rules/rules.json`) — it fired 680 times and amplifies the existing positive bias.

---

## Priority 3 — Reduce Waste Rate (8.09% → <5%)

The waste rate exceeds target. Root causes:
1. **Positive forecast bias** (+12.5%) causes over-ordering — see Priority 2.2.
2. **DAVE_003 rule** applies a blanket ×1.15 uplift to all perishable items regardless of current stock levels. It fired 680/845 adjusted rows.
3. **Warehouse capacity set to 100,000** (raised from original 5,000 to fix MIP infeasibility) removes the capacity constraint entirely, allowing unconstrained over-ordering.

**Recommended fixes:**
- Recalibrate `DAVE_003` to only fire when current inventory is below a threshold (add a `min_inventory_condition` to the rule in `rules.json`).
- Revisit `WAREHOUSE_CAPACITY_PER_STORE` in `config/settings.py` — determine a realistic value for the target scenario and re-test MIP feasibility.
- Add a perishability penalty term to the MIP objective in `src/06_inventory_optimisation.py` that explicitly penalises ordering units that will expire before the next order cycle.

---

## Priority 4 — Pipeline Productionisation

### 4.1 — Replace synthetic data with real data
The pipeline is designed to accept real Corporación Favorita-style CSVs (or equivalent operational data from SAP). When connecting to real data:
- Update `config/settings.py`: `N_ITEMS`, `N_STORES`, `SYNTHETIC_START_DATE`, `SYNTHETIC_END_DATE` will be superseded by actual data range.
- Validate schema compatibility in `src/01_data_prep.py` — confirm column names match the real source system export.
- Expect MAPE to improve significantly with real data (less stochastic noise, more observations per series).

### 4.2 — Scheduled pipeline execution
Currently all steps are run manually. For production:
- Wrap steps 01–07 in an orchestration framework (e.g. **Prefect**, **Apache Airflow**, or a simple **cron + shell script**).
- Each step is already idempotent (reads parquet, writes parquet); no changes to the scripts are needed.
- Suggested schedule: weekly run on Monday morning, before the ordering window opens.

### 4.3 — Dashboard hardening
The Streamlit dashboard (`src/08_dashboard.py`) is a prototype. Before sharing with operators:
- Add authentication (Streamlit Community Cloud supports GitHub OAuth; self-hosted options include `streamlit-authenticator`).
- Replace local parquet file reads with a database or cloud storage backend (e.g. Azure Blob, S3, or a lightweight SQLite for single-server deployments).
- Add a feedback mechanism: operators should be able to flag recommendations they override, which feeds back into rule calibration.

### 4.4 — Unit tests
No tests currently exist. Minimum viable test coverage:
- `config/settings.py`: assert `WAREHOUSE_CAPACITY_PER_STORE > 0`, cost parameters are positive.
- `src/05_business_rules.py`: for each rule, assert the output multiplier is applied correctly on a known fixture.
- `src/06_inventory_optimisation.py`: assert the MIP returns `Optimal` status and `order_qty >= MOQ` for a small synthetic problem.

---

## Priority 5 — Strategic Enhancements (Longer Term)

| Enhancement | Why | Recommended approach |
|---|---|---|
| **Decision-aware forecasting** | Current model minimises MAPE, not business cost. For perishables, asymmetric costs (waste vs. stockout) should shape the forecast. | SPO+ (Smart Predict-then-Optimise) or cost-sensitive quantile loss |
| **Dynamic safety stock** | Current safety stock is static (rules-based). It should vary by item, season, and supplier lead time. | Extend `src/05_business_rules.py` to read lead times from a supplier table and compute safety stock dynamically |
| **Supplier lead time modelling** | The MIP assumes fixed lead times. Stochastic lead times increase effective stockout risk. | Add a lead time distribution per supplier; convert the MIP to a two-stage stochastic programme |
| **Multi-echelon optimisation** | Currently optimises each store independently. Cross-store allocation (e.g. redistribute overstock from Store A to Store B) can reduce both waste and stockouts. | Extend `src/06_inventory_optimisation.py` with inter-store transfer decision variables |
| **Reinforcement learning policy** | Replaces the forecast+MIP pipeline with a direct state→action policy. Suitable once a good simulation environment exists. | Use the synthetic data generator as a gym environment; train with PPO or SAC |

---

## Immediate Action Checklist

- [ ] Fix interval coverage NaN bug (`src/07_evaluation.py` — column name mismatch)
- [ ] Tune LightGBM hyperparameters to bring MAPE below 30%
- [ ] Add bias correction post-processing to address +12.5% over-forecast
- [ ] Recalibrate `DAVE_003` rule trigger condition to reduce waste rate
- [ ] Determine realistic `WAREHOUSE_CAPACITY_PER_STORE` value for the target scenario
- [ ] Add minimum unit tests for business rules and MIP solver
- [ ] Connect to real data source (SAP export or equivalent) and validate schema
