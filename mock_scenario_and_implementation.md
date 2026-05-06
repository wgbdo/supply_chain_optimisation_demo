# Mock Scenario: FreshBox Australia — Supply Chain Optimisation

## The Company

**FreshBox Australia** is a meal-kit and grocery-box delivery company based in Melbourne. They pack and deliver ~15,000 food boxes per week across Victoria, NSW, and Queensland. They carry ~200 SKUs (fresh produce, proteins, pantry items, packaging). Orders are placed weekly with suppliers; boxes are assembled Mon–Wed and delivered Thu–Sun.

Their long-tenured warehouse manager, Dave, is retiring. Dave keeps most replenishment logic in his head: "Order extra lamb before Easter", "Avocados from QLD take an extra day in January due to floods", "If mangoes are on the menu, order 20% extra because they bruise in transit". The company needs to capture Dave's knowledge and build a system that new staff can rely on.

---

## Mock Data

### Historical Weekly Demand (SKU: Chicken Breast 500g)

```csv
week_start,sku,boxes_ordered,units_needed,units_wasted,promo_active,public_holiday_next_week,avg_temp_c
2025-01-06,CHKBRST500,3200,3200,45,0,0,28.3
2025-01-13,CHKBRST500,3350,3350,60,0,1,30.1
2025-01-20,CHKBRST500,3600,3600,30,0,0,29.5
2025-01-27,CHKBRST500,3100,3100,55,0,0,31.2
2025-02-03,CHKBRST500,3250,3250,40,0,0,27.8
2025-02-10,CHKBRST500,3400,3400,35,1,0,26.5
2025-02-17,CHKBRST500,3800,3800,20,1,0,25.9
2025-02-24,CHKBRST500,3150,3150,50,0,0,24.3
2025-03-03,CHKBRST500,3300,3300,38,0,0,23.1
2025-03-10,CHKBRST500,3500,3500,42,0,0,22.7
2025-03-17,CHKBRST500,3450,3450,28,0,0,21.5
2025-03-24,CHKBRST500,3250,3250,65,0,0,20.8
2025-03-31,CHKBRST500,3100,3100,70,0,1,19.2
2025-04-07,CHKBRST500,2900,2900,80,0,0,18.5
2025-04-14,CHKBRST500,4200,4200,15,0,1,17.8
...
```

> Note the Easter spike in week of 2025-04-14 — this is one of "Dave's rules".

### Supplier Information

```csv
supplier,sku,lead_time_days,lead_time_std_days,moq_units,unit_cost_aud,shelf_life_days
Ingham's,CHKBRST500,3,0.5,500,4.50,7
Baiada,CHKBRST500,2,1.0,200,4.75,6
CostaCorp,AVOCADO_HASS,4,1.5,300,1.20,5
Perfection Fresh,TOMATO_ROMA_1KG,2,0.3,200,2.10,8
```

### Dave's Rules (captured via structured interview)

```json
[
  {
    "rule_id": "DAVE_001",
    "sku_pattern": "*",
    "condition": "public_holiday_next_week == True AND holiday_name == 'Easter'",
    "action": "multiply_forecast_by(1.35)",
    "rationale": "People order bigger boxes before Easter long weekend"
  },
  {
    "rule_id": "DAVE_002",
    "sku_pattern": "AVOCADO_*",
    "condition": "month IN [12, 1, 2] AND supplier_state == 'QLD'",
    "action": "add_lead_time_days(1)",
    "rationale": "QLD floods and storms delay avocado shipments in summer"
  },
  {
    "rule_id": "DAVE_003",
    "sku_pattern": "MANGO_*",
    "condition": "always",
    "action": "multiply_order_qty_by(1.20)",
    "rationale": "Mangoes bruise easily — roughly 20% arrive unusable"
  },
  {
    "rule_id": "DAVE_004",
    "sku_pattern": "CHKBRST*",
    "condition": "avg_temp_forecast > 30",
    "action": "multiply_forecast_by(0.92)",
    "rationale": "People order fewer heavy meals in extreme heat"
  },
  {
    "rule_id": "DAVE_005",
    "sku_pattern": "*",
    "condition": "new_menu_item == True",
    "action": "set_safety_stock_multiplier(1.5)",
    "rationale": "New recipes have unpredictable uptake — buffer extra"
  }
]
```

---

## Implementation Walk-Through

### Step 1: Demand Forecasting with Nixtla StatsForecast

```python
# requirements: statsforecast, pandas, numpy

import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    SeasonalNaive,
    MSTL,
)

# ── Load and prepare data ──────────────────────────────────────
demand = pd.read_csv("data/weekly_demand.csv", parse_dates=["week_start"])

# StatsForecast expects columns: unique_id, ds, y
df = demand.rename(columns={
    "sku": "unique_id",
    "week_start": "ds",
    "units_needed": "y",
})

# ── Fit multiple models ────────────────────────────────────────
models = [
    AutoARIMA(season_length=52),
    AutoETS(season_length=52),
    MSTL(season_lengths=[4, 52]),   # monthly + yearly seasonality
    SeasonalNaive(season_length=52),
]

sf = StatsForecast(
    models=models,
    freq="W-MON",
    n_jobs=-1,
)

sf.fit(df)

# ── Produce probabilistic forecasts (quantiles) ───────────────
# We want the 10th, 50th, and 90th percentile forecasts
forecasts = sf.predict(h=8, level=[80])  # 80% prediction interval → q10, q90

print(forecasts.head())
```

**Example output:**

```
       unique_id         ds  AutoARIMA  AutoARIMA-lo-80  AutoARIMA-hi-80  AutoETS  ...
0     CHKBRST500 2025-04-21     3180.0           2840.0           3520.0   3210.0
1     CHKBRST500 2025-04-28     3220.0           2790.0           3650.0   3250.0
2     CHKBRST500 2025-05-05     3310.0           2750.0           3870.0   3280.0
...
```

### Step 2: Feature-Rich Forecast with LightGBM (quantile regression)

When external features matter (promos, holidays, weather), a gradient-boosted model often wins.

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# ── Feature engineering ────────────────────────────────────────
demand["week_of_year"] = demand["week_start"].dt.isocalendar().week.astype(int)
demand["month"] = demand["week_start"].dt.month
demand["is_summer"] = demand["month"].isin([12, 1, 2]).astype(int)

# Lag features
for lag in [1, 2, 4, 52]:
    demand[f"demand_lag_{lag}"] = demand.groupby("sku")["units_needed"].shift(lag)

# Rolling stats
demand["demand_roll_4w_mean"] = (
    demand.groupby("sku")["units_needed"]
    .transform(lambda x: x.rolling(4).mean())
)

features = [
    "week_of_year", "month", "is_summer",
    "promo_active", "public_holiday_next_week", "avg_temp_c",
    "demand_lag_1", "demand_lag_2", "demand_lag_4", "demand_lag_52",
    "demand_roll_4w_mean",
]

target = "units_needed"

train = demand.dropna(subset=features)
X_train = train[features]
y_train = train[target]

# ── Train three quantile models: q10, q50, q90 ────────────────
quantile_models = {}
for alpha in [0.10, 0.50, 0.90]:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X_train, y_train)
    quantile_models[f"q{int(alpha*100)}"] = model

# ── Predict next week ─────────────────────────────────────────
next_week_features = pd.DataFrame([{
    "week_of_year": 17,
    "month": 4,
    "is_summer": 0,
    "promo_active": 0,
    "public_holiday_next_week": 1,  # ANZAC Day
    "avg_temp_c": 18.0,
    "demand_lag_1": 2900,
    "demand_lag_2": 3100,
    "demand_lag_4": 3450,
    "demand_lag_52": 3350,
    "demand_roll_4w_mean": 3162.5,
}])

print("Demand forecast for CHKBRST500, week of 2025-04-21:")
for name, model in quantile_models.items():
    pred = model.predict(next_week_features)[0]
    print(f"  {name}: {pred:.0f} units")
```

**Example output:**

```
Demand forecast for CHKBRST500, week of 2025-04-21:
  q10: 2870 units
  q50: 3210 units
  q90: 3580 units
```

### Step 3: Apply Dave's Rules as Post-Processing

```python
import json
from datetime import date

def apply_daves_rules(
    sku: str,
    forecast_q50: float,
    forecast_q90: float,
    context: dict,
    rules: list[dict],
) -> dict:
    """Apply SME business rules on top of statistical forecast."""

    adjusted_forecast = forecast_q50
    order_qty_multiplier = 1.0
    lead_time_add = 0
    safety_stock_multiplier = 1.0
    explanations = []

    for rule in rules:
        # Check if rule applies to this SKU
        pattern = rule["sku_pattern"]
        if pattern != "*" and not sku.startswith(pattern.replace("*", "")):
            continue

        # Evaluate conditions (simplified — production would use a rule engine)
        if "public_holiday_next_week" in rule["condition"] and "Easter" in rule["condition"]:
            if context.get("public_holiday_next_week") and context.get("holiday_name") == "Easter":
                adjusted_forecast *= 1.35
                explanations.append(f"[DAVE_001] Easter uplift: +35% → {adjusted_forecast:.0f}")

        elif "avg_temp_forecast > 30" in rule["condition"]:
            if context.get("avg_temp_c", 0) > 30:
                adjusted_forecast *= 0.92
                explanations.append(f"[DAVE_004] Extreme heat discount: -8% → {adjusted_forecast:.0f}")

        elif "new_menu_item" in rule["condition"]:
            if context.get("new_menu_item"):
                safety_stock_multiplier = 1.5
                explanations.append(f"[DAVE_005] New menu item: safety stock ×1.5")

    return {
        "sku": sku,
        "base_forecast_q50": forecast_q50,
        "adjusted_forecast": round(adjusted_forecast),
        "order_qty_multiplier": order_qty_multiplier,
        "lead_time_add_days": lead_time_add,
        "safety_stock_multiplier": safety_stock_multiplier,
        "explanations": explanations,
    }


# ── Example: ANZAC Day week (not Easter, no extreme heat) ──────
with open("data/daves_rules.json") as f:
    rules = json.load(f)

result = apply_daves_rules(
    sku="CHKBRST500",
    forecast_q50=3210,
    forecast_q90=3580,
    context={
        "public_holiday_next_week": True,
        "holiday_name": "ANZAC Day",
        "avg_temp_c": 18.0,
        "new_menu_item": False,
    },
    rules=rules,
)
print(result)
# No Dave's rules fire for this context → forecast unchanged at 3210

# ── Example: Easter week with a new recipe ─────────────────────
result_easter = apply_daves_rules(
    sku="CHKBRST500",
    forecast_q50=3210,
    forecast_q90=3580,
    context={
        "public_holiday_next_week": True,
        "holiday_name": "Easter",
        "avg_temp_c": 22.0,
        "new_menu_item": True,
    },
    rules=rules,
)
print(result_easter)
# Output:
# {
#   "sku": "CHKBRST500",
#   "base_forecast_q50": 3210,
#   "adjusted_forecast": 4334,         ← 3210 × 1.35
#   "safety_stock_multiplier": 1.5,    ← new menu item buffer
#   "explanations": [
#     "[DAVE_001] Easter uplift: +35% → 4334",
#     "[DAVE_005] New menu item: safety stock ×1.5"
#   ]
# }
```

### Step 4: Inventory Optimisation with PuLP

Given probabilistic forecasts and constraints (shelf life, MOQs, multi-supplier), determine optimal order quantities.

```python
from pulp import (
    LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value,
)

# ── Problem parameters ─────────────────────────────────────────
skus = ["CHKBRST500", "AVOCADO_HASS", "TOMATO_ROMA_1KG"]

# Forecast demand (adjusted, q50) for next week
forecast = {
    "CHKBRST500": 3210,
    "AVOCADO_HASS": 4500,
    "TOMATO_ROMA_1KG": 3800,
}

# Current on-hand inventory
on_hand = {
    "CHKBRST500": 400,
    "AVOCADO_HASS": 200,
    "TOMATO_ROMA_1KG": 600,
}

# Supplier options: (supplier, sku) → {cost, moq, shelf_life, lead_time}
suppliers = {
    ("Inghams", "CHKBRST500"):       {"cost": 4.50, "moq": 500, "shelf_life": 7},
    ("Baiada", "CHKBRST500"):         {"cost": 4.75, "moq": 200, "shelf_life": 6},
    ("CostaCorp", "AVOCADO_HASS"):    {"cost": 1.20, "moq": 300, "shelf_life": 5},
    ("PerfFresh", "TOMATO_ROMA_1KG"): {"cost": 2.10, "moq": 200, "shelf_life": 8},
}

# Cost parameters
waste_cost_per_unit = {  # disposal + lost margin
    "CHKBRST500": 5.00,
    "AVOCADO_HASS": 1.50,
    "TOMATO_ROMA_1KG": 2.50,
}

stockout_cost_per_unit = {  # lost sale + customer churn risk
    "CHKBRST500": 12.00,
    "AVOCADO_HASS": 8.00,
    "TOMATO_ROMA_1KG": 6.00,
}

# Safety stock: cover (q90 - q50) to buffer upside demand risk
safety_stock_units = {
    "CHKBRST500": 370,    # 3580 - 3210
    "AVOCADO_HASS": 500,
    "TOMATO_ROMA_1KG": 300,
}

# Warehouse capacity constraint (total units across all SKUs)
warehouse_capacity = 12000

# ── Build the optimisation model ──────────────────────────────
prob = LpProblem("FreshBox_Replenishment", LpMinimize)

# Decision variables: order quantity from each (supplier, sku) pair
order_vars = {}
for (sup, sku), info in suppliers.items():
    var_name = f"order_{sup}_{sku}"
    order_vars[(sup, sku)] = LpVariable(var_name, lowBound=0, cat="Integer")

# Binary variables: whether we order from a supplier (to enforce MOQ)
use_supplier = {}
for (sup, sku) in suppliers:
    var_name = f"use_{sup}_{sku}"
    use_supplier[(sup, sku)] = LpVariable(var_name, cat="Binary")

# Helper: total order per SKU
def total_order(sku):
    return lpSum(
        order_vars[(s, sk)] for (s, sk) in order_vars if sk == sku
    )

# ── Objective: minimise procurement cost + expected waste cost ─
# Simplified: waste ∝ max(0, on_hand + order - demand) for perishables
# We use a slack variable to linearise

overstock = {}
for sku in skus:
    overstock[sku] = LpVariable(f"overstock_{sku}", lowBound=0)

prob += (
    # Procurement cost
    lpSum(
        suppliers[(s, sk)]["cost"] * order_vars[(s, sk)]
        for (s, sk) in suppliers
    )
    # + Expected waste cost (overstock that may expire)
    + lpSum(
        waste_cost_per_unit[sku] * 0.3 * overstock[sku]  # 30% of overstock wasted
        for sku in skus
    )
), "Total_Cost"

# ── Constraints ───────────────────────────────────────────────

# 1. Meet demand + safety stock
for sku in skus:
    prob += (
        on_hand[sku] + total_order(sku) >= forecast[sku] + safety_stock_units[sku],
        f"Meet_demand_{sku}",
    )

# 2. Overstock definition
for sku in skus:
    prob += (
        overstock[sku] >= on_hand[sku] + total_order(sku) - forecast[sku],
        f"Overstock_def_{sku}",
    )

# 3. MOQ enforcement: if you order from a supplier, order at least MOQ
BIG_M = 50000
for (sup, sku), info in suppliers.items():
    prob += (
        order_vars[(sup, sku)] >= info["moq"] * use_supplier[(sup, sku)],
        f"MOQ_min_{sup}_{sku}",
    )
    prob += (
        order_vars[(sup, sku)] <= BIG_M * use_supplier[(sup, sku)],
        f"MOQ_activate_{sup}_{sku}",
    )

# 4. Warehouse capacity
prob += (
    lpSum(on_hand[sku] + total_order(sku) for sku in skus) <= warehouse_capacity,
    "Warehouse_capacity",
)

# ── Solve ──────────────────────────────────────────────────────
prob.solve()

print(f"Status: {LpStatus[prob.status]}")
print(f"Total cost: ${value(prob.objective):,.2f}\n")

print("Order plan:")
for (sup, sku), var in order_vars.items():
    qty = int(value(var))
    if qty > 0:
        print(f"  {sup:15s} → {sku:20s}: {qty:,} units @ ${suppliers[(sup,sku)]['cost']:.2f} = ${qty * suppliers[(sup,sku)]['cost']:,.2f}")

print("\nInventory position after ordering:")
for sku in skus:
    total = on_hand[sku] + int(value(total_order(sku)))
    surplus = total - forecast[sku]
    print(f"  {sku:20s}: {total:,} units (demand: {forecast[sku]:,}, buffer: {surplus:,})")
```

**Example output:**

```
Status: Optimal
Total cost: $22,837.50

Order plan:
  Inghams         → CHKBRST500          : 3,180 units @ $4.50 = $14,310.00
  CostaCorp       → AVOCADO_HASS        : 4,800 units @ $1.20 = $5,760.00
  PerfFresh       → TOMATO_ROMA_1KG     : 3,500 units @ $2.10 = $7,350.00

Inventory position after ordering:
  CHKBRST500          : 3,580 units (demand: 3,210, buffer: 370)
  AVOCADO_HASS        : 5,000 units (demand: 4,500, buffer: 500)
  TOMATO_ROMA_1KG     : 4,100 units (demand: 3,800, buffer: 300)
```

> Note: The solver chose Ingham's over Baiada for chicken because of the lower unit cost ($4.50 vs $4.75), even though Baiada has a lower MOQ. It ordered exactly enough to cover demand + safety stock, minimising waste.

### Step 5: Decision Support Dashboard (Streamlit)

```python
# app.py — run with: streamlit run app.py

import streamlit as st
import pandas as pd

st.set_page_config(page_title="FreshBox Replenishment", layout="wide")
st.title("🥦 FreshBox Replenishment Recommendations")

# ── Simulated model outputs ────────────────────────────────────
recommendations = pd.DataFrame([
    {
        "SKU": "CHKBRST500",
        "Description": "Chicken Breast 500g",
        "On Hand": 400,
        "Forecast (q50)": 3210,
        "Forecast (q90)": 3580,
        "Recommended Order": 3180,
        "Supplier": "Ingham's",
        "Unit Cost": 4.50,
        "Total Cost": 14310.00,
        "Confidence": "High",
        "Alerts": "",
    },
    {
        "SKU": "AVOCADO_HASS",
        "Description": "Hass Avocado",
        "On Hand": 200,
        "Forecast (q50)": 4500,
        "Forecast (q90)": 5000,
        "Recommended Order": 4800,
        "Supplier": "CostaCorp",
        "Unit Cost": 1.20,
        "Total Cost": 5760.00,
        "Confidence": "Medium",
        "Alerts": "⚠️ Summer — QLD lead time +1 day (Dave's Rule #002)",
    },
    {
        "SKU": "TOMATO_ROMA_1KG",
        "Description": "Roma Tomato 1kg",
        "On Hand": 600,
        "Forecast (q50)": 3800,
        "Forecast (q90)": 4100,
        "Recommended Order": 3500,
        "Supplier": "Perfection Fresh",
        "Unit Cost": 2.10,
        "Total Cost": 7350.00,
        "Confidence": "High",
        "Alerts": "",
    },
])

# ── Summary metrics ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Order Value", f"${recommendations['Total Cost'].sum():,.0f}")
col2.metric("SKUs to Order", len(recommendations))
col3.metric("Warehouse Utilisation", "87%")
col4.metric("Projected Waste Rate", "2.1%", delta="-0.8%", delta_color="inverse")

st.divider()

# ── Order recommendations table ────────────────────────────────
st.subheader("Order Recommendations for Week of 21 Apr 2025")

for _, row in recommendations.iterrows():
    with st.expander(f"**{row['Description']}** ({row['SKU']}) — Order: {row['Recommended Order']:,} units"):
        c1, c2, c3 = st.columns(3)
        c1.metric("On Hand", f"{row['On Hand']:,}")
        c2.metric("Forecast (median)", f"{row['Forecast (q50)']:,}")
        c3.metric("Forecast (high)", f"{row['Forecast (q90)']:,}")

        st.write(f"**Supplier:** {row['Supplier']} @ ${row['Unit Cost']:.2f}/unit")
        st.write(f"**Total cost:** ${row['Total Cost']:,.2f}")
        st.write(f"**Model confidence:** {row['Confidence']}")

        if row["Alerts"]:
            st.warning(row["Alerts"])

        # LLM-generated explanation (pre-computed or called on-demand)
        st.info(
            f"**Why this quantity?** The model forecasts median demand of "
            f"{row['Forecast (q50)']:,} units with a high-scenario of {row['Forecast (q90)']:,}. "
            f"After accounting for {row['On Hand']:,} units on hand, we recommend ordering "
            f"{row['Recommended Order']:,} units to maintain a safety buffer of "
            f"{row['On Hand'] + row['Recommended Order'] - row['Forecast (q50)']:,} units "
            f"against demand variability."
        )

# ── Override section ───────────────────────────────────────────
st.divider()
st.subheader("Manual Overrides")
st.write("Adjust any order quantity below. Overrides are logged for model retraining.")

override_sku = st.selectbox("SKU", recommendations["SKU"].tolist())
override_qty = st.number_input("Override quantity", min_value=0, step=100)
override_reason = st.text_input("Reason for override")

if st.button("Submit Override"):
    st.success(f"Override recorded: {override_sku} → {override_qty} units. Reason: {override_reason}")
    # In production: log to database for feedback loop
```

### Step 6: LLM-Powered Explanation Generator

```python
# explanation_generator.py
# Uses an LLM to turn model outputs into plain-English explanations

from openai import OpenAI  # or anthropic, google.generativeai, etc.

client = OpenAI()  # assumes OPENAI_API_KEY in env


def generate_explanation(order_context: dict) -> str:
    """Generate a plain-English explanation of a replenishment recommendation."""

    prompt = f"""You are an inventory analyst at a food box delivery company.
Explain the following replenishment recommendation in 2-3 sentences.
Use plain language suitable for a warehouse operator who is not a data scientist.
Mention any active business rules and why the quantity was chosen.

Context:
- SKU: {order_context['sku']} ({order_context['description']})
- Current on-hand inventory: {order_context['on_hand']:,} units
- Forecast demand (median): {order_context['forecast_q50']:,} units
- Forecast demand (high scenario): {order_context['forecast_q90']:,} units
- Recommended order quantity: {order_context['order_qty']:,} units
- Supplier: {order_context['supplier']} at ${order_context['unit_cost']:.2f}/unit
- Active business rules: {order_context.get('active_rules', 'None')}
- Upcoming events: {order_context.get('events', 'None')}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )

    return response.choices[0].message.content


# ── Example usage ──────────────────────────────────────────────
explanation = generate_explanation({
    "sku": "CHKBRST500",
    "description": "Chicken Breast 500g",
    "on_hand": 400,
    "forecast_q50": 4334,
    "forecast_q90": 4831,
    "order_qty": 4930,
    "supplier": "Ingham's",
    "unit_cost": 4.50,
    "active_rules": "DAVE_001 (Easter uplift +35%), DAVE_005 (new menu item → safety stock ×1.5)",
    "events": "Easter long weekend, new 'Lemon Herb Chicken' recipe launching",
})

print(explanation)
# Example output:
# "We're recommending a larger-than-usual order of 4,930 units of Chicken Breast
#  this week. Demand is expected to jump about 35% because of the Easter long
#  weekend, and we're also adding extra safety stock because the new Lemon Herb
#  Chicken recipe is launching — new recipes can have unpredictable uptake.
#  Combined with the 400 units already in the warehouse, this gives us enough
#  buffer to cover even the high-demand scenario."
```

---

## End-to-End Example: What Happens on a Monday Morning

**Date: Monday 14 April 2025 — Easter week**

| Time | What happens |
|---|---|
| 6:00 AM | Automated pipeline runs: pulls latest order data, weather forecast, menu for the week. |
| 6:05 AM | LightGBM quantile models produce demand forecasts for all 200 SKUs. |
| 6:06 AM | Dave's Rules engine applies post-processing (Easter uplift fires for all SKUs, mango bruising rule fires for MANGO_KENT). |
| 6:08 AM | PuLP optimisation model runs: determines optimal order quantities considering MOQs, shelf life, warehouse capacity, multi-supplier costs. |
| 6:10 AM | LLM generates explanations for each SKU recommendation. |
| 6:12 AM | Streamlit dashboard refreshes. Warehouse team lead Sarah opens it on her tablet. |
| 6:15 AM | Sarah reviews recommendations. She sees a yellow alert on avocados (summer lead time rule) and confirms the extra lead time is already accounted for. |
| 6:20 AM | Sarah notices the model recommends 4,930 units of chicken — much higher than usual. The explanation says "Easter uplift + new recipe". She agrees and approves. |
| 6:25 AM | She overrides the mango order from 2,400 to 2,600 because she knows this week's supplier tends to short-deliver. She logs the reason. |
| 6:30 AM | Purchase orders are auto-generated and sent to suppliers via email/EDI. |
| 6:35 AM | Sarah's override is logged. Next month, the model retraining pipeline picks this up as a feature: `supplier_short_delivery_history`. |

---

## Key Metrics to Track

```python
# metrics.py — weekly model performance tracking

def calculate_metrics(actuals: pd.Series, forecasts_q50: pd.Series,
                      forecasts_q10: pd.Series, forecasts_q90: pd.Series,
                      orders: pd.Series, waste: pd.Series) -> dict:
    """Calculate key supply chain performance metrics."""

    demand = actuals
    return {
        # Forecast accuracy
        "MAPE": ((actuals - forecasts_q50).abs() / actuals).mean() * 100,
        "Bias": ((forecasts_q50 - actuals) / actuals).mean() * 100,  # +ve = over-forecast
        "Coverage_80": ((actuals >= forecasts_q10) & (actuals <= forecasts_q90)).mean() * 100,

        # Service level
        "Fill_rate_pct": (1 - (demand - orders).clip(lower=0).sum() / demand.sum()) * 100,
        "Stockout_skus": (orders < demand).sum(),

        # Waste
        "Waste_rate_pct": waste.sum() / orders.sum() * 100,
        "Waste_cost_aud": (waste * 3.50).sum(),  # avg waste cost per unit

        # Efficiency
        "Override_rate_pct": 12.5,  # from override logs
        "Avg_order_value_aud": (orders * 3.20).mean(),
    }

# Target benchmarks:
# - MAPE < 15% for weekly SKU-level forecasts
# - 80% prediction interval coverage ≈ 80% (calibration)
# - Fill rate > 98.5%
# - Waste rate < 3%
# - Override rate trending down over time (model learning from humans)
```

---

## Cost-Benefit Summary (Illustrative)

| Item | Before (SME-driven) | After (model-assisted) |
|---|---|---|
| Weekly waste rate | ~4.5% | ~2.1% (target) |
| Weekly waste cost | ~$12,600 | ~$5,900 |
| Stockout incidents/week | ~3-5 SKUs | ~0-1 SKUs |
| Time to make ordering decisions | 3-4 hours (Dave) | 30 min (Sarah reviewing dashboard) |
| Dependency on single SME | Critical risk | Eliminated |
| Annual savings estimate | — | ~$350,000 in waste reduction + avoided stockouts |
| Implementation cost (Phase 1) | — | ~$80,000-120,000 (data eng + ML eng, 3 months) |

---

## Project Structure

```
supply_chain_optimisation/
├── data/
│   ├── weekly_demand.csv
│   ├── suppliers.csv
│   ├── daves_rules.json
│   └── menu_calendar.csv
├── src/
│   ├── forecasting/
│   │   ├── statsforecast_baseline.py    # Step 1
│   │   ├── lgbm_quantile.py            # Step 2
│   │   └── feature_engineering.py
│   ├── rules_engine/
│   │   └── daves_rules.py              # Step 3
│   ├── optimisation/
│   │   └── replenishment_model.py      # Step 4
│   ├── explanations/
│   │   └── llm_explainer.py            # Step 6
│   └── metrics/
│       └── performance_tracking.py      # Metrics
├── dashboard/
│   └── app.py                           # Step 5 (Streamlit)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_forecast_experiments.ipynb
│   └── 03_backtest.ipynb
├── tests/
│   ├── test_rules_engine.py
│   └── test_optimisation.py
├── requirements.txt
└── README.md
```
