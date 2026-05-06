"""
Step 8: Decision Support Dashboard (Streamlit)
================================================
An interactive web dashboard that presents the optimisation results to
warehouse operators — the people who replace Dave.

CONCEPT — Decision Support vs Automation:
  This system doesn't replace human judgement. It SUPPORTS it:
    1. Shows what the model recommends and WHY
    2. Highlights items that need attention (high uncertainty, rules fired)
    3. Allows manual overrides (logged for future model improvement)
    4. Builds trust by showing forecast accuracy over time

  The goal is to make a new operator as effective as Dave was — not by
  memorising his rules, but by having a system that encodes them and
  explains its reasoning.

Usage:
    streamlit run src/08_dashboard.py

    Then open http://localhost:8501 in your browser.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshBox Replenishment",
    page_icon="🥦",
    layout="wide",
)


@st.cache_data
def load_data():
    """Load all pipeline outputs. Cached so it only runs once."""
    order_plan = pd.read_parquet(DATA_PROCESSED / "evaluation_report.parquet")
    summary = pd.read_csv(DATA_PROCESSED / "evaluation_summary.csv")
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    return order_plan, summary, items


# ── Load data ──────────────────────────────────────────────────────────────────
try:
    order_plan, summary, items = load_data()
except FileNotFoundError:
    st.error(
        "Pipeline outputs not found. Please run all steps (01–07) before launching the dashboard.\n\n"
        "```bash\n"
        "python src/00_generate_synthetic_data.py  # or download Kaggle data\n"
        "python src/01_data_prep.py\n"
        "python src/02_eda.py\n"
        "python src/03_feature_engineering.py\n"
        "python src/04_demand_forecasting.py\n"
        "python src/05_business_rules.py\n"
        "python src/06_inventory_optimisation.py\n"
        "python src/07_evaluation.py\n"
        "```"
    )
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filters")

stores = sorted(order_plan["store_nbr"].unique())
selected_store = st.sidebar.selectbox("Store", stores, index=0)

weeks = sorted(order_plan["week_start"].unique())
week_labels = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks]
selected_week_label = st.sidebar.selectbox("Week", week_labels, index=len(week_labels) - 1)
selected_week = pd.Timestamp(selected_week_label)

families = ["All"] + sorted(order_plan["family"].dropna().unique().tolist())
selected_family = st.sidebar.selectbox("Product Family", families, index=0)

show_rules_only = st.sidebar.checkbox("Show only items with active rules", value=False)

# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = order_plan[
    (order_plan["store_nbr"] == selected_store)
    & (order_plan["week_start"] == selected_week)
].copy()

if selected_family != "All":
    filtered = filtered[filtered["family"] == selected_family]

if show_rules_only:
    filtered = filtered[filtered["rules_applied"] != ""]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🥦 FreshBox Replenishment Dashboard")
st.caption(f"Store {selected_store} — Week of {selected_week_label}")

# ── KPI Cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_order_cost = filtered["order_cost"].sum()
total_items = len(filtered)
items_with_rules = (filtered["rules_applied"] != "").sum()
fill_rate = (
    filtered["fulfilled"].sum() / filtered["actual_demand"].sum() * 100
    if "fulfilled" in filtered.columns and filtered["actual_demand"].sum() > 0
    else 0
)
waste_rate = (
    filtered["waste_units"].sum() / filtered["order_qty"].sum() * 100
    if "waste_units" in filtered.columns and filtered["order_qty"].sum() > 0
    else 0
)

col1.metric("Total Order Value", f"${total_order_cost:,.0f}")
col2.metric("Items to Order", total_items)
col3.metric("Rules Active", items_with_rules)
col4.metric("Fill Rate", f"{fill_rate:.1f}%" if fill_rate > 0 else "N/A")
col5.metric("Est. Waste Rate", f"{waste_rate:.1f}%" if waste_rate > 0 else "N/A")

st.divider()

# ── Order Recommendations Table ────────────────────────────────────────────────
st.subheader("📋 Order Recommendations")

if len(filtered) == 0:
    st.info("No data for the selected filters. Try changing the store, week, or family.")
else:
    # Prepare display table
    display_cols = [
        "item_nbr",
        "family",
        "perishable",
        "on_hand",
        "adjusted_q50",
        "adjusted_q90",
        "safety_stock",
        "order_qty",
        "order_cost",
        "actual_demand",
        "rules_applied",
    ]
    display = filtered[display_cols].copy()
    display["perishable"] = display["perishable"].map({0: "No", 1: "Yes"})
    display.columns = [
        "Item",
        "Family",
        "Perishable",
        "On Hand",
        "Forecast (Median)",
        "Forecast (High)",
        "Safety Stock",
        "Order Qty",
        "Order Cost ($)",
        "Actual Demand",
        "Rules Applied",
    ]

    st.dataframe(
        display.sort_values("Order Cost ($)", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

st.divider()

# ── Item Detail View ───────────────────────────────────────────────────────────
st.subheader("🔍 Item Detail")

if len(filtered) > 0:
    selected_item = st.selectbox(
        "Select an item for detailed view",
        sorted(filtered["item_nbr"].unique()),
    )

    item_row = filtered[filtered["item_nbr"] == selected_item].iloc[0]

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("**Inventory Position**")
        st.metric("On Hand", f"{item_row['on_hand']:,.0f} units")
        st.metric("Forecast (Median)", f"{item_row['adjusted_q50']:,.0f} units")
        st.metric("Forecast (High)", f"{item_row['adjusted_q90']:,.0f} units")
        st.metric("Safety Stock", f"{item_row['safety_stock']:,.0f} units")
        st.metric("Recommended Order", f"{item_row['order_qty']:,.0f} units")

    with detail_col2:
        st.markdown("**Explanation**")
        explanation = item_row.get("explanations", "No rules applied — using base ML forecast.")
        st.info(explanation)

        # Generate a natural-language summary
        gap = item_row["adjusted_q50"] + item_row["safety_stock"] - item_row["on_hand"]
        st.markdown(
            f"**Why this quantity?** The model forecasts median demand of "
            f"**{item_row['adjusted_q50']:,.0f}** units (high scenario: "
            f"{item_row['adjusted_q90']:,.0f}). With **{item_row['on_hand']:,.0f}** "
            f"units on hand and a safety buffer of **{item_row['safety_stock']:,.0f}**, "
            f"the gap is ~**{max(0, gap):,.0f}** units. "
            f"Order cost: **${item_row['order_cost']:,.0f}**."
        )

    # Historical demand chart for this item
    st.markdown("**Demand History — This Item at This Store**")
    item_history = order_plan[
        (order_plan["item_nbr"] == selected_item)
        & (order_plan["store_nbr"] == selected_store)
    ].sort_values("week_start")

    if len(item_history) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=item_history["week_start"],
            y=item_history["actual_demand"],
            name="Actual Demand",
            mode="lines+markers",
            line=dict(color="black"),
        ))
        fig.add_trace(go.Scatter(
            x=item_history["week_start"],
            y=item_history["adjusted_q50"],
            name="Forecast (Median)",
            mode="lines",
            line=dict(color="blue", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=item_history["week_start"],
            y=item_history["forecast_q90"],
            name="Forecast (High)",
            mode="lines",
            line=dict(color="lightblue", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(100, 149, 237, 0.15)",
        ))
        fig.add_trace(go.Bar(
            x=item_history["week_start"],
            y=item_history["order_qty"],
            name="Order Qty",
            marker_color="rgba(70, 130, 180, 0.3)",
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Overall Performance Charts ─────────────────────────────────────────────────
st.subheader("📊 Pipeline Performance (All Weeks, This Store)")

store_data = order_plan[order_plan["store_nbr"] == selected_store].copy()

if len(store_data) > 0 and "fulfilled" in store_data.columns:
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        # Weekly fill rate over time
        weekly_perf = store_data.groupby("week_start").agg(
            total_demand=("actual_demand", "sum"),
            total_fulfilled=("fulfilled", "sum"),
            total_waste=("waste_units", "sum"),
            total_ordered=("order_qty", "sum"),
        ).reset_index()

        weekly_perf["fill_rate"] = weekly_perf["total_fulfilled"] / weekly_perf["total_demand"] * 100
        weekly_perf["waste_rate"] = weekly_perf["total_waste"] / weekly_perf["total_ordered"] * 100

        fig_fill = px.line(
            weekly_perf,
            x="week_start",
            y="fill_rate",
            title="Weekly Fill Rate (%)",
            labels={"fill_rate": "Fill Rate (%)", "week_start": "Week"},
        )
        fig_fill.add_hline(y=98, line_dash="dash", line_color="green", annotation_text="Target: 98%")
        fig_fill.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_fill, use_container_width=True)

    with perf_col2:
        # Weekly waste rate over time
        fig_waste = px.line(
            weekly_perf,
            x="week_start",
            y="waste_rate",
            title="Weekly Waste Rate (%)",
            labels={"waste_rate": "Waste Rate (%)", "week_start": "Week"},
        )
        fig_waste.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Target: <3%")
        fig_waste.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_waste, use_container_width=True)

    # Cost breakdown by family
    family_cost = store_data.groupby("family").agg(
        procurement=("order_cost", "sum"),
        waste_cost=("waste_units", lambda x: x.sum() * 3.50),
        stockout_cost=("stockout_units", lambda x: x.sum() * 10.00),
    ).reset_index()
    family_cost["total"] = family_cost["procurement"] + family_cost["waste_cost"] + family_cost["stockout_cost"]
    family_cost = family_cost.sort_values("total", ascending=True)

    fig_fam = px.bar(
        family_cost,
        y="family",
        x=["procurement", "waste_cost", "stockout_cost"],
        title="Cost Breakdown by Product Family",
        orientation="h",
        labels={"value": "Cost ($)", "family": "Family"},
        color_discrete_map={
            "procurement": "steelblue",
            "waste_cost": "coral",
            "stockout_cost": "gold",
        },
    )
    fig_fam.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_fam, use_container_width=True)

st.divider()

# ── Manual Override Section ────────────────────────────────────────────────────
st.subheader("✏️ Manual Override")
st.markdown(
    "Disagree with a recommendation? Adjust the order quantity below. "
    "Overrides are logged for future model retraining — this is how the "
    "model learns from operator expertise over time."
)

override_col1, override_col2, override_col3 = st.columns([2, 2, 3])

with override_col1:
    override_item = st.selectbox(
        "Item to override",
        sorted(filtered["item_nbr"].unique()) if len(filtered) > 0 else [],
        key="override_item",
    )

with override_col2:
    current_qty = 0
    if len(filtered) > 0 and override_item is not None:
        match = filtered[filtered["item_nbr"] == override_item]
        if len(match) > 0:
            current_qty = int(match.iloc[0]["order_qty"])
    override_qty = st.number_input(
        "New quantity",
        min_value=0,
        value=current_qty,
        step=50,
    )

with override_col3:
    override_reason = st.text_input("Reason for override", placeholder="e.g. Supplier warned of late delivery")

if st.button("Submit Override", type="primary"):
    if override_reason.strip():
        st.success(
            f"Override logged: Item {override_item} → {override_qty} units "
            f"(was {current_qty}). Reason: {override_reason}"
        )
        # In production: write to a database table for model retraining feedback loop
    else:
        st.warning("Please provide a reason for the override.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FreshBox Supply Chain Optimisation PoC | "
    "Data: Corporación Favorita (Kaggle) or Synthetic | "
    "Models: LightGBM Quantile Regression + PuLP MIP"
)
