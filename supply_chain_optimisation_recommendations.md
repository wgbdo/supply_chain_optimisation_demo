# Supply Chain Optimisation Recommendations: Food Box Distribution (Australia)

## Context

A food distribution company in Australia packages and delivers food boxes to households. Inventory decisions are currently driven by SME experience and live inventory counts. With SME attrition, the goal is to encode this knowledge into a system that supports less-experienced operators in making high-quality inventory and replenishment decisions.

**Their data currently lives in SAP**, which is a significant factor in the technology and platform choices below.

---

## 1. Is Forecast-Then-Optimise Still the Best Approach?

**Short answer: it's still a strong foundation, but it's no longer the only — or always the best — option.**

The classic two-stage pipeline (demand forecast → deterministic/stochastic optimisation) remains well-understood and interpretable. However, it has known weaknesses:

- **Error propagation**: forecast errors feed directly into the optimiser, which treats the forecast as truth.
- **Misaligned objectives**: the forecasting model minimises prediction error (e.g. MAE/RMSE), not business cost (e.g. stockout cost vs. waste cost). For perishable food this mismatch is especially painful.
- **Static assumptions**: classical safety-stock and (s, S) / (R, Q) policies assume stationary demand distributions, which rarely hold for food boxes with seasonal, promotional, and trend-driven demand.

Modern advances offer three progressively more integrated alternatives:

| Approach | Description | When to use |
|---|---|---|
| **Forecast-then-Optimise (enhanced)** | ML-based probabilistic demand forecasting (quantile/distributional) feeding into a stochastic or robust optimisation model | Good baseline; interpretable; works well when you have clean historical data and relatively simple constraints |
| **Predict-then-Optimise with decision-aware loss** | Train the forecasting model using a loss function that accounts for downstream decision costs (SPO+, decision-focused learning) | When asymmetric costs matter a lot (stockout vs. waste), which is exactly the case for perishable food |
| **End-to-end / Reinforcement Learning** | Skip the explicit forecast; learn inventory policies directly from state (inventory, calendar, weather, etc.) to action (order quantities) | When the action space is complex, constraints interact, and you have a good simulation environment |

**Recommendation for this company**: Start with **enhanced forecast-then-optimise** using probabilistic forecasts, then layer in **decision-aware training** once the baseline is operational. This gives you fast time-to-value with a clear upgrade path.

---

## 2. SAP as the Data Source — Key Considerations

Since all operational data resides in SAP (likely S/4HANA or ECC), the data extraction strategy is a critical early decision:

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **SAP Datasphere** | SAP's own cloud data layer; exposes SAP tables as federated/replicated datasets | Clean semantic layer, real-time capable, SAP-native | Licensing cost; ties you deeper into SAP ecosystem |
| **SAP BW/4HANA or BW extractors** | Traditional SAP data warehouse extraction | Well-understood by SAP teams; handles complex SAP data models | Heavy, slower to iterate |
| **CDS Views + OData / RFC** | Expose SAP data via Core Data Services views, consume via API | Lightweight, selective, good for specific tables (VBAK/VBAP for orders, MARD for inventory, MSEG for movements) | Requires ABAP/SAP development skills |
| **Third-party connectors** | Tools like Fivetran, Airbyte, or Informatica SAP connectors that replicate SAP tables to a cloud data warehouse (BigQuery, Snowflake, etc.) | Decouples ML platform from SAP; analysts work in SQL/Python | Added infrastructure; data latency |
| **DataRobot SAP integration** | DataRobot connects natively to SAP HANA, S/4HANA, and SAP Datasphere | Minimal data movement; pre-built supply chain AI apps for SAP | Requires DataRobot licensing (see Section 3.5) |

**Recommendation**: If the company is already invested in SAP and wants the fastest path, **DataRobot's native SAP integration** or **SAP Datasphere → ML platform** are the lowest-friction options. If they want more flexibility and to avoid vendor lock-in, extract to a cloud warehouse (e.g. BigQuery or Snowflake via Fivetran) and build on open-source tools.

**Key SAP tables to extract for supply chain modelling:**
- **Demand/Orders**: `VBAK`/`VBAP` (sales orders), `LIKP`/`LIPS` (deliveries)
- **Inventory**: `MARD` (stock per storage location), `MCHB` (batch stock), `MSEG` (material movements)
- **Purchasing**: `EKKO`/`EKPO` (purchase orders), `EKET` (delivery schedules)
- **Material master**: `MARA`/`MARC` (material data, MRP settings, shelf life)
- **Supplier**: `LFA1` (vendor master), `EINA`/`EINE` (purchasing info records)

---

## 3. Recommended Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PLATFORM                        │
│  SAP S/4HANA → (Datasphere / CDS Views / Fivetran)     │
│  + external sources (weather, promotions, public hols)  │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  DEMAND SENSING │  ← near-real-time signals
       │  & FORECASTING  │    (order run-rate, web traffic)
       └───────┬─────────┘
               │  probabilistic forecasts (quantiles / distributions)
       ┌───────▼─────────────┐
       │  INVENTORY           │
       │  OPTIMISATION        │  ← constraints: shelf life, supplier MOQs,
       │  (replenishment,     │    warehouse capacity, delivery routes
       │   safety stock,      │
       │   allocation)        │
       └───────┬──────────────┘
               │  recommended orders / allocations
       ┌───────▼──────────┐
       │  DECISION SUPPORT │  ← human-in-the-loop UI
       │  DASHBOARD        │    (shows recommendations + rationale)
       └──────────────────┘
```

---

## 4. Key Components and Tool Recommendations

### 4.1 Demand Forecasting

| Tool / Library | Type | Notes |
|---|---|---|
| **Amazon Forecast** | Managed service (AWS) | Low-code, probabilistic, handles cold-start SKUs. Good if already on AWS. |
| **Azure AI / Azure Machine Learning** | Managed service (Azure) | AutoML for time series with built-in probabilistic outputs. Good if on Azure. |
| **Google Vertex AI Forecast** | Managed service (GCP) | Similar managed offering on GCP. |
| **Nixtla (TimeGPT / StatsForecast / NeuralForecast)** | Open-source + API | TimeGPT is a foundation model for time series — zero-shot forecasting with minimal setup. StatsForecast and NeuralForecast give you classical and deep-learning models with a clean API. **Strong recommendation for fast delivery.** |
| **PyTorch Forecasting / GluonTS** | Open-source | DeepAR, Temporal Fusion Transformer, N-BEATS, etc. More control, more engineering effort. |
| **Prophet / NeuralProphet** | Open-source | Good starting point for trend + seasonality + holiday effects. Less suited for intermittent or bursty demand. |
| **LightGBM / XGBoost** | Open-source | Feature-engineered tabular approach. Often competitive with deep models, especially at scale with good feature engineering. |

**Recommendation**: Start with **Nixtla StatsForecast + TimeGPT** for fast baseline probabilistic forecasts. Graduate to **LightGBM quantile regression** or **Temporal Fusion Transformer** for SKUs where interpretability or feature richness matters.

### 4.2 Inventory Optimisation

| Tool / Library | Type | Notes |
|---|---|---|
| **Google OR-Tools** | Open-source | Mixed-integer programming, constraint programming. Free and powerful. |
| **PuLP + CBC / HiGHS** | Open-source (Python) | Simple LP/MIP modelling in Python. HiGHS is a modern, fast open-source solver. |
| **Pyomo + HiGHS / Gurobi** | Open-source + commercial solver | More expressive modelling language for complex constraints. |
| **Gurobi / CPLEX** | Commercial solver | Best-in-class solvers for large MIP problems. Gurobi offers academic and startup pricing. |
| **scipy.optimize** | Open-source | Fine for simple single-SKU newsvendor-type problems. |
| **Inventory optimisation platforms (e.g. Slimstock, EazyStock, Relex)** | SaaS | End-to-end platforms. Faster to deploy but less customisable and more expensive long-term. |

**Recommendation**: Use **PuLP or Pyomo with HiGHS** (open-source) for the optimisation layer. Move to **Gurobi** if problem scale or solve-time demands it. This keeps costs low and gives full control.

### 4.3 Perishable / Food-Specific Considerations

- **Shelf-life constraints** must be modelled explicitly. FIFO/FEFO (First Expired, First Out) logic in the optimisation.
- **Waste cost** is a first-class objective, not just a constraint. The model should trade off stockout risk vs. spoilage.
- **Substitution effects**: when one item is out of stock, demand may shift to another. Model cross-item effects if data supports it.
- **Supplier lead-time variability**: Australian supply chains can have variable lead times (domestic agriculture, import logistics). Use stochastic lead-time models.

### 4.4 Decision Support and Knowledge Capture

This is critical given the SME-loss problem. The system should not just output numbers — it should explain *why*.

| Tool / Approach | Notes |
|---|---|
| **Streamlit / Dash / Gradio** | Rapid Python dashboards to surface recommendations with explanations. |
| **LLM-powered explanations** | Use an LLM (GPT-4o, Claude, Gemini) to generate natural-language explanations of model decisions: "We recommend ordering 500 units of SKU-X because demand is forecast to spike 30% next week due to the Easter holiday, and current stock covers only 3 days." |
| **Structured knowledge base** | Interview remaining SMEs and encode their rules (e.g. "always order extra avocados before a long weekend") as business rules or features in the model. Use an LLM to help extract and structure this knowledge. |
| **Alerting / exception management** | Flag items where the model recommendation diverges significantly from what a human would expect, so operators learn and trust the system incrementally. |

### 4.5 Managed Platform Option: DataRobot

[DataRobot](https://www.datarobot.com/solutions/supply-chain-operations/) offers a **Supply Chain & Ops AI App Suite** that is particularly relevant here because of the company's SAP footprint. Key points:

**What it provides:**
- Pre-built agentic AI apps for **demand planning, lead time estimation, inventory management, and stockout risk reduction** — all core to this use case.
- **Native SAP integration**: connects directly to SAP HANA, S/4HANA, SAP IBP, and SAP Datasphere. This eliminates the need for a separate data extraction pipeline.
- AutoML for time series with built-in probabilistic forecasting, feature engineering, and model selection.
- AI governance and observability out of the box (model drift monitoring, approval workflows).
- Claimed results: **25% production cost savings**, **3× faster time to production**, **30% fewer forecasting errors** (vendor-reported).

**When DataRobot makes sense:**
- The team **does not have deep ML engineering capability** and wants to get to production fast.
- Being on SAP already means the integration path is well-trodden — DataRobot is an SAP Endorsed App.
- Budget exists for platform licensing (~$100K–$300K+/year depending on scale and contract — request a quote).
- The company values **governance and auditability** (regulatory, or internal controls).

**When to build custom instead:**
- The team has (or plans to hire) ML engineers who want full control over model architecture.
- Budget is constrained — open-source tools (Nixtla + PuLP + Streamlit) can achieve 80% of the value at a fraction of the cost.
- Highly bespoke optimisation constraints (e.g. complex multi-echelon, cross-docking) that may not fit neatly into a managed platform's templates.

**Verdict for this company**: DataRobot is a **strong contender given the SAP data source** and the SME-loss urgency. It can compress the Phase 1 timeline significantly (weeks instead of months). The trade-off is ongoing licensing cost and less customisation flexibility. A pragmatic approach: **evaluate DataRobot for the forecasting and demand planning layer** (where their SAP integration and AutoML shine), and **keep the optimisation layer custom** (PuLP/Pyomo) where perishable food constraints need precise modelling.

**Other managed platforms in this space:**

| Platform | SAP Integration | Supply Chain Focus | Notes |
|---|---|---|---|
| **DataRobot** | Native (SAP Endorsed App) | Strong (dedicated app suite) | Best SAP integration story; AutoML + pre-built apps |
| **Blue Yonder (by Panasonic)** | Strong (SAP partnership) | Core business | End-to-end supply chain planning; heavy, enterprise-grade |
| **o9 Solutions** | Yes | Strong | AI-native planning platform; good for integrated business planning |
| **Relex Solutions** | Yes | Strong (retail/food focus) | Particularly strong in **fresh food and perishables** — worth evaluating |
| **SAP IBP** | Native | Core SAP offering | Demand planning + inventory optimisation within SAP itself; less ML-advanced |
| **Kinaxis** | Yes | Strong | Real-time concurrent planning; good for complex supply networks |

---

## 5. Accelerators: What Can Deliver Faster?

| Accelerator | Time saving | Trade-off |
|---|---|---|
| **DataRobot + SAP integration** | Weeks instead of months — native SAP connectors + pre-built supply chain apps | Licensing cost (~$100K+/yr); less flexibility for custom optimisation |
| **Nixtla TimeGPT** (foundation model for time series) | Days instead of weeks for initial forecasts | Less customisable than training your own models |
| **Cloud AutoML** (AWS Forecast, Azure AutoML, Vertex AI) | Weeks instead of months for production forecasting | Vendor lock-in, ongoing cost; no native SAP integration |
| **LLM-assisted knowledge elicitation** | Days to capture SME heuristics | Requires validation; heuristics may be suboptimal |
| **Pre-built inventory optimisation templates** (e.g. Google OR-Tools examples, Pyomo cookbooks) | Days instead of weeks for optimisation models | Need adaptation to your specific constraints |
| **SaaS platforms** (Relex, Blue Yonder, o9) | Months instead of building custom | High licensing cost, less flexibility |

---

## 6. Recommended Phased Roadmap

Two parallel tracks depending on build-vs-buy decision:

### Option A: Managed Platform (DataRobot / Relex)

**Phase 1 — Deploy (1-2 months)**
- Connect DataRobot to SAP HANA / S/4HANA via native integration.
- Configure pre-built demand planning app with historical orders, inventory, and supplier data.
- Interview SMEs and encode rules as features / business logic in the platform.
- Deploy initial forecasts and review with operators.

**Phase 2 — Tune & Extend (2-4 months)**
- Add external features (weather, promotions, holiday calendar).
- Configure inventory management app; tune stockout vs. waste trade-offs.
- Overlay custom optimisation (PuLP/Pyomo) for perishable-specific shelf-life constraints if the platform's built-in capabilities don't cover them.
- Backtest against historical SME decisions.

**Phase 3 — Advanced (4-8 months)**
- Leverage DataRobot's model monitoring and auto-retraining.
- Add LLM-generated explanations for operator dashboard.
- Explore agentic AI capabilities for automated exception handling.

### Option B: Custom Build (Open-Source + SAP Extraction)

**Phase 1 — Foundation (1-3 months)**
- Set up SAP data extraction (CDS Views + Fivetran → cloud warehouse, or SAP Datasphere).
- Interview SMEs and capture decision rules in a structured format.
- Deploy baseline probabilistic demand forecasts using Nixtla or LightGBM.
- Build a simple replenishment recommendation engine (newsvendor / base-stock policy) using PuLP/Pyomo.
- Stand up a Streamlit dashboard for operators.

**Phase 2 — Enhancement (3-6 months)**
- Add external features to the forecast (promotions, holidays, weather, price).
- Implement shelf-life-aware multi-SKU optimisation with waste minimisation.
- Add LLM-generated explanations for recommendations.
- Backtest against historical decisions to quantify improvement over SME baseline.

**Phase 3 — Advanced (6-12 months)**
- Explore decision-aware forecasting (SPO+ / differentiable optimisation layers).
- Evaluate reinforcement learning for dynamic replenishment policies.
- Integrate demand sensing (real-time order velocity, web traffic) for short-horizon adjustments.
- Continuous model monitoring and retraining pipeline.

---

## 7. Summary

| Question | Answer |
|---|---|
| Is forecast-then-optimise still the best approach? | It's still a strong baseline, but use **probabilistic** forecasts and consider **decision-aware** training to avoid the cost-mismatch problem. |
| What's the fastest path to value? | **DataRobot** (native SAP integration + pre-built supply chain apps) for forecasting, plus **PuLP/HiGHS** for custom perishable optimisation. Alternatively, **Nixtla TimeGPT + PuLP + Streamlit** if budget is constrained. |
| Does SAP as the data source change things? | Yes — it makes **DataRobot** and **Relex** particularly attractive because both have strong SAP integration, reducing data engineering effort significantly. It also means SAP IBP is worth evaluating as a native option, though it's less ML-advanced. |
| What about DataRobot specifically? | Strong fit given SAP data, SME-loss urgency, and the need for fast deployment. Native SAP connectors + pre-built demand/inventory apps can compress Phase 1 to weeks. Trade-off: licensing cost and less flexibility for highly custom optimisation logic. |
| What about other off-the-shelf platforms? | **Relex** is worth a look specifically for fresh food/perishables. **Blue Yonder** and **o9** are enterprise-grade alternatives. All have SAP integration. |
| How do we capture SME knowledge? | Structured interviews + encode as business rules and model features. Use LLMs to accelerate extraction and generate operator-facing explanations. |
| What's the biggest risk? | **Data quality and SAP extraction complexity.** SAP data models are notoriously difficult to work with. Budget time for understanding the specific SAP configuration — custom fields, variant configurations, and company-specific table extensions. No model can compensate for missing or inaccurate demand, inventory, or waste data. |
