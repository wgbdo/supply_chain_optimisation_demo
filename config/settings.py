"""
Central configuration for the supply chain optimisation PoC.

All paths, parameters, and constants in one place so you don't have to hunt
through multiple files to change something.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"          # synthetic data output / fallback
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = DATA_PROCESSED / "plots"
MODELS_DIR = DATA_PROCESSED / "models"
RULES_PATH = PROJECT_ROOT / "business_rules" / "rules.json"

# Optional: path to real Kaggle Favorita CSVs.
# If this directory exists and contains train.csv, the pipeline will read from
# here instead of data/raw/ (which holds synthetic data).
# Set to None to always use synthetic data.
KAGGLE_DATA_PATH = Path(
    r"C:\Users\Wilter.Grobler\OneDrive - BDO\Documents\Youfoodz\kaggle_data_files"
)

# Resolved raw data source: use Kaggle data if available, otherwise synthetic
RAW_DATA_SOURCE = (
    KAGGLE_DATA_PATH
    if KAGGLE_DATA_PATH is not None and (KAGGLE_DATA_PATH / "train.csv").exists()
    else DATA_RAW
)

# Create output directories if they don't exist
for d in [DATA_PROCESSED, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Scope ─────────────────────────────────────────────────────────────────
# The full Favorita dataset is ~125M rows. For a PoC demo, we filter down to a
# manageable subset. Adjust these to increase/decrease scope.

# Number of top-selling items to keep (out of ~4,000)
N_ITEMS = 50

# Number of stores to keep (out of 54)
N_STORES = 5

# Aggregation frequency: we roll up daily → weekly for demand forecasting.
# Weekly is more appropriate for a food-box company that orders weekly.
AGG_FREQ = "W-MON"  # ISO weeks starting Monday

# ── Forecasting ────────────────────────────────────────────────────────────────
# How many weeks ahead to forecast
FORECAST_HORIZON = 8

# Train/test split: hold out the last N weeks for backtesting
HOLDOUT_WEEKS = 8

# Quantiles for probabilistic forecasting:
#   q10 = low scenario (only 10% chance demand is below this)
#   q50 = median (best single-point estimate)
#   q90 = high scenario (only 10% chance demand exceeds this)
QUANTILES = [0.10, 0.50, 0.90]

# ── Inventory Optimisation ─────────────────────────────────────────────────────
# These simulate the cost structure of a food distributor.
# In a real project, you'd pull these from SAP or finance.

# Cost of one unit of waste (disposal + lost margin) — AUD
WASTE_COST_PER_UNIT = 3.50

# Cost of one unit of stockout (lost sale + customer churn risk) — AUD
# Typically stockout cost >> waste cost for subscription-based food boxes,
# because losing a customer has long-term revenue impact.
STOCKOUT_COST_PER_UNIT = 10.00

# Average procurement cost per unit — AUD
AVG_PROCUREMENT_COST = 4.00

# Supplier minimum order quantity (units)
DEFAULT_MOQ = 100

# Shelf life in days for perishable items
PERISHABLE_SHELF_LIFE_DAYS = 7
NON_PERISHABLE_SHELF_LIFE_DAYS = 90

# Warehouse capacity (total units across all SKUs for a single store)
# ~30,000 is tight enough to bind on peak weeks (holiday uplifts pushing demand
# to ~27,000+) without causing infeasibility on normal weeks (~18,300 avg)
WAREHOUSE_CAPACITY_PER_STORE = 500_000  # scaled up for real Favorita data (~691 avg units × 49 items = ~34k/week)

# Perishable waste cost multiplier: perishable overstock costs this many times
# more than non-perishable overstock, because of short shelf life / disposal.
# Used in the MIP objective to penalise over-ordering perishable items more heavily.
PERISHABLE_WASTE_COST_MULTIPLIER = 2.5

# Assumed lead time in days from supplier to warehouse
DEFAULT_LEAD_TIME_DAYS = 3

# ── Synthetic Data ─────────────────────────────────────────────────────────────
# Parameters for synthetic data generation (used by 00_generate_synthetic_data.py)
SYNTHETIC_N_STORES = 5
SYNTHETIC_N_ITEMS = 50
SYNTHETIC_START_DATE = "2015-01-01"
SYNTHETIC_END_DATE = "2017-08-15"
SYNTHETIC_PERISHABLE_FRACTION = 0.35  # 35% of items are perishable
