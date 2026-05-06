"""
Run the full supply chain optimisation pipeline end-to-end.

Step 0 always runs the unit test suite first. If any test fails the pipeline
aborts — this guards against regressions before touching any data.

Usage:
    python run_pipeline.py              # full run
    python run_pipeline.py --skip-tests # bypass tests (not recommended)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(label: str, *args: str) -> None:
    """Run a subprocess step and abort the pipeline on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run([PYTHON, *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: '{label}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def main() -> None:
    skip_tests = "--skip-tests" in sys.argv

    # ── Step 0: Unit tests ────────────────────────────────────────────────────
    # Tests MUST pass before any data is read or written. This prevents a broken
    # business rule or MIP constraint from silently corrupting downstream outputs.
    if not skip_tests:
        run(
            "Step 0: Unit tests (pytest)",
            "-m", "pytest", "tests/", "-v", "--tb=short",
        )
    else:
        print("\n[WARNING] Skipping unit tests (--skip-tests flag set)")

    # ── Pipeline steps 01–07 ──────────────────────────────────────────────────
    steps = [
        ("Step 1: Data preparation",        "src/01_data_prep.py"),
        ("Step 2: EDA",                     "src/02_eda.py"),
        ("Step 3: Feature engineering",     "src/03_feature_engineering.py"),
        ("Step 4: Demand forecasting",      "src/04_demand_forecasting.py"),
        ("Step 5: Business rules",          "src/05_business_rules.py"),
        ("Step 6: Inventory optimisation",  "src/06_inventory_optimisation.py"),
        ("Step 7: Evaluation",              "src/07_evaluation.py"),
    ]

    for label, script in steps:
        run(label, script)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print("  Launch dashboard: streamlit run src/08_dashboard.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
