"""
Run the full supply chain optimisation pipeline end-to-end.

Step 0 always runs the unit test suite first. If any test fails the pipeline
aborts — this guards against regressions before touching any data.

Usage:
    python run_pipeline.py                  # full run (tests + steps 1-7 + dashboard)
    python run_pipeline.py --from-step 4    # resume from step 4 (skip 1-3)
    python run_pipeline.py --skip-tests     # bypass pytest (not recommended)
    python run_pipeline.py --no-dashboard   # skip launching Streamlit at the end

Flags can be combined:
    python run_pipeline.py --from-step 6 --skip-tests
"""

import subprocess
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ── Step registry ─────────────────────────────────────────────────────────────
# Each entry: (step_number, human_label, script_path_or_None)
# step_number 0 = pytest, steps 1-7 = pipeline, step 8 = Streamlit dashboard
STEPS = [
    (1, "Step 1: Data preparation",       "src/01_data_prep.py"),
    (2, "Step 2: EDA",                    "src/02_eda.py"),
    (3, "Step 3: Feature engineering",    "src/03_feature_engineering.py"),
    (4, "Step 4: Demand forecasting",     "src/04_demand_forecasting.py"),
    (5, "Step 5: Business rules",         "src/05_business_rules.py"),
    (6, "Step 6: Inventory optimisation", "src/06_inventory_optimisation.py"),
    (7, "Step 7: Evaluation",             "src/07_evaluation.py"),
]


def run(label: str, *args: str) -> None:
    """Run a subprocess step and abort the pipeline on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run([PYTHON, *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: '{label}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def parse_args() -> dict:
    args = sys.argv[1:]
    opts = {
        "skip_tests": False,
        "no_dashboard": False,
        "from_step": 1,
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--skip-tests":
            opts["skip_tests"] = True
        elif a == "--no-dashboard":
            opts["no_dashboard"] = True
        elif a == "--from-step":
            i += 1
            if i >= len(args):
                print("ERROR: --from-step requires a number (e.g. --from-step 4)")
                sys.exit(1)
            try:
                opts["from_step"] = int(args[i])
            except ValueError:
                print(f"ERROR: --from-step value must be an integer, got '{args[i]}'")
                sys.exit(1)
        elif a.startswith("--from-step="):
            try:
                opts["from_step"] = int(a.split("=", 1)[1])
            except ValueError:
                print(f"ERROR: --from-step value must be an integer, got '{a}'")
                sys.exit(1)
        else:
            print(f"WARNING: Unknown argument '{a}' (ignored)")
        i += 1
    return opts


def main() -> None:
    opts = parse_args()
    from_step = opts["from_step"]

    if from_step < 1 or from_step > 7:
        print(f"ERROR: --from-step must be between 1 and 7, got {from_step}")
        sys.exit(1)

    if from_step > 1:
        print(f"\n[INFO] Resuming from step {from_step} — steps 1-{from_step - 1} skipped.")

    # ── Step 0: Unit tests ────────────────────────────────────────────────────
    # Tests gate the pipeline. Skip only with --skip-tests (or when resuming
    # mid-pipeline where upstream data is already known-good).
    if not opts["skip_tests"] and from_step == 1:
        run(
            "Step 0: Unit tests (pytest)",
            "-m", "pytest", "tests/", "-v", "--tb=short",
        )
    elif opts["skip_tests"]:
        print("\n[WARNING] Skipping unit tests (--skip-tests flag set)")
    else:
        print(f"\n[INFO] Skipping tests (resuming from step {from_step})")

    # ── Pipeline steps 1–7 ───────────────────────────────────────────────────
    for step_num, label, script in STEPS:
        if step_num < from_step:
            continue
        run(label, script)

    # ── Step 8: Dashboard ─────────────────────────────────────────────────────
    # Launch Streamlit in a background process so the pipeline script exits
    # cleanly. The dashboard will keep running until the user closes the terminal.
    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}\n")

    if not opts["no_dashboard"]:
        print("Launching dashboard at http://localhost:8501 ...")
        print("(Run with --no-dashboard to skip this step)\n")
        # Open the browser after a short delay so Streamlit has time to start
        subprocess.Popen(
            [PYTHON, "-m", "streamlit", "run", "src/08_dashboard.py",
             "--server.headless", "false"],
            cwd=PROJECT_ROOT,
        )
    else:
        print("Dashboard skipped (--no-dashboard). To launch manually:")
        print("  streamlit run src/08_dashboard.py\n")


if __name__ == "__main__":
    main()
