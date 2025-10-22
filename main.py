"""
main.py – Master launcher for Flexseal SRAT pipeline

Phases:
    1. baseline – Physics baseline only
    2. fit      – Baseline + hierarchical regression
    3. srat-s   – Add SRAT-S surrogate
    4. srat-a   – Add SRAT-A anomaly evaluation (requires SRAT-S)
    5. full     – Run all phases

The CLI now prompts the user to select a mode interactively.
"""

# ---------------------------------------------------------------------------
# Clean out any stale byte-code before importing the package
# ---------------------------------------------------------------------------
import os, shutil
root = os.path.dirname(os.path.abspath(__file__))
for dirpath, dirnames, filenames in os.walk(root):
    if "__pycache__" in dirnames:
        shutil.rmtree(os.path.join(dirpath, "__pycache__"), ignore_errors=True)

# ---------------------------------------------------------------------------
# Debug: show which montecarlo module Python will actually load
# ---------------------------------------------------------------------------
import importlib.util, sys
spec = importlib.util.find_spec("flexseal.montecarlo")
if spec:
    print(f"[DEBUG] flexseal.montecarlo will be imported from: {spec.origin}")
else:
    print("[DEBUG] flexseal.montecarlo not found on current sys.path")
print("[DEBUG] sys.path search order:")
for p in sys.path:
    print("   ", p)
# ---------------------------------------------------------------------------

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

from flexseal.physics_model import compute_baseline
from flexseal.regression import fit_hierarchical_residual_model
from flexseal.plotting import make_summary_plots
from flexseal.export import write_results
from flexseal import shim_model
...


# Optional SRAT-S / SRAT-A / SRAT-G
try:
    from flexseal.srat_s import run_srat_s, SurrogateResult
    HAS_SRAT_S = True
except Exception:
    HAS_SRAT_S = False

try:
    from flexseal.srat_a import run_srat_a
    HAS_SRAT_A = True
except Exception:
    HAS_SRAT_A = False

try:
    from flexseal.srat_g import run_srat_g
    HAS_SRAT_G = True
except Exception:
    HAS_SRAT_G = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interactive_phase() -> str:
    """Prompt the user to choose which phases to run."""
    print("\nSelect SRAT run mode:\n")
    print(" 1) baseline – Physics baseline only")
    print(" 2) fit      – Baseline + hierarchical regression")
    print(" 3) srat-s   – Add SRAT-S surrogate")
    print(" 4) srat-a   – Add SRAT-A anomaly evaluation")
    print(" 5) full     – Run all phases\n")
    while True:
        choice = input("Enter a number (1–5): ").strip()
        mapping = {
            "1": "baseline",
            "2": "fit",
            "3": "srat-s",
            "4": "srat-a",
            "5": "full"
        }
        if choice in mapping:
            return mapping[choice]
        print("Invalid choice. Please enter 1–5.")


def parse_args() -> argparse.Namespace:
    """Only --config is needed; phase is chosen interactively."""
    ap = argparse.ArgumentParser(
        description="Flexseal Spring Rate Analysis Tool (SRAT)"
    )
    ap.add_argument(
        "--config",
        default="input_deck.csv",
        help="Path to the primary input deck (settings + data)."
    )
    return ap.parse_args()


def load_input_deck(path: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Read SETTINGS and DATA sections of the input deck."""
    settings, settings_lines, data_lines = {}, [], []
    with open(path, "r", encoding="utf-8") as f:
        in_data = False
        for line in f:
            if line.strip().startswith("#---- DATA"):
                in_data = True
                continue
            if not in_data:
                if line.strip() and not line.strip().startswith("#"):
                    settings_lines.append(line)
            else:
                if line.strip():
                    data_lines.append(line)

    s_df = pd.read_csv(pd.io.common.StringIO("".join(settings_lines)))
    for _, row in s_df.iterrows():
        key = str(row["key"]).strip()
        raw_val = str(row["value"]).split("#", 1)[0].strip()
        try:
            val = float(raw_val) if "." in raw_val or raw_val.isdigit() else raw_val
        except Exception:
            val = raw_val
        settings[key] = val

    for required in ("mean_radius_in", "loaded_area_in2", "perimeter_in"):
        if required not in settings:
            raise ValueError(f"Missing required geometry value '{required}' in SETTINGS")

    fcd_df = pd.read_csv(pd.io.common.StringIO("".join(data_lines)))
    settings["__config_path__"] = str(Path(path).resolve())
    return settings, fcd_df


def _ensure_run_folder(settings: dict) -> Tuple[str, str]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_out = settings.get("outdir", "results")
    run_folder = os.path.join(root_out, stamp)
    os.makedirs(run_folder, exist_ok=True)
    settings["current_run_folder"] = run_folder
    return run_folder, stamp


def _copy_input_deck(settings: dict, run_folder: str, stamp: str) -> None:
    deck_path = settings.get("__config_path__")
    if deck_path and os.path.isfile(deck_path):
        shutil.copy2(deck_path, os.path.join(run_folder, f"input_deck_{stamp}.csv"))

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Flexseal SRAT CLI."""
    args = parse_args()
    settings, fcd_df = load_input_deck(args.config)

    # ------------------------------------------------------------------
    # SRAT-X: Shim subtraction
    # ------------------------------------------------------------------
    from flexseal import shim_model

    shim_path = r"C:\Users\U317688\OneDrive - L3Harris - GCCHigh\Sentinel\flexseal\Data\Sentinel Flexseal Preliminary Design Tool_v1.9.xlsm"

    print("[SRAT-X] Reading shim parameters…")
    try:
        k_shim = shim_model.compute_shim_stiffness(shim_path, sheet_name="Design Sheet")
        print(f"[SRAT-X] Shim stiffness (K_shim): {k_shim:.2f} lbf-in/deg")
        settings["K_shim"] = k_shim

        if "spring_rate_meas" in fcd_df.columns:
            # Preserve original total stiffness
            fcd_df["K_meas_total"] = fcd_df["spring_rate_meas"]

            # Subtract shim contribution to isolate rubber
            fcd_df["K_rubber_meas"] = fcd_df["spring_rate_meas"] - k_shim
            fcd_df["spring_rate_meas"] = fcd_df["K_rubber_meas"]

            print("[SRAT-X] Subtracted shim stiffness; added K_meas_total and K_rubber_meas to dataframe.")
        else:
            print("[SRAT-X] ⚠ No spring_rate_meas column in input deck.")
    except Exception as e:
        print(f"[SRAT-X] ⚠ Shim subtraction failed: {e}")
        settings["K_shim"] = 0.0

    # ------------------------------------------------------------------
    # Interactive menu AFTER shim subtraction
    # ------------------------------------------------------------------
    phase = _interactive_phase()       # "baseline","fit","srat-s","srat-a","srat-g","full"

    # ------------------------------------------------------------------
    # Set up run folder
    # ------------------------------------------------------------------
    run_folder, stamp = _ensure_run_folder(settings)

    print(f"\n=== Flexseal (SRAT) run started at {stamp} ===")
    print(f"Results will be stored in: {run_folder}\n")

    # ------------------------------------------------------------------
    # Step 1 – Physics baseline (SRAT-C core physics)
    # ------------------------------------------------------------------
    print("Step 1: Computing physics baseline …")
    baseline_df = compute_baseline(fcd_df, settings)
    print(f"   Baseline rows: {len(baseline_df)}")

    if "residual" in baseline_df.columns:
        valid = baseline_df["residual"].notna().sum()
        print(f"   Residual values present: {valid}")
        mean_K_pred = baseline_df["K_pred"].mean()
        std_K_pred = baseline_df["K_pred"].std()
        print(f"   Mean predicted rotational spring rate: "
              f"{mean_K_pred:.2f} lbf-in/deg (SD: {std_K_pred:.2f})")

        if "K_meas_total" in baseline_df.columns and "K_rubber_meas" in baseline_df.columns:
            mean_total = baseline_df["K_meas_total"].mean()
            mean_rubber = baseline_df["K_rubber_meas"].mean()
            print(f"[SRAT-X] Mean total stiffness (measured): {mean_total:.2f}")
            print(f"[SRAT-X] Mean rubber-only stiffness:      {mean_rubber:.2f}")

        if "spring_rate_meas" in baseline_df.columns:
            mean_K_meas = baseline_df["spring_rate_meas"].mean()
            std_K_meas = baseline_df["spring_rate_meas"].std()
            print(f"   Mean measured rotational spring rate (post-shim): "
                  f"{mean_K_meas:.2f} lbf-in/deg (SD: {std_K_meas:.2f})\n")

    if phase == "baseline":
        print("Step 2: Creating plots …")
        plot_info = make_summary_plots(baseline_df, None, None, settings)
        print(f"   Saved plots: {list(plot_info.values())}\n")

        print("Step 3: Writing Excel results …")
        write_results(baseline_df, None, None, settings)
        _copy_input_deck(settings, run_folder, stamp)
        print("=== SRAT baseline run complete ===\n")
        return

    # ------------------------------------------------------------------
    # Step 2 – SRAT-C regression fit
    # ------------------------------------------------------------------
    print("Step 2: Fitting regression model …")
    fit_result = fit_hierarchical_residual_model(baseline_df, settings)
    if fit_result.model_obj is None:
        print("   ⚠ No residual data to model; skipping regression output.\n")
    else:
        print("   Regression complete.")
        print("   Coefficients:")
        print(fit_result.coef_table.to_string(index=False))
        print()

    if phase == "fit":
        print("Step 3: Creating plots …")
        plot_info = make_summary_plots(baseline_df, fit_result, None, settings)
        print(f"   Saved plots: {list(plot_info.values())}\n")

        print("Step 4: Writing Excel results …")
        write_results(baseline_df, fit_result, None, settings)
        _copy_input_deck(settings, run_folder, stamp)
        print("=== SRAT-C (fit) run complete ===\n")
        return

    # ------------------------------------------------------------------
    # Step 3 – SRAT-S surrogate
    # ------------------------------------------------------------------
    sres = None
    if phase in ("srat-s", "srat-a", "srat-g", "full"):
        if not HAS_SRAT_S:
            print("⚠ SRAT-S module not available. Install/import flexseal.srat_s.")
        else:
            print("Step 3: Running SRAT-S surrogate …")
            sres: SurrogateResult = run_srat_s(baseline_df, settings)
            print(f"   Trained on reference rows: {len(sres.train_index)}")
            print(f"   Predictors used: {', '.join(sres.predictors)}")
            print("   Surrogate calibration (reference): "
                  f"slope={sres.calib_ref['slope'].iloc[0]:.3f}, "
                  f"R²={sres.calib_ref['r2'].iloc[0]:.3f}, "
                  f"MAE={sres.calib_ref['mae'].iloc[0]:.3f}, "
                  f"RMSE={sres.calib_ref['rmse'].iloc[0]:.3f}")
            print("   Surrogate calibration (all rows): "
                  f"slope={sres.calib_all['slope'].iloc[0]:.3f}, "
                  f"R²={sres.calib_all['r2'].iloc[0]:.3f}, "
                  f"MAE={sres.calib_all['mae'].iloc[0]:.3f}, "
                  f"RMSE={sres.calib_all['rmse'].iloc[0]:.3f}")

    # ------------------------------------------------------------------
    # Step 4 – SRAT-A anomaly evaluation
    # ------------------------------------------------------------------
    if phase in ("srat-a", "full"):
        if not HAS_SRAT_A:
            print("⚠ SRAT-A module not available. Install/import flexseal.srat_a.")
        elif sres is None or sres.model_obj is None:
            print("⚠ SRAT-A skipped: SRAT-S surrogate not available.")
        else:
            print("Step 4: Running SRAT-A anomaly analysis …")
            run_srat_a(baseline_df, sres, settings)

    # ------------------------------------------------------------------
    # Step 5 – SRAT-G global sensitivity
    # ------------------------------------------------------------------
    if phase in ("srat-g", "full"):
        if not HAS_SRAT_G:
            print("⚠ SRAT-G module not available. Install/import flexseal.srat_g.")
        elif sres is None or sres.model_obj is None:
            print("⚠ SRAT-G skipped: SRAT-S surrogate not available.")
        else:
            print("Step 5: Running SRAT-G global sensitivity …")
            run_srat_g(baseline_df, settings, sres)

    # ------------------------------------------------------------------
    # Final – plots & Excel
    # ------------------------------------------------------------------
    step_num = "6" if phase in ("srat-a", "srat-g", "full") else "5"
    print(f"Step {step_num}: Creating plots …")
    plot_info = make_summary_plots(baseline_df, fit_result, None, settings)
    print(f"   Saved plots: {list(plot_info.values())}\n")

    print(f"Step {step_num + 'A'}: Writing Excel results …")
    write_results(baseline_df, fit_result, None, settings)
    _copy_input_deck(settings, run_folder, stamp)

    print("=== SRAT run complete ===\n")

    # ---------------------------------------------------------------
    # Optional VectorCal post-processing
    # ---------------------------------------------------------------
    resp = input("Run VectorCal on a nozzle-vector Excel file? [y/N] ").strip().lower()
    if resp == "y":
        vc_path = input("Enter full path to the Excel file: ").strip()

        # Safely pull mean spring rate from SRAT baseline output
        k_pred = None
        if "K_pred" in baseline_df.columns:
            k_pred_val = baseline_df["K_pred"].mean()
            if pd.notna(k_pred_val):
                k_pred = float(k_pred_val)

        if k_pred is None:
            print("⚠  No valid K_pred column found; VectorCal skipped.\n")
        else:
            from flexseal.vectorcal import run_vectorcal
            run_vectorcal(
                vectorcal_file=vc_path,
                srat_spring_rate=k_pred,
                outdir=run_folder   # reuse SRAT results folder
                # sheet_name intentionally omitted – run_vectorcal will
                # try the requested sheet if supplied or fall back to the first.
            )
            print("VectorCal complete.\n")
    else:
        print("VectorCal skipped.\n")

if __name__ == "__main__":
    main()
