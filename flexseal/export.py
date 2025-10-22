import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional


# ======================================================================
#  SMALL UTILS
# ======================================================================

def _safe_mean_std(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if series is None or series.empty:
        return (None, None)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (None, None)
    return (float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0)


def _mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x))) if x.size else float("nan")


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


# ======================================================================
#  DIAGNOSTIC COMPUTATION
# ======================================================================

def _compute_diagnostics(baseline_df: pd.DataFrame, fit_result: Any) -> pd.DataFrame:
    """Compute single-row diagnostic summary for SRAT-C baseline and regression."""
    out: Dict[str, Any] = {}

    if "spring_rate_meas" not in baseline_df.columns or "K_pred" not in baseline_df.columns:
        return pd.DataFrame([{"note": "Diagnostics skipped – missing spring_rate_meas or K_pred"}])

    valid = baseline_df.dropna(subset=["spring_rate_meas", "K_pred", "residual"])
    if valid.empty:
        return pd.DataFrame([{"note": "Diagnostics skipped – no valid residuals"}])

    K_meas = valid["spring_rate_meas"].astype(float).values
    K_pred = valid["K_pred"].astype(float).values
    residual = valid["residual"].astype(float).values

    # Basic model fit metrics
    try:
        slope, intercept = np.polyfit(K_pred, K_meas, 1)
        r2 = float(np.corrcoef(K_pred, K_meas)[0, 1] ** 2)
    except Exception:
        slope, intercept, r2 = (float("nan"), float("nan"), float("nan"))
    out["slope_meas_vs_pred"] = slope
    out["intercept_meas_vs_pred"] = intercept
    out["r2_meas_vs_pred"] = r2

    # Residual stats
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    out["median_residual"] = med
    out["mad_residual"] = mad
    out["pct_outliers_gt3mad"] = (
        0.0 if mad == 0 else 100.0 * float(np.mean(np.abs(residual - med) > 3 * mad))
    )

    # Surrogate health check (if S exists)
    if "S" in valid.columns:
        S = pd.to_numeric(valid["S"], errors="coerce").dropna()
        out["pct_S_outside_0.2_5"] = 100.0 * float(((S < 0.2) | (S > 5)).mean()) if not S.empty else None
    else:
        out["pct_S_outside_0.2_5"] = None

    # Top variance contributor (if present)
    if hasattr(fit_result, "var_decomp") and isinstance(getattr(fit_result, "var_decomp"), pd.DataFrame) and not fit_result.var_decomp.empty:
        top = fit_result.var_decomp.sort_values("variance_share", ascending=False).iloc[0]
        out["top_partial_R2_factor"] = top.get("factor", None)
        out["top_partial_R2_pct"] = _num(top.get("variance_share", None))
    else:
        out["top_partial_R2_factor"] = None
        out["top_partial_R2_pct"] = None

    return pd.DataFrame([out])


# ======================================================================
#  OPTIONAL: PULL SRAT-S CALIBRATION FROM FILE IF AVAILABLE
# ======================================================================

def _try_read_surrogate_calibration(run_folder: str) -> Dict[str, Optional[float]]:
    """
    Try to parse SRAT-S calibration metrics from surrogate_results.csv,
    which can have varying column structures. We attempt flexible extraction.
    Returns keys (may be None if not found):
      - sur_cal_ref_slope, sur_cal_ref_r2, sur_cal_ref_mae, sur_cal_ref_rmse
      - sur_cal_all_slope, sur_cal_all_r2, sur_cal_all_mae, sur_cal_all_rmse
    """
    out = {
        "sur_cal_ref_slope": None, "sur_cal_ref_r2": None, "sur_cal_ref_mae": None, "sur_cal_ref_rmse": None,
        "sur_cal_all_slope": None, "sur_cal_all_r2": None, "sur_cal_all_mae": None, "sur_cal_all_rmse": None,
    }
    fpath = os.path.join(run_folder, "surrogate_results.csv")
    if not os.path.isfile(fpath):
        return out

    try:
        df = pd.read_csv(fpath)
    except Exception:
        return out

    # We expect either tidy columns like ["section","slope","r2","mae","rmse", ...]
    # or wide lines where numeric metrics are tucked into later columns.
    def _extract(row) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        # direct named columns
        s = _num(row.get("slope", None))
        r = _num(row.get("r2", None))
        m = _num(row.get("mae", None))
        rm = _num(row.get("rmse", None))
        if s is not None or r is not None or m is not None or rm is not None:
            return (s, r, m, rm)
        # fallback: scan numeric values in row in order and take first 4 as (slope, r2, mae, rmse)
        try:
            vals = [ _num(v) for v in row.tolist() if _num(v) is not None ]
            # be a little selective: we want smallish MAE/RMSE and |slope| not crazy
            if len(vals) >= 4:
                return (vals[0], vals[1], vals[2], vals[3])
        except Exception:
            pass
        return (None, None, None, None)

    # Find rows labeled calibration_ref / calibration_all in a generic 'section' or first col
    section_col = "section" if "section" in df.columns else df.columns[0]
    for which, prefix in [("calibration_ref", "sur_cal_ref_"), ("calibration_all", "sur_cal_all_")]:
        try:
            rows = df[df[section_col].astype(str).str.contains(which, na=False)]
            if not rows.empty:
                s, r, m, rm = _extract(rows.iloc[0])
                out[prefix + "slope"] = s
                out[prefix + "r2"] = r
                out[prefix + "mae"] = m
                out[prefix + "rmse"] = rm
        except Exception:
            # ignore and leave as None
            pass

    return out


# ======================================================================
#  ANOMALY SUMMARY (SRAT-A)
# ======================================================================

def _summarize_anomalies(baseline_df: pd.DataFrame, run_folder: str) -> Tuple[int, Optional[str]]:
    """
    Returns (count, ids_string)
      - count: number of anomaly rows (bool column 'is_anomaly_gt4MAD' if present,
               else try SRAT-A report; else 0)
      - ids_string: comma-separated 'lot-roll' for up to first 5 anomalies if lot/roll exist
    """
    ids: List[str] = []

    if "is_anomaly_gt4MAD" in baseline_df.columns:
        mask = baseline_df["is_anomaly_gt4MAD"].astype(bool)
        cnt = int(mask.sum())
        if cnt > 0 and {"lot", "roll"}.issubset(baseline_df.columns):
            sub = baseline_df.loc[mask, ["lot", "roll"]].astype(str).head(5)
            ids = (sub["lot"] + "-" + sub["roll"]).tolist()
        return (cnt, ", ".join(ids) if ids else None)

    # Try reading anomaly_report.csv as fallback
    fpath = os.path.join(run_folder, "anomaly_report.csv")
    if os.path.isfile(fpath):
        try:
            adf = pd.read_csv(fpath)
            # Heuristic: look for a boolean-ish column that marks anomalies
            flag_cols = [c for c in adf.columns if "anomaly" in c.lower() or "flag" in c.lower()]
            cnt = 0
            if flag_cols:
                flags = pd.to_numeric(adf[flag_cols[0]], errors="coerce")
                if flags.notna().any():
                    cnt = int((flags != 0).sum())
                else:
                    # treat strings like "True"/"False"
                    cnt = int(adf[flag_cols[0]].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            else:
                cnt = 0
            return (cnt, None)
        except Exception:
            return (0, None)

    return (0, None)


# ======================================================================
#  PLOTTING
# ======================================================================

def _make_summary_plots(baseline_df: pd.DataFrame, fit_result: Any, mc_out: Any, outdir: str) -> list[str]:
    """Generate key summary plots and return list of saved file paths."""
    plot_paths: list[str] = []

    if mc_out and hasattr(mc_out, "samples") and "K_sim" in mc_out.samples:
        ax = mc_out.samples["K_sim"].plot(kind="hist", bins=40, alpha=0.7)
        ax.set_title("Simulated Spring Rate Distribution")
        ax.set_xlabel("K_sim")
        fig = ax.get_figure()
        fname = os.path.join(outdir, "mc_hist.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        plot_paths.append(fname)

    return plot_paths


# ======================================================================
#  BUILD COMPACT REVIEW ROW (for o3-mini)
# ======================================================================

def _build_review_row(baseline_df: pd.DataFrame,
                      diag: pd.DataFrame,
                      settings: dict,
                      run_folder: str) -> Dict[str, Any]:
    """
    Compose a single flat dict of compact, numeric metrics for o3-mini.
    Gracefully degrades (uses NaN/None) if pieces are missing.
    """
    flat: Dict[str, Any] = {}

    # start with what diagnostics computed
    for k, v in diag.iloc[0].items():
        flat[k] = v

    # core aggregates (if present)
    if {"spring_rate_meas", "K_pred", "residual"}.issubset(baseline_df.columns):
        valid = baseline_df.dropna(subset=["spring_rate_meas", "K_pred", "residual"])
        flat["n_valid_rows"] = int(len(valid))

        k_pred_mean, k_pred_sd = _safe_mean_std(valid["K_pred"])
        k_meas_mean, k_meas_sd = _safe_mean_std(valid["spring_rate_meas"])
        flat["k_pred_mean"] = k_pred_mean
        flat["k_pred_sd"] = k_pred_sd
        flat["k_meas_mean"] = k_meas_mean
        flat["k_meas_sd"] = k_meas_sd

        resid = pd.to_numeric(valid["residual"], errors="coerce").dropna().values
        flat["residual_mae"] = _mae(resid) if resid.size else None
        flat["residual_rmse"] = _rmse(resid) if resid.size else None
    else:
        flat["n_valid_rows"] = 0
        flat["k_pred_mean"] = None
        flat["k_pred_sd"] = None
        flat["k_meas_mean"] = None
        flat["k_meas_sd"] = None
        flat["residual_mae"] = None
        flat["residual_rmse"] = None

    # anomalies (SRAT-A)
    cnt, ids = _summarize_anomalies(baseline_df, run_folder)
    flat["anomaly_rows_count"] = cnt
    if ids:
        flat["anomaly_ids_preview"] = ids  # short, comma-separated string of first few

    # surrogate calibration (SRAT-S) – optional
    flat.update(_try_read_surrogate_calibration(run_folder))

    # echo a few safe settings (optional, numeric only)
    for key in ["K_parasitic_mean", "surrogate_anomaly_k_mad", "mean_radius_in", "loaded_area_in2"]:
        if key in settings:
            flat[key] = _num(settings.get(key))

    return flat


# ======================================================================
#  MASTER EXPORT PIPELINE
# ======================================================================

def write_results(
    baseline_df: pd.DataFrame,
    fit_result: Any,
    mc_out: Any,
    settings: dict
) -> None:
    """
    Full export sequence:
    - Writes Excel workbook (baseline, coefficients, diagnostics, MC data)
    - Writes diagnostic summary CSV
    - Writes compact review CSV for GPT-mini (expanded)
    - Copies input deck
    - Writes consolidated TXT report
    - Generates summary plots
    """

    root_results = r"C:\Users\U317688\OneDrive - L3Harris - GCCHigh\Sentinel\results"

    run_folder = settings.get("current_run_folder")
    if not run_folder:
        run_folder = os.path.join(root_results, "manual_run")
    else:
        run_folder = os.path.join(root_results, os.path.basename(run_folder))

    os.makedirs(run_folder, exist_ok=True)

    out_xlsx = os.path.join(run_folder, "flexseal_results.xlsx")
    out_csv = os.path.join(run_folder, "diagnostics_summary.csv")
    review_csv = os.path.join(run_folder, "srat_review.csv")

    # ------------------------------------------------------------------
    # Excel Workbook
    # ------------------------------------------------------------------
    with pd.ExcelWriter(out_xlsx) as xw:
        baseline_df.to_excel(xw, sheet_name="baseline", index=False)

        if fit_result:
            if hasattr(fit_result, "coef_table") and isinstance(fit_result.coef_table, pd.DataFrame):
                fit_result.coef_table.to_excel(xw, sheet_name="coefficients", index=False)
            if hasattr(fit_result, "var_decomp") and isinstance(fit_result.var_decomp, pd.DataFrame):
                fit_result.var_decomp.to_excel(xw, sheet_name="variance_decomp", index=False)

        diag = _compute_diagnostics(baseline_df, fit_result)
        diag.to_excel(xw, sheet_name="diagnostics", index=False)

        if mc_out:
            if hasattr(mc_out, "samples") and isinstance(mc_out.samples, pd.DataFrame):
                mc_out.samples.to_excel(xw, sheet_name="mc_samples", index=False)
            if hasattr(mc_out, "summary") and isinstance(mc_out.summary, pd.DataFrame):
                mc_out.summary.to_excel(xw, sheet_name="mc_summary", index=False)
            if hasattr(mc_out, "sensitivity") and isinstance(mc_out.sensitivity, pd.DataFrame):
                mc_out.sensitivity.to_excel(xw, sheet_name="sensitivity", index=False)

    # Save diagnostics summary
    diag.to_csv(out_csv, index=False)

    # ------------------------------------------------------------------
    # Compact Review CSV for GPT-mini (expanded with context)
    # ------------------------------------------------------------------
    flat_row = _build_review_row(baseline_df, diag, settings, run_folder)
    # write as key-value two-column CSV
    flat_df = pd.DataFrame({"metric": list(flat_row.keys()), "value": list(flat_row.values())})
    flat_df.to_csv(review_csv, index=False)

    # ------------------------------------------------------------------
    # Copy Input Deck (for traceability)
    # ------------------------------------------------------------------
    deck_path = settings.get("__config_path__")
    if deck_path and os.path.isfile(deck_path):
        shutil.copy2(deck_path, os.path.join(run_folder, os.path.basename(deck_path)))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_paths = _make_summary_plots(baseline_df, fit_result, mc_out, run_folder)

    # ------------------------------------------------------------------
    # Master TXT Report
    # ------------------------------------------------------------------
    master_txt = os.path.join(run_folder, "SRAT_master_report.txt")
    with open(master_txt, "w", encoding="utf-8") as out:

        def section(title: str):
            out.write("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80 + "\n")

        section("Input Deck")
        if deck_path and os.path.isfile(deck_path):
            with open(deck_path, encoding="utf-8") as f:
                out.write(f.read())

        section("Diagnostics Summary")
        with open(out_csv, encoding="utf-8") as f:
            out.write(f.read())

        section("Compact Review (for GPT-mini)")
        with open(review_csv, encoding="utf-8") as f:
            out.write(f.read())

        for title, fname in [
            ("SRAT-S Surrogate Results", "surrogate_results.csv"),
            ("SRAT-A Anomaly Report", "anomaly_report.csv"),
            ("SRAT-G Governance Report", "governance_report.csv"),
        ]:
            fpath = os.path.join(run_folder, fname)
            if os.path.isfile(fpath):
                section(title)
                with open(fpath, encoding="utf-8") as f:
                    out.write(f.read())

        if plot_paths:
            section("Generated Plots")
            for p in plot_paths:
                out.write(f"{os.path.basename(p)}\n")

    # ------------------------------------------------------------------
    # Console Output
    # ------------------------------------------------------------------
    print(f"   Wrote Excel results with diagnostics to: {out_xlsx}")
    print(f"   Wrote one-row diagnostics CSV to:        {out_csv}")
    print(f"   Wrote compact review CSV (for GPT-mini): {review_csv}")
    print(f"   Wrote consolidated master TXT to:        {master_txt}")
    if plot_paths:
        print(f"   Saved plots: {', '.join(os.path.basename(p) for p in plot_paths)}")
    print(f"   Copied input deck to:                    {run_folder}")
