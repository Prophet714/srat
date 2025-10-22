"""
vectorcal.py – Compare SRAT spring-rate predictions to nozzle-vector calibration data
(Force × Moment-arm only version)

Key behaviour
-------------
• Accepts .xlsx or .csv; falls back gracefully.
• Detects radians vs degrees automatically for the feedback angle.
• Measured torque is **always** FSum × MArm.  TorqueSum column is ignored.
• Computes:
    – TorquePred_nominal  = (SRAT spring rate) × angle (deg)
    – TorquePred_fit      = best-fit scaled prediction
• Outputs:
    – vectorcal_results.csv / .xlsx   (row-by-row results)
    – vectorcal_summary.csv           (key statistics and metadata)
"""

from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _try_read_xlsx_then_csv(path: str, sheet_name: int | str | None) -> pd.DataFrame:
    """Try Excel first (with fallback engine), then CSV."""
    try:
        if sheet_name is None:
            try:
                return pd.read_excel(path, sheet_name=0)
            except Exception:
                return pd.read_excel(path)
        else:
            try:
                return pd.read_excel(path, sheet_name=sheet_name)
            except Exception:
                print(f"⚠ Could not open sheet {sheet_name!r}; trying first sheet.")
                try:
                    return pd.read_excel(path, sheet_name=0)
                except Exception:
                    return pd.read_excel(path)
    except Exception:
        return pd.read_csv(path)

def _norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Case/space-insensitive map from logical name → actual column name."""
    canon = {c.lower().replace(" ", "").replace("\t", ""): c for c in df.columns}

    def pick(candidates: Iterable[str]) -> str | None:
        for key in candidates:
            c = canon.get(key.lower().replace(" ", ""))
            if c is not None:
                return c
        return None

    return {
        "angle":     pick(["fdbkdegsum","fdbkdeg","pfdbkdeg+yfdbkdeg","angledeg","angle"]),
        "force_sum": pick(["fsum","forcesum","f"]),
        "marm":      pick(["marm","momentarm","arm","marminch","marm_inch"]),
    }

def _coerce_num(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _maybe_radians_to_degrees(series: pd.Series) -> Tuple[pd.Series, str]:
    """If max angle is between ~0.2 and ~6.5, assume radians → convert to degrees."""
    s = series.dropna()
    if s.empty:
        return series, "unknown"
    mx = float(s.max())
    if 0.2 < mx <= 6.5:
        return np.degrees(series), "radians→degrees"
    return series, "degrees"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_vectorcal(
    *,
    vectorcal_file: str,
    srat_spring_rate: float | None,
    outdir: str,
    sheet_name: int | str | None = None,
    fit_scale: bool = True,
) -> None:
    """
    Parameters
    ----------
    vectorcal_file : str
        Path to .xlsx or .csv with vector-cal data.
    srat_spring_rate : float | None
        SRAT mean spring rate (lbf-in/deg).  If None, we fit slope directly.
    outdir : str
        Folder for vectorcal_results.{csv,xlsx} and vectorcal_summary.csv.
    sheet_name : int | str | None, optional
        Excel sheet name/index. Ignored for CSV. Defaults to first sheet.
    fit_scale : bool
        If True, also fit a scale factor k for TorquePred.
    """
    df = _try_read_xlsx_then_csv(vectorcal_file, sheet_name)
    if df.empty:
        raise ValueError(f"No rows read from {vectorcal_file!r}")

    colmap = _norm_cols(df)
    angle_col = colmap["angle"]
    fsum_col  = colmap["force_sum"]
    marm_col  = colmap["marm"]

    if angle_col is None:
        raise KeyError("Need a feedback angle column (e.g. FdbkDegSum or AngleDeg).")
    if fsum_col is None or marm_col is None:
        raise KeyError("Need both FSum and MArm columns for torque = FSum × MArm.")

    _coerce_num(df, [angle_col, fsum_col, marm_col])
    df = df.dropna(subset=[angle_col, fsum_col, marm_col]).copy()
    if df.empty:
        raise ValueError("Required columns contain no valid numeric rows.")

    # Ensure angle is degrees
    df["_angle_deg"], angle_units = _maybe_radians_to_degrees(df[angle_col])
    print("ℹ Detected radians; converted to degrees." if angle_units == "radians→degrees"
          else "ℹ Treating angle column as degrees.")

    # Measured torque = FSum × MArm  (Path-B only)
    df["_torque_meas"] = df[fsum_col] * df[marm_col]
    used_meas = f"{fsum_col} × {marm_col}"

    # Nominal prediction using SRAT spring rate
    if srat_spring_rate is not None and not math.isnan(srat_spring_rate):
        df["_torque_pred_nominal"] = float(srat_spring_rate) * df["_angle_deg"]
    else:
        df["_torque_pred_nominal"] = np.nan
        print("⚠ No SRAT spring rate provided; nominal prediction left NaN.")

    # Optionally fit a global scale factor k
    k_hat = np.nan
    if fit_scale:
        x = df["_angle_deg"].to_numpy(dtype=float)
        y = df["_torque_meas"].to_numpy(dtype=float)
        if srat_spring_rate and srat_spring_rate != 0:
            if np.nansum(x**2) > 0:
                k_hat = np.nansum((srat_spring_rate * x) * y) / np.nansum((srat_spring_rate * x) ** 2)
            df["_torque_pred_fit"] = (k_hat if np.isfinite(k_hat) else 1.0) * (srat_spring_rate * df["_angle_deg"])
        else:
            if np.nansum(x**2) > 0:
                k_hat = np.nansum(x * y) / np.nansum(x**2)
            df["_torque_pred_fit"] = k_hat * df["_angle_deg"]
    else:
        df["_torque_pred_fit"] = df["_torque_pred_nominal"]

    # Simple metrics
    def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        a = pd.to_numeric(y_true, errors="coerce")
        b = pd.to_numeric(y_pred, errors="coerce")
        mask = a.notna() & b.notna()
        if mask.sum() < 2:
            return {k: np.nan for k in ("r2","mae","rmse","bias","slope","intercept")}
        a = a[mask].to_numpy()
        b = b[mask].to_numpy()
        r = np.corrcoef(a, b)[0, 1]
        r2 = float(r * r)
        mae = float(np.mean(np.abs(b - a)))
        rmse = float(np.sqrt(np.mean((b - a) ** 2)))
        bias = float(np.mean(b - a))
        slope, intercept = np.polyfit(b, a, 1)
        return {"r2": r2, "mae": mae, "rmse": rmse,
                "bias": bias, "slope": float(slope), "intercept": float(intercept)}

    stats_nom = _metrics(df["_torque_meas"], df["_torque_pred_nominal"])
    stats_fit = _metrics(df["_torque_meas"], df["_torque_pred_fit"])

    # Output tidy DataFrame
    out = df.copy()
    out.rename(columns={
        angle_col: "FdbkDegSum_raw",
        fsum_col:  "FSum_raw",
        marm_col:  "MArm_raw",
    }, inplace=True)
    out = out[[
        "FdbkDegSum_raw", "_angle_deg",
        "FSum_raw", "MArm_raw",
        "_torque_meas", "_torque_pred_nominal", "_torque_pred_fit"
    ]]
    out.rename(columns={
        "_angle_deg": "AngleDeg",
        "_torque_meas": "TorqueMeas",
        "_torque_pred_nominal": "TorquePred_nominal",
        "_torque_pred_fit": "TorquePred_fit",
    }, inplace=True)

    meta_rows = [
        {"key": "measured_torque_source", "value": used_meas},
        {"key": "angle_units_in_file", "value": angle_units},
        {"key": "srat_spring_rate_lbf_in_per_deg", "value": srat_spring_rate},
        {"key": "k_hat_scale_factor", "value": k_hat},
        {"key": "nominal_r2", "value": stats_nom["r2"]},
        {"key": "nominal_mae", "value": stats_nom["mae"]},
        {"key": "nominal_rmse", "value": stats_nom["rmse"]},
        {"key": "nominal_bias", "value": stats_nom["bias"]},
        {"key": "fit_r2", "value": stats_fit["r2"]},
        {"key": "fit_mae", "value": stats_fit["mae"]},
        {"key": "fit_rmse", "value": stats_fit["rmse"]},
        {"key": "fit_bias", "value": stats_fit["bias"]},
        {"key": "fit_reg_slope(meas_vs_pred)", "value": stats_fit["slope"]},
        {"key": "fit_reg_intercept(meas_vs_pred)", "value": stats_fit["intercept"]},
    ]
    meta = pd.DataFrame(meta_rows)

    os.makedirs(outdir, exist_ok=True)
    out_csv  = os.path.join(outdir, "vectorcal_results.csv")
    out_xlsx = os.path.join(outdir, "vectorcal_results.xlsx")
    meta_csv = os.path.join(outdir, "vectorcal_summary.csv")

    out.to_csv(out_csv, index=False)
    meta.to_csv(meta_csv, index=False)
    try:
        with pd.ExcelWriter(out_xlsx) as xw:
            out.to_excel(xw, index=False, sheet_name="results")
            meta.to_excel(xw, index=False, sheet_name="summary")
    except Exception:
        pass

    # Console summary
    print(f"   VectorCal measured torque source: {used_meas}")
    if srat_spring_rate is not None:
        print(f"   SRAT spring rate (lbf-in/deg): {srat_spring_rate:.3f}")
    if np.isfinite(k_hat):
        print(f"   Fitted scale k: {k_hat:.4f}  →  effective spring = {k_hat * (srat_spring_rate or 1):.3f} lbf-in/deg")
    print(f"   Nominal R²: {stats_nom['r2']:.3f} | MAE: {stats_nom['mae']:.2f} | RMSE: {stats_nom['rmse']:.2f} | Bias: {stats_nom['bias']:.2f}")
    print(f"   Fitted  R²: {stats_fit['r2']:.3f} | MAE: {stats_fit['mae']:.2f} | RMSE: {stats_fit['rmse']:.2f} | Bias: {stats_fit['bias']:.2f}")
    print(f"   Wrote VectorCal CSV : {out_csv}")
    print(f"   Wrote VectorCal XLSX: {out_xlsx}")
    print(f"   Wrote summary CSV   : {meta_csv}")
