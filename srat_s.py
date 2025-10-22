"""
srat_s.py – SRAT-S surrogate model for residuals r = K_meas – K_pred

* Fits a statistical surrogate on residuals using process variables.
* Trains on a robust reference subset.
* Writes **one combined CSV** (and optional XLSX) with every SRAT-S result.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

# ---------------------------------------------------------------------------
# Data class holding model + all outputs
# ---------------------------------------------------------------------------
@dataclass
class SurrogateResult:
    model_obj: Optional[Any]
    train_index: pd.Index
    predictors: List[str]
    coef_table: pd.DataFrame
    var_decomp: pd.DataFrame
    calib_ref: pd.DataFrame
    calib_all: pd.DataFrame
    predictions: pd.DataFrame
    flags: pd.DataFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def _fit_ols(formula: str, data: pd.DataFrame):
    fit = smf.ols(formula, data=data).fit()
    ci = fit.conf_int(alpha=0.05)
    coef_df = pd.DataFrame({
        "name": fit.params.index,
        "estimate": fit.params.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
        "p_value": fit.pvalues.values
    })
    return fit, coef_df

def _partial_r2_table(response: str, predictors: List[str], data: pd.DataFrame) -> pd.DataFrame:
    if not predictors:
        return pd.DataFrame(columns=["factor", "variance_share"])
    full = f"{response} ~ {' + '.join(predictors)}"
    base_r2 = smf.ols(full, data=data).fit().rsquared
    rows = []
    for p in predictors:
        reduced = [x for x in predictors if x != p]
        red = f"{response} ~ {' + '.join(reduced)}" if reduced else f"{response} ~ 1"
        r2_red = smf.ols(red, data=data).fit().rsquared
        rows.append({"factor": p, "variance_share": max(base_r2 - r2_red, 0.0) * 100})
    return pd.DataFrame(rows).sort_values("variance_share", ascending=False).reset_index(drop=True)

def _calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "mae": np.nan, "rmse": np.nan}
    b, a = np.polyfit(y_pred, y_true, 1)
    r2 = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2) if len(y_true) > 1 else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"slope": b, "intercept": a, "r2": r2, "mae": mae, "rmse": rmse}

def _pick_predictors(df: pd.DataFrame) -> List[str]:
    candidates = [
        "shear_mod50_psi", "hardnessA", "thickness_in",
        "age_days", "cure_torque", "t90_min", "S"
    ]
    return [c for c in candidates if c in df.columns]

def _reference_mask(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.Series:
    n = len(df)
    mask = pd.Series(False, index=df.index)

    ref_col = settings.get("reference_col")
    if ref_col and ref_col in df.columns:
        col = df[ref_col]
        if col.dropna().isin([True, False]).all():
            return col.astype(bool).reindex(df.index, fill_value=False)

    ref_lots = settings.get("reference_lots")
    if ref_lots:
        lots_set = {s.strip() for s in str(ref_lots).replace(",", " ").split() if s.strip()}
        if "lot" in df.columns and lots_set:
            return df["lot"].astype(str).isin(lots_set)

    res = pd.to_numeric(df.get("residual", pd.Series(index=df.index, dtype=float)), errors="coerce")
    med = float(np.nanmedian(res))
    mad = _mad(res[res.notna()].values) if res.notna().any() else np.nan
    k = float(settings.get("ref_k", 2.5))
    if not np.isnan(mad) and mad > 0:
        mask = (np.abs(res - med) <= k * mad)
    else:
        mask[:] = True
    return mask

def _anomaly_flags(err: np.ndarray, k_mad: float) -> Tuple[np.ndarray, float, float]:
    """
    Flag anomalies where |error - median| > k_mad * MAD.
    k_mad is passed in from settings (default 4.0).
    """
    med = float(np.median(err))
    mad = float(np.median(np.abs(err - med)))
    if mad == 0:
        flags = np.zeros_like(err, dtype=bool)
    else:
        flags = np.abs(err - med) > (k_mad * mad)
    return flags, med, mad

# ---------------------------------------------------------------------------
# Main SRAT-S routine
# ---------------------------------------------------------------------------
def run_srat_s(baseline_df: pd.DataFrame, settings: Dict[str, Any], outdir: Optional[str] = None) -> SurrogateResult:
    """
    Train the SRAT-S surrogate and write a **single combined CSV/XLSX**.

    outdir : optional path to write surrogate_results.csv (defaults to settings["current_run_folder"])
    """
    if "residual" not in baseline_df.columns:
        raise KeyError("SRAT-S requires 'residual' column from SRAT-C baseline.")

    df = _coerce_numeric(baseline_df.copy(),
        ["residual", "shear_mod50_psi", "hardnessA", "thickness_in",
         "age_days", "cure_torque", "t90_min", "S"])

    predictors = settings.get("predictors_override") or _pick_predictors(df)
    predictors = [p for p in predictors if p in df.columns]
    if not predictors:
        raise ValueError("No usable predictors found for SRAT-S.")

    ref_mask = _reference_mask(df, settings)
    train_df = df.loc[ref_mask].dropna(subset=["residual"] + predictors)
    if train_df.empty:
        raise ValueError("Reference subset for SRAT-S is empty after filtering.")

    response = "residual"
    formula = f"{response} ~ {' + '.join(predictors)}"

    fit, coef_df = _fit_ols(formula, train_df)
    var_decomp = _partial_r2_table(response, predictors, train_df)

    pred_train = fit.predict(train_df)
    for p in predictors:
        if p not in df.columns:
            df[p] = np.nan
    pred_all = fit.predict(df)

    calib_ref = pd.DataFrame([_calibration(train_df["residual"].values, pred_train.values)])
    calib_all = pd.DataFrame([_calibration(df["residual"].values, pred_all.values)])

    pred_df = pd.DataFrame({
        "index": df.index,
        "residual": df["residual"],
        "pred_surrogate": pred_all,
        "surrogate_error": df["residual"] - pred_all
    })

    # ---- anomaly detection with configurable k_mad (default 4.0) ----
    k_mad = float(settings.get("surrogate_anomaly_k_mad", 4.0))
    flags_bool, err_med, err_mad = _anomaly_flags(pred_df["surrogate_error"].values, k_mad)
    flags_df = pd.DataFrame({
        "index": df.index,
        "is_anomaly_gt{k}_MAD".format(k=int(k_mad)) if k_mad != 4.0 else "is_anomaly_gt4MAD": flags_bool,
        "err_median": err_med,
        "err_MAD": err_mad
    })

    result = SurrogateResult(
        model_obj=fit,
        train_index=train_df.index,
        predictors=predictors,
        coef_table=coef_df.reset_index(drop=True),
        var_decomp=var_decomp.reset_index(drop=True),
        calib_ref=calib_ref,
        calib_all=calib_all,
        predictions=pred_df,
        flags=flags_df
    )

    # ----------------- single combined export -----------------
    outdir = outdir or settings.get("current_run_folder", ".")
    os.makedirs(outdir, exist_ok=True)
    combined = []

    def add_section(name: str, frame: pd.DataFrame):
        f = frame.copy()
        f.insert(0, "section", name)
        combined.append(f)

    add_section("coefficients", result.coef_table)
    add_section("variance_decomp", result.var_decomp)
    add_section("calibration_ref", result.calib_ref)
    add_section("calibration_all", result.calib_all)
    add_section("predictions", result.predictions)
    add_section("flags", result.flags)

    big_df = pd.concat(combined, axis=0, ignore_index=True)
    big_csv = os.path.join(outdir, "surrogate_results.csv")
    big_xlsx = os.path.join(outdir, "surrogate_results.xlsx")
    big_df.to_csv(big_csv, index=False)
    try:
        big_df.to_excel(big_xlsx, index=False)
    except Exception:
        # Excel export optional; ignore if openpyxl not installed
        pass

    print(f"   Wrote single surrogate results CSV: {big_csv}")
    print(f"   Wrote single surrogate results XLSX: {big_xlsx}")

    return result
