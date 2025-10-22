# srat_a.py
# -----------------------------------------------------------------------------
# SRAT-A: Anomaly Evaluator
#
# Uses the SRAT-S surrogate to:
#  * select anomalies (auto via SRAT-S flags or manual via settings),
#  * run counterfactual one-at-a-time resets to reference means,
#  * compute minimum single-knob adjustment to hit a target residual,
#  * estimate local sensitivity (Morris-lite) around each anomaly,
#  * export a single CSV and a multi-sheet XLSX in the run folder.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import re
import glob
import numpy as np
import pandas as pd


@dataclass
class AnomalyOutputs:
    catalog: pd.DataFrame
    counterfactual: pd.DataFrame
    min_adjust: pd.DataFrame
    local_sensitivity: pd.DataFrame
    summary: pd.DataFrame


# ------------------------------ utils -------------------------------- #

def _get_run_folder(settings: Dict[str, Any]) -> str:
    outdir = settings.get("current_run_folder") or settings.get("outdir") or "results"
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _parse_range_token(tok: str, base: Optional[float] = None) -> Tuple[float, float]:
    """
    Parse a single bound token like '±5', '±5%', '0.38..0.42', '[-0.01,0.02]'.
    Returns (delta_minus, delta_plus) for relative; for absolute returns (low, high).
    """
    s = str(tok).strip()
    # ±N or ±N%
    m = re.match(r"^±\s*([0-9.]+)\s*(%)?$", s)
    if m:
        val = float(m.group(1))
        if m.group(2):  # percent
            if base is None or np.isnan(base):
                return (0.0, 0.0)
            d = abs(base) * val / 100.0
            return (-d, +d)
        return (-val, +val)

    # Absolute range "a..b" or "[a,b]"
    m = re.match(r"^\[?\s*([\-0-9.]+)\s*[,\.]{1,2}\s*([\-0-9.]+)\s*\]?$", s)
    if m:
        lo = float(m.group(1)); hi = float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    # Plain number => treat as ±number
    try:
        v = float(s)
        return (-v, +v)
    except Exception:
        return (0.0, 0.0)

def _range_for_var(name: str, settings: Dict[str, Any], x0: float) -> Tuple[float, float, bool]:
    """
    Return (low, high, absolute_mode).
    - If settings has 'range_<name>' token:
        * '±5' / '±5%' => relative around x0
        * 'a..b' / '[a,b]' => absolute bounds, absolute_mode=True
    - Else falls back to 'local_perturb_pct' as ±% of x0 (default 10%).
    """
    key = f"range_{name}"
    tok = settings.get(key)
    if tok is not None and str(tok).strip():
        lo, hi = _parse_range_token(str(tok), base=x0)
        abs_match = bool(re.match(r"^\[?\s*([\-0-9.]+)\s*[,\.]{1,2}\s*([\-0-9.]+)\s*\]?$", str(tok).strip()))
        if abs_match:
            return (float(lo), float(hi), True)
        # relative deltas
        return (x0 + lo, x0 + hi, False)

    # Fallback: ±p% of x0
    p = float(settings.get("local_perturb_pct", 0.10))
    lo = x0 * (1.0 - p)
    hi = x0 * (1.0 + p)
    return (lo, hi, False)

def _ols_params_from_model(model_obj: Any, predictors: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Extract (intercept, {var: coef}) from a statsmodels OLS/MixedLM fit.
    """
    params = getattr(model_obj, "params", None)
    if params is None:
        raise ValueError("SRAT-S model_obj has no .params; expected statsmodels fit.")
    intercept = 0.0
    coefs: Dict[str, float] = {}
    # statsmodels may label intercept 'Intercept' or 'const'
    for k, v in params.items():
        k = str(k)
        if k in ("Intercept", "const"):
            intercept = float(v)
        else:
            coefs[k] = float(v)
    coefs = {k: coefs.get(k, 0.0) for k in predictors}
    return intercept, coefs

def _ols_params_from_surrogate_csv(csv_path: str) -> Tuple[List[str], float, Dict[str, float], pd.DataFrame, pd.Index]:
    """
    Parse the single SRAT-S CSV (combined sections) and recover:
      - predictors (from coefficients excluding Intercept)
      - intercept, coefs dict
      - flags DataFrame (needs 'index' and 'is_anomaly_gt3MAD')
      - a best-effort train_index (robust inlier mask on residuals if predictions exist)
    """
    df = pd.read_csv(csv_path)
    if "section" not in df.columns:
        raise ValueError("surrogate_results.csv missing 'section' column.")

    coef = df[df["section"] == "coefficients"].copy()
    if coef.empty or "name" not in coef.columns or "estimate" not in coef.columns:
        raise ValueError("No coefficients section in surrogate CSV.")
    coef["name"] = coef["name"].astype(str)
    coef["estimate"] = pd.to_numeric(coef["estimate"], errors="coerce")

    # intercept
    inter_mask = coef["name"].isin(["Intercept", "const"])
    intercept_series = coef.loc[inter_mask, "estimate"]
    intercept = float(intercept_series.iloc[0]) if not intercept_series.empty else 0.0

    # predictors / coefs
    pred_rows = coef.loc[~inter_mask, ["name", "estimate"]].dropna()
    predictors = pred_rows["name"].tolist()
    coefs = dict(zip(pred_rows["name"], pred_rows["estimate"]))

    # flags
    flags = df[df["section"] == "flags"].copy()
    if not flags.empty and "index" in flags.columns:
        flags = flags[["index"] + [c for c in flags.columns if c not in ("section", "index")]]
        flags["index"] = pd.to_numeric(flags["index"], errors="coerce").astype("Int64")
        flags = flags.dropna(subset=["index"]).copy()
        flags = flags.set_index(flags["index"].astype(int), drop=True)
        flags.drop(columns=["index"], inplace=True, errors="ignore")
    else:
        # empty fallback
        flags = pd.DataFrame()

    # train_index heuristic:
    # if predictions are present, take inliers (|error - med| <= 2.5*MAD)
    train_index = pd.Index([])
    preds = df[df["section"] == "predictions"].copy()
    if not preds.empty and {"index", "residual", "pred_surrogate"}.issubset(preds.columns):
        preds["index"] = pd.to_numeric(preds["index"], errors="coerce").astype("Int64")
        preds = preds.dropna(subset=["index"]).copy()
        preds["residual"] = pd.to_numeric(preds["residual"], errors="coerce")
        preds["pred_surrogate"] = pd.to_numeric(preds["pred_surrogate"], errors="coerce")
        err = (preds["residual"] - preds["pred_surrogate"]).to_numpy()
        med = float(np.nanmedian(err))
        mad = float(np.nanmedian(np.abs(err - med)))
        if mad > 0 and preds["index"].notna().any():
            inliers = np.abs(err - med) <= 2.5 * mad
            train_index = pd.Index(preds.loc[inliers, "index"].astype(int).tolist())

    return predictors, intercept, coefs, flags, train_index

def _predict_linear(intercept: float, coefs: Dict[str, float], row: pd.Series) -> float:
    s = intercept
    for k, b in coefs.items():
        s += b * float(row.get(k, np.nan))
    return s

# --------------------------- core computations --------------------------- #

def _select_anomalies(baseline_df: pd.DataFrame,
                      s_flags: pd.DataFrame,
                      settings: Dict[str, Any]) -> pd.Series:
    """
    Returns a boolean mask over baseline_df.index for rows to analyze.
    Priority:
      1) anomaly_select == "list": use anomaly_list tokens "Lot-Roll" OR integer indices
      2) else use anomaly_flag_col from s_flags (default: is_anomaly_gt3MAD)
    """
    sel_mode = str(settings.get("anomaly_select", "auto")).lower()
    if sel_mode == "list":
        raw = str(settings.get("anomaly_list", "")).strip()
        if not raw:
            return pd.Series(False, index=baseline_df.index)
        tokens = [t.strip() for t in re.split(r"[;, ]+", raw) if t.strip()]
        mask = pd.Series(False, index=baseline_df.index)
        # Try lot-roll tokens first
        if "lot" in baseline_df.columns and "roll" in baseline_df.columns:
            lr = (baseline_df["lot"].astype(str) + "-" + baseline_df["roll"].astype(str)).tolist()
            lr_map = {tok: i for i, tok in enumerate(lr)}
            for t in tokens:
                if t.isdigit():
                    idx = int(t)
                    if idx in mask.index:
                        mask.loc[idx] = True
                else:
                    i = lr_map.get(t)
                    if i is not None and i in mask.index:
                        mask.loc[i] = True
        else:
            for t in tokens:
                if t.isdigit():
                    idx = int(t)
                    if idx in mask.index:
                        mask.loc[idx] = True
        return mask

    # auto mode
    flag_col = str(settings.get("anomaly_flag_col", "is_anomaly_gt3MAD"))
    flags = s_flags.copy()
    if "index" in flags.columns:
        flags = flags.set_index("index", drop=True)
    out = pd.Series(False, index=baseline_df.index)
    if flag_col in flags.columns:
        common = out.index.intersection(flags.index)
        out.loc[common] = flags.loc[common, flag_col].astype(bool)
    return out

def _reference_means(df: pd.DataFrame, train_index: pd.Index, predictors: List[str]) -> Dict[str, float]:
    if len(train_index) == 0:
        # fallback: use all rows as "reference"
        ref = df[predictors].copy()
    else:
        ref = df.loc[train_index, predictors].copy()
    ref = _coerce_numeric(ref, predictors)
    return {p: float(ref[p].mean()) for p in predictors}

def _counterfactual_one_at_a_time(df: pd.DataFrame,
                                  idx_list: List[int],
                                  predictors: List[str],
                                  intercept: float,
                                  coefs: Dict[str, float],
                                  ref_means: Dict[str, float],
                                  target: float,
                                  ref_policy: str) -> pd.DataFrame:
    rows = []
    for i in idx_list:
        row = df.loc[i]
        r0 = _predict_linear(intercept, coefs, row)
        for p in predictors:
            row_cf = row.copy()
            # For now ref_policy variants map to same ref_means; hook is here for future policies
            new_val = ref_means[p]
            row_cf[p] = new_val
            r_cf = _predict_linear(intercept, coefs, row_cf)
            rows.append({
                "index": i,
                "predictor": p,
                "baseline_residual": r0,
                "reset_to": new_val,
                "r_after_reset": r_cf,
                "delta_r": r_cf - r0,
                "toward_target": abs(r_cf - target) < abs(r0 - target)
            })
    return pd.DataFrame(rows)

def _min_adjust_single_knob(df: pd.DataFrame,
                            idx_list: List[int],
                            predictors: List[str],
                            intercept: float,
                            coefs: Dict[str, float],
                            settings: Dict[str, Any],
                            target: float,
                            tol: float = 0.05) -> pd.DataFrame:
    """
    Closed-form along 1D (linear surrogate):
      r_hat = a + sum b_k x_k
      solve for x_j*:  a + b_j x*_j + sum_{k≠j} b_k x_k = target
      => x*_j = (target - a - sum_{k≠j} b_k x_k) / b_j
    Then clamp to per-variable bounds from settings; mark feasibility.
    """
    rows = []
    for i in idx_list:
        base = df.loc[i].copy()
        r0 = _predict_linear(intercept, coefs, base)
        for p in predictors:
            b_j = coefs.get(p, 0.0)
            if abs(b_j) < 1e-12:
                rows.append({
                    "index": i, "predictor": p,
                    "baseline_residual": r0,
                    "required_adjustment": np.nan,
                    "new_value": np.nan,
                    "predicted_residual": r0,
                    "hit_target": False,
                    "feasible": False,
                    "reason": "zero_sensitivity"
                })
                continue
            # Sum others
            sum_others = intercept
            for k, b_k in coefs.items():
                if k == p:
                    continue
                sum_others += b_k * float(base.get(k, np.nan))
            x_star = (target - sum_others) / b_j
            # Range for this variable
            cur = float(base.get(p, np.nan))
            lo, hi, _absolute_mode = _range_for_var(p, settings, cur)
            # Clamp
            x_new = float(np.clip(x_star, lo, hi))
            # New residual
            temp = base.copy()
            temp[p] = x_new
            r_new = _predict_linear(intercept, coefs, temp)
            hit = abs(r_new - target) <= tol
            feas = (x_new == x_star)
            rows.append({
                "index": i, "predictor": p,
                "baseline_residual": r0,
                "required_adjustment": x_star - cur,
                "new_value": x_new,
                "predicted_residual": r_new,
                "hit_target": bool(hit),
                "feasible": bool(feas),
                "reason": "" if feas else "clamped_to_bounds"
            })
    return pd.DataFrame(rows)

def _local_morris(df: pd.DataFrame,
                  idx_list: List[int],
                  predictors: List[str],
                  intercept: float,
                  coefs: Dict[str, float],
                  settings: Dict[str, Any],
                  trajectories: int = 8,
                  steps: int = 6) -> pd.DataFrame:
    """
    Morris-lite around each anomaly:
      - Build per-variable grids within bounds (from settings).
      - For each trajectory, random start on the grid; step each var once.
      - Compute elementary effect per var; aggregate μ* (mean abs) and σ.
    """
    rows = []
    rng = np.random.default_rng(0)
    for i in idx_list:
        base = df.loc[i].copy()
        # Build grids and bounds
        grids = {}
        for p in predictors:
            cur = float(base.get(p, np.nan))
            lo, hi, _ = _range_for_var(p, settings, cur)
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = cur, cur
            grids[p] = np.array([cur]) if hi == lo else np.linspace(lo, hi, steps)

        def eval_row(r: pd.Series) -> float:
            return _predict_linear(intercept, coefs, r)

        ee: Dict[str, List[float]] = {p: [] for p in predictors}
        # trajectories
        for _t in range(trajectories):
            r = base.copy()
            # random start on grid for each predictor
            for p in predictors:
                r[p] = float(rng.choice(grids[p]))
            y0 = eval_row(r)
            # step each predictor once (random neighbor on grid)
            for p in predictors:
                grid = grids[p]
                if len(grid) < 2:
                    continue
                cur_val = float(r[p])
                j = np.searchsorted(grid, cur_val)
                if j <= 0:
                    j2 = 1
                elif j >= len(grid) - 1:
                    j2 = len(grid) - 2
                else:
                    j2 = j + (1 if rng.random() < 0.5 else -1)
                x2 = float(grid[j2])
                r2 = r.copy()
                r2[p] = x2
                y1 = eval_row(r2)
                delta = (y1 - y0) / (x2 - cur_val) if (x2 - cur_val) != 0 else 0.0
                ee[p].append(delta)
                r = r2
                y0 = y1

        for p in predictors:
            vals = np.array(ee[p]) if ee[p] else np.array([0.0])
            mu_star = float(np.mean(np.abs(vals)))
            sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append({
                "index": i, "predictor": p,
                "morris_mu_abs": mu_star,
                "morris_sigma": sigma
            })
    return pd.DataFrame(rows)


# ------------------------------ export -------------------------------- #

def _export_anomaly(out: AnomalyOutputs, settings: Dict[str, Any]) -> None:
    run_folder = _get_run_folder(settings)

    # Combined CSV with section tag
    sections: List[pd.DataFrame] = []
    def add(name: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        x = df.copy()
        x.insert(0, "section", name)
        sections.append(x)

    add("catalog", out.catalog)
    add("counterfactual_one_at_a_time", out.counterfactual)
    add("min_adjust_to_target", out.min_adjust)
    add("local_sensitivity_morris", out.local_sensitivity)
    add("summary", out.summary)

    combo = pd.concat(sections, ignore_index=True) if sections else pd.DataFrame(columns=["section"])
    csv_path = os.path.join(run_folder, "anomaly_report.csv")
    combo.to_csv(csv_path, index=False)

    # Multi-sheet Excel (optional)
    xlsx_path = os.path.join(run_folder, "anomaly_report.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path) as xw:
            if not out.catalog.empty: out.catalog.to_excel(xw, sheet_name="catalog", index=False)
            if not out.counterfactual.empty: out.counterfactual.to_excel(xw, sheet_name="counterfactual", index=False)
            if not out.min_adjust.empty: out.min_adjust.to_excel(xw, sheet_name="min_adjust", index=False)
            if not out.local_sensitivity.empty: out.local_sensitivity.to_excel(xw, sheet_name="local_sensitivity", index=False)
            if not out.summary.empty: out.summary.to_excel(xw, sheet_name="summary", index=False)
    except Exception:
        # Excel export is best-effort; ignore openpyxl/file-lock issues
        pass

    print(f"   Wrote SRAT-A report CSV:  {csv_path}")
    print(f"   Wrote SRAT-A report XLSX: {xlsx_path}")


# ------------------------------ public API ------------------------------ #

def run_srat_a(
    baseline_df: pd.DataFrame,
    s_result: Any,
    settings: Dict[str, Any]
) -> AnomalyOutputs:
    """
    SRAT-A anomaly evaluation. Prefers a live SRAT-S result; if unavailable,
    falls back to loading the single 'surrogate_results.csv' from the current
    run folder and reconstructing intercept/coeffs/flags.
    """
    # Try preferred route: use in-memory SRAT-S model
    predictors: List[str] = []
    intercept: float = 0.0
    coefs: Dict[str, float] = {}
    flags_df = pd.DataFrame()
    train_index = pd.Index([])

    try:
        if s_result is not None and getattr(s_result, "model_obj", None) is not None:
            predictors = list(getattr(s_result, "predictors", []))
            intercept, coefs = _ols_params_from_model(s_result.model_obj, predictors)
            flags_df = s_result.flags.copy() if hasattr(s_result, "flags") else pd.DataFrame()
            if "index" in flags_df.columns:
                flags_df = flags_df.set_index("index", drop=True)
            train_index = getattr(s_result, "train_index", pd.Index([]))
        else:
            raise RuntimeError("No live SRAT-S model present.")
    except Exception as e:
        # Fallback route: parse surrogate_results.csv from run folder
        run_folder = _get_run_folder(settings)
        candidates = sorted(glob.glob(os.path.join(run_folder, "surrogate_results.csv")))
        if not candidates:
            raise ValueError(
                "SRAT-A: No live SRAT-S model and no surrogate_results.csv found in run folder."
            ) from e
        csv_path = candidates[-1]
        predictors, intercept, coefs, flags_df, train_index = _ols_params_from_surrogate_csv(csv_path)

    if not predictors or not coefs:
        raise ValueError("SRAT-A: could not recover predictors/coefficients from SRAT-S.")

    df = baseline_df.copy()
    df = _coerce_numeric(df, ["residual"] + predictors)

    # Choose anomalies
    anomaly_mask = _select_anomalies(df, flags_df, settings)
    idx_list = [int(i) for i in df.index[anomaly_mask].tolist()]

    catalog = pd.DataFrame({"index": idx_list})
    if "lot" in df.columns:
        catalog["lot"] = df.loc[idx_list, "lot"].astype(str).values
    if "roll" in df.columns:
        catalog["roll"] = df.loc[idx_list, "roll"].astype(str).values
    catalog["residual"] = df.loc[idx_list, "residual"].values

    if len(idx_list) == 0:
        outputs = AnomalyOutputs(
            catalog=catalog,
            counterfactual=pd.DataFrame(columns=["index","predictor","baseline_residual","reset_to","r_after_reset","delta_r","toward_target"]),
            min_adjust=pd.DataFrame(columns=["index","predictor","baseline_residual","required_adjustment","new_value","predicted_residual","hit_target","feasible","reason"]),
            local_sensitivity=pd.DataFrame(columns=["index","predictor","morris_mu_abs","morris_sigma"]),
            summary=pd.DataFrame(columns=["index","lot","roll","top_contributors","smallest_fix","recommendation"])
        )
        _export_anomaly(outputs, settings)
        return outputs

    # Reference means (for counterfactual resets)
    ref_means = _reference_means(df, train_index, predictors)

    # Config
    target = float(settings.get("target_residual", 0.0))
    ref_policy = str(settings.get("ref_policy", "global_mean")).lower()

    # Counterfactual one-at-a-time
    cf_df = _counterfactual_one_at_a_time(df, idx_list, predictors, intercept, coefs, ref_means, target, ref_policy)

    # Minimum single-knob adjust to hit target
    min_df = _min_adjust_single_knob(df, idx_list, predictors, intercept, coefs, settings, target, tol=float(settings.get("target_tol", 0.05)))

    # Local sensitivity (Morris-lite)
    ls_df = _local_morris(df, idx_list, predictors, intercept, coefs, settings,
                          trajectories=int(settings.get("local_mc_trajectories", 8)),
                          steps=int(settings.get("local_mc_steps", 6)))

    # Build summary per anomaly
    summary_rows = []
    for i in idx_list:
        # Top contributors by counterfactual delta toward target (most negative delta_r is best)
        cf_i = cf_df[cf_df["index"] == i].copy().sort_values("delta_r")
        top3 = "; ".join((cf_i["predictor"].head(3) + "=" + cf_i["delta_r"].head(3).round(3).astype(str)).tolist())

        # Smallest single-knob feasible fix
        min_i = min_df[(min_df["index"] == i) & (min_df["hit_target"])].copy()
        if not min_i.empty:
            min_i["abs_adj"] = min_i["required_adjustment"].abs()
            best = min_i.sort_values(["abs_adj"]).iloc[0]
            smallest_fix = f"{best['predictor']} -> {best['new_value']:.6g} (Δ={best['required_adjustment']:.6g})"
            rec = f"Set {best['predictor']} to {best['new_value']:.6g} to reach residual≈{target}."
        else:
            smallest_fix = "none_feasible_single_knob"
            top2 = cf_i.head(2)["predictor"].tolist()
            rec = f"Single-knob infeasible within bounds; consider paired changes on {', '.join(top2)}."

        row = {
            "index": i,
            "lot": df.loc[i, "lot"] if "lot" in df.columns else "",
            "roll": df.loc[i, "roll"] if "roll" in df.columns else "",
            "top_contributors": top3,
            "smallest_fix": smallest_fix,
            "recommendation": rec
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    outputs = AnomalyOutputs(
        catalog=catalog.reset_index(drop=True),
        counterfactual=cf_df.reset_index(drop=True),
        min_adjust=min_df.reset_index(drop=True),
        local_sensitivity=ls_df.reset_index(drop=True),
        summary=summary.reset_index(drop=True)
    )

    _export_anomaly(outputs, settings)
    return outputs
