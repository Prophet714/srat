# srat_g.py
# -----------------------------------------------------------------------------
# SRAT-G: Governance / SPC & Health Monitoring
#
# Consumes SRAT-C baseline (and optionally SRAT-S outputs) to compute:
#   - Reference-based control limits for one or more metrics (default: residual)
#   - Rolling KPIs (mean, std, MAE, RMSE), EWMA, CUSUM
#   - Capability indices (Cp/Cpk) vs user-specified specs
#   - Western Electric rule flags (OOC, runs, trends, 2-of-3 near limit)
#   - Aggregated summary and per-metric/per-row flags
#
# Exports a SINGLE CSV and a multi-sheet XLSX to the run folder.
# Dependencies: numpy, pandas
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import re
import numpy as np
import pandas as pd


# ------------------------------- data classes ------------------------------- #

@dataclass
class GovernanceOutputs:
    settings_used: pd.DataFrame
    reference_stats: pd.DataFrame
    control_limits: pd.DataFrame
    capability: pd.DataFrame
    rolling_kpis: pd.DataFrame
    ewma: pd.DataFrame
    cusum: pd.DataFrame
    flags: pd.DataFrame
    summary: pd.DataFrame


# --------------------------------- helpers --------------------------------- #

def _get_run_folder(settings: Dict[str, Any]) -> str:
    run = settings.get("current_run_folder") or settings.get("outdir") or "results"
    os.makedirs(run, exist_ok=True)
    return run

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def _reference_mask(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.Series:
    """
    Same precedence as SRAT-S:
      1) settings["reference_col"] -> boolean column
      2) settings["reference_lots"] -> list/str matched to df['lot']
      3) robust MAD filter on residual: |r - med| <= ref_k * MAD
    """
    mask = pd.Series(False, index=df.index)
    # (1) boolean col
    ref_col = settings.get("reference_col")
    if ref_col and ref_col in df.columns:
        col = df[ref_col]
        if col.dropna().isin([True, False]).all():
            return col.astype(bool).reindex(df.index, fill_value=False)

    # (2) list of lots
    ref_lots = settings.get("reference_lots")
    if ref_lots and "lot" in df.columns:
        if isinstance(ref_lots, str):
            lots = {s.strip() for s in ref_lots.replace(",", " ").split() if s.strip()}
        else:
            lots = set(map(str, list(ref_lots)))
        return df["lot"].astype(str).isin(lots)

    # (3) fallback: MAD filter on residual if present
    r = pd.to_numeric(df.get("residual", pd.Series(index=df.index, dtype=float)), errors="coerce")
    med = float(np.nanmedian(r))
    mad = _mad(r.values[np.isfinite(r.values)])
    k = float(settings.get("ref_k", 2.5))
    if np.isfinite(mad) and mad > 0:
        return np.abs(r - med) <= k * mad
    mask[:] = True
    return mask

def _metric_list(settings: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    """
    Metrics to govern. Default priority:
      residual (required for SRAT-G usefulness),
      surrogate_error (if SRAT-S ran),
      spring_rate_meas, K_pred (if present).
    Users can override with settings["govern_metrics"] = "m1,m2,..."
    """
    user = settings.get("govern_metrics")
    if user:
        metrics = [m.strip() for m in str(user).split(",") if m.strip()]
    else:
        metrics = []
        if "residual" in df.columns:
            metrics.append("residual")
        if "surrogate_error" in df.columns:
            metrics.append("surrogate_error")
        for m in ("spring_rate_meas", "K_pred"):
            if m in df.columns:
                metrics.append(m)
    # keep unique order
    out = []
    for m in metrics:
        if m not in out and m in df.columns:
            out.append(m)
    if not out:
        raise ValueError("SRAT-G: No usable governance metrics found in data.")
    return out


# ----------------------------- SPC core pieces ----------------------------- #

def _ref_mean_std(series: pd.Series, ref_mask: pd.Series) -> Tuple[float, float]:
    ref = series[ref_mask]
    mu = float(ref.mean())
    sd = float(ref.std(ddof=1)) if ref.size > 1 else float(np.nan)
    return mu, sd

def _control_limits(mu: float, sd: float, z: float = 3.0) -> Tuple[float, float, float, float]:
    """
    Return (UCL, LCL, UWL, LWL) using mean ± z*sd (z=3 default).
    Warning levels (UWL/LWL) use mean ± 2*sd.
    """
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan, np.nan
    ucl = mu + z * sd
    lcl = mu - z * sd
    uwl = mu + 2.0 * sd
    lwl = mu - 2.0 * sd
    return ucl, lcl, uwl, lwl

def _capability(series: pd.Series, lsl: Optional[float], usl: Optional[float]) -> Dict[str, float]:
    """
    Compute Cp and Cpk if spec limits are provided (LSL/USL).
    """
    x = pd.to_numeric(series, errors="coerce")
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1)) if x.count() > 1 else np.nan
    out = {"mu": mu, "sd": sd, "Cp": np.nan, "Cpk": np.nan}
    if not np.isfinite(sd) or sd == 0:
        return out
    if lsl is None and usl is None:
        return out
    if lsl is not None and usl is not None and usl > lsl:
        out["Cp"] = (usl - lsl) / (6.0 * sd)
        out["Cpk"] = min((usl - mu) / (3.0 * sd), (mu - lsl) / (3.0 * sd))
    elif usl is not None:
        out["Cp"] = np.nan
        out["Cpk"] = (usl - mu) / (3.0 * sd)
    elif lsl is not None:
        out["Cp"] = np.nan
        out["Cpk"] = (mu - lsl) / (3.0 * sd)
    return out

def _rolling_kpis(series: pd.Series, w: int) -> pd.DataFrame:
    """
    Rolling mean/std/MAE/RMSE over window w against the series' own mean (approx diagnostic).
    For residuals, mean≈0; for others, it's still a useful stability check.
    """
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce")})
    df["roll_mean"] = df["x"].rolling(w, min_periods=1).mean()
    df["roll_std"] = df["x"].rolling(w, min_periods=2).std()
    # errors vs rolling mean (one-step lag for unbiased error proxy)
    base = df["roll_mean"].shift(1)
    err = df["x"] - base
    df["roll_mae"] = err.abs().rolling(w, min_periods=1).mean()
    df["roll_rmse"] = np.sqrt((err.pow(2)).rolling(w, min_periods=1).mean())
    return df

def _ewma(series: pd.Series, lam: float = 0.2) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = np.empty_like(x, dtype=float)
    cur = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = cur
            continue
        cur = v if not np.isfinite(cur) else lam * v + (1 - lam) * cur
        out[i] = cur
    return pd.Series(out, index=series.index, name=f"EWMA_{lam}")

def _cusum(series: pd.Series, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """
    One-sided CUSUM around series mean (approx). k is ref value (sd units), h is decision interval.
    We compute standardized C+ and C- (not hard-thresholded to alerts; alerting is done in flags).
    """
    x = pd.to_numeric(series, errors="coerce")
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1)) or 1.0
    z = (x - mu) / sd
    c_plus = np.zeros(len(z)); c_minus = np.zeros(len(z))
    for i, zi in enumerate(z):
        zi = 0.0 if not np.isfinite(zi) else zi
        c_plus[i] = max(0.0, c_plus[i - 1] + (zi - k)) if i > 0 else max(0.0, zi - k)
        c_minus[i] = min(0.0, c_minus[i - 1] + (zi + k)) if i > 0 else min(0.0, zi + k)
    return pd.DataFrame({"CUSUM_plus": c_plus, "CUSUM_minus": c_minus}, index=series.index)

def _western_electric_flags(series: pd.Series, mu: float, sd: float) -> pd.DataFrame:
    """
    Western Electric-like rule checks:
      - rule1: any point beyond 3σ
      - rule2: 2 of 3 consecutive beyond 2σ on same side
      - rule3: 4 of 5 consecutive beyond 1σ on same side
      - rule4: 8 consecutive on same side of mean
      - rule5: 6-point monotonically increasing or decreasing (trend)
    Returns a DataFrame with boolean columns for each rule and 'any_rule'.
    """
    x = pd.to_numeric(series, errors="coerce").values
    n = len(x)
    out = {
        "rule1_gt3sigma": np.zeros(n, dtype=bool),
        "rule2_2of3_gt2sigma_same_side": np.zeros(n, dtype=bool),
        "rule3_4of5_gt1sigma_same_side": np.zeros(n, dtype=bool),
        "rule4_8_same_side": np.zeros(n, dtype=bool),
        "rule5_6_trend": np.zeros(n, dtype=bool),
    }
    if not np.isfinite(sd) or sd == 0 or n == 0:
        df = pd.DataFrame(out, index=series.index)
        df["any_rule"] = False
        return df

    z = (x - mu) / sd
    # rule1
    out["rule1_gt3sigma"] = np.abs(z) > 3.0

    # helper for same side
    def same_side(vals: np.ndarray) -> Optional[int]:
        # +1 if all > 0; -1 if all < 0; else 0
        pos = np.all(vals > 0)
        neg = np.all(vals < 0)
        if pos: return 1
        if neg: return -1
        return 0

    # rule2: 2 of 3 beyond 2σ on same side
    for i in range(2, n):
        window = z[i-2:i+1]
        sides = np.sign(window)
        mask2 = np.abs(window) > 2.0
        if mask2.sum() >= 2:
            # check same side among those >2σ
            signs = sides[mask2]
            if len(signs) >= 2 and (np.all(signs > 0) or np.all(signs < 0)):
                out["rule2_2of3_gt2sigma_same_side"][i] = True

    # rule3: 4 of 5 beyond 1σ on same side
    for i in range(4, n):
        window = z[i-4:i+1]
        mask1 = np.abs(window) > 1.0
        if mask1.sum() >= 4:
            signs = np.sign(window[mask1])
            if np.all(signs > 0) or np.all(signs < 0):
                out["rule3_4of5_gt1sigma_same_side"][i] = True

    # rule4: 8 same side of mean
    for i in range(7, n):
        window = z[i-7:i+1]
        s = same_side(window)
        if s in (1, -1):
            out["rule4_8_same_side"][i] = True

    # rule5: 6-point monotonic trend
    for i in range(5, n):
        w = x[i-5:i+1]
        inc = np.all(np.diff(w) > 0)
        dec = np.all(np.diff(w) < 0)
        out["rule5_6_trend"][i] = bool(inc or dec)

    df = pd.DataFrame(out, index=series.index)
    df["any_rule"] = df.any(axis=1)
    return df


# --------------------------------- main API -------------------------------- #

def run_srat_g(
    baseline_df: pd.DataFrame,
    settings: Dict[str, Any],
    s_result: Optional[Any] = None
) -> GovernanceOutputs:
    """
    SRAT-G governance layer. Works with SRAT-C baseline (and optionally SRAT-S).
    Will look for 'surrogate_error' from s_result.predictions if present.
    """

    df = baseline_df.copy()

    # If SRAT-S predictions are provided, merge surrogate_error into df for governance
    if s_result is not None and hasattr(s_result, "predictions") and isinstance(s_result.predictions, pd.DataFrame):
        pred = s_result.predictions.copy()
        # ensure index alignment: predictions had reset_index(drop=False)
        if "index" in pred.columns:
            pred = pred.set_index("index")
        # join on index; keep surrogate_error
        if "surrogate_error" in pred.columns:
            df.loc[pred.index, "surrogate_error"] = pred.loc[:, "surrogate_error"].values

    # Which metrics to monitor
    metrics = _metric_list(settings, df)
    df = _coerce_numeric(df, metrics + ["residual"])

    # Identify reference rows
    ref_mask = _reference_mask(df, settings)

    # SPC params
    z = float(settings.get("spc_z", 3.0))
    warn_z = float(settings.get("spc_warn_z", 2.0))  # used indirectly for rules
    ewma_lambda = float(settings.get("ewma_lambda", 0.2))
    cusum_k = float(settings.get("cusum_k", 0.5))
    cusum_h = float(settings.get("cusum_h", 5.0))  # not directly used to flag; output for context
    roll_w = int(settings.get("rolling_window", 8))

    settings_used = pd.DataFrame([{
        "metrics": ",".join(metrics),
        "ref_rows": int(ref_mask.sum()),
        "spc_z": z, "spc_warn_z": warn_z,
        "ewma_lambda": ewma_lambda,
        "cusum_k": cusum_k, "cusum_h": cusum_h,
        "rolling_window": roll_w
    }])

    # Per-metric computations
    ref_stats_rows = []
    cl_rows = []
    cap_rows = []
    rolling_blocks = []
    ewma_blocks = []
    cusum_blocks = []
    flags_blocks = []

    for m in metrics:
        series = pd.to_numeric(df[m], errors="coerce")

        # reference mean/std and limits
        mu, sd = _ref_mean_std(series, ref_mask)
        ucl, lcl, uwl, lwl = _control_limits(mu, sd, z=z)
        ref_stats_rows.append({"metric": m, "mu_ref": mu, "sd_ref": sd, "n_ref": int(ref_mask.sum())})
        cl_rows.append({"metric": m, "UCL": ucl, "LCL": lcl, "UWL": uwl, "LWL": lwl})

        # capability (optional spec)
        lsl = settings.get(f"{m}_LSL")
        usl = settings.get(f"{m}_USL")
        lsl = float(lsl) if lsl is not None and str(lsl).strip() != "" else None
        usl = float(usl) if usl is not None and str(usl).strip() != "" else None
        cap = _capability(series[ref_mask], lsl, usl)  # capability based on reference data
        cap_rows.append({"metric": m, "LSL": lsl, "USL": usl, **cap})

        # rolling KPIs
        roll = _rolling_kpis(series, w=roll_w)
        roll.insert(0, "metric", m)
        rolling_blocks.append(roll)

        # ewma/cusum
        ew = _ewma(series, lam=ewma_lambda)
        ew = pd.DataFrame({"metric": m, "EWMA": ew})
        ewma_blocks.append(ew)

        cu = _cusum(series, k=cusum_k, h=cusum_h)
        cu.insert(0, "metric", m)
        cusum_blocks.append(cu)

        # WE rule flags
        we = _western_electric_flags(series, mu, sd)
        we.insert(0, "metric", m)
        flags_blocks.append(we)

    reference_stats = pd.DataFrame(ref_stats_rows)
    control_limits = pd.DataFrame(cl_rows)
    capability = pd.DataFrame(cap_rows)

    rolling_kpis = pd.concat(rolling_blocks, axis=0).reset_index(names="index")
    ewma = pd.concat(ewma_blocks, axis=0).reset_index(names="index")
    cusum = pd.concat(cusum_blocks, axis=0).reset_index(names="index")
    flags = pd.concat(flags_blocks, axis=0).reset_index(names="index")

    # Build per-index summary across metrics (any_rule per index)
    any_by_index = flags.groupby("index")["any_rule"].any().rename("any_rule_any_metric")
    # also capture which metrics triggered
    trig = (
        flags.loc[flags["any_rule"], ["index", "metric"]]
        .groupby("index")["metric"]
        .apply(lambda s: ",".join(sorted(set(s.astype(str)))))
        .rename("metrics_triggered")
    )
    summary = pd.concat([any_by_index, trig], axis=1).fillna({"metrics_triggered": ""}).reset_index()

    outputs = GovernanceOutputs(
        settings_used=settings_used,
        reference_stats=reference_stats,
        control_limits=control_limits,
        capability=capability,
        rolling_kpis=rolling_kpis,
        ewma=ewma,
        cusum=cusum,
        flags=flags,
        summary=summary
    )

    _export_governance(outputs, settings)
    return outputs


# ---------------------------------- export --------------------------------- #

def _export_governance(out: GovernanceOutputs, settings: Dict[str, Any]) -> None:
    run_folder = _get_run_folder(settings)

    # One combined CSV with section tag
    sections: List[pd.DataFrame] = []

    def add(tag: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        x = df.copy()
        x.insert(0, "section", tag)
        sections.append(x)

    add("settings_used", out.settings_used)
    add("reference_stats", out.reference_stats)
    add("control_limits", out.control_limits)
    add("capability", out.capability)
    add("rolling_kpis", out.rolling_kpis)
    add("ewma", out.ewma)
    add("cusum", out.cusum)
    add("flags", out.flags)
    add("summary", out.summary)

    combo = pd.concat(sections, ignore_index=True) if sections else pd.DataFrame(columns=["section"])
    csv_path = os.path.join(run_folder, "governance_report.csv")
    combo.to_csv(csv_path, index=False)

    # Multi-sheet Excel
    xlsx_path = os.path.join(run_folder, "governance_report.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path) as xw:
            out.settings_used.to_excel(xw, sheet_name="settings_used", index=False)
            out.reference_stats.to_excel(xw, sheet_name="reference_stats", index=False)
            out.control_limits.to_excel(xw, sheet_name="control_limits", index=False)
            out.capability.to_excel(xw, sheet_name="capability", index=False)
            out.rolling_kpis.to_excel(xw, sheet_name="rolling_kpis", index=False)
            out.ewma.to_excel(xw, sheet_name="ewma", index=False)
            out.cusum.to_excel(xw, sheet_name="cusum", index=False)
            out.flags.to_excel(xw, sheet_name="flags", index=False)
            out.summary.to_excel(xw, sheet_name="summary", index=False)
    except Exception:
        # Excel is optional; don't crash on missing engines or file locks
        pass

    print(f"   Wrote SRAT-G report CSV:  {csv_path}")
    print(f"   Wrote SRAT-G report XLSX: {xlsx_path}")
