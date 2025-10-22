# physics_model.py
# SRAT-C physics baseline (quasi-static) with optional parasitic stiffness and rate correction.
#
# Exposes:
#   compute_baseline(fcd_df: pd.DataFrame, settings: dict) -> pd.DataFrame
#     Adds columns for shape factor S, Gent–Lindley factor f(S),
#     linear shear stiffness k_lin, predicted rotational spring rate K_pred,
#     and residual = K_meas - K_pred (if measurement is present).
#
# Assumptions:
#   * Quasi-static use of a single effective small-strain shear modulus (QLS, 50 psi).
#   * Small-angle linearization.
#   * Parasitics act in parallel and are modeled as an additive stiffness K_p (lbf-in/deg).
#   * Optional empirical rate correction: K_pred *= (omega/omega_ref) ** n
#
# Required SETTINGS:
#   mean_radius_in        : effective rotation radius [in]
#   loaded_area_in2       : loaded pad area A [in^2]
#   perimeter_in          : loaded perimeter P [in]
#
# Optional SETTINGS:
#   K_parasitic_mean      : additive parasitic stiffness [lbf-in/deg] (default 0.0)
#   use_rate_correction   : bool (default False)
#   test_rate_deg_s       : angular rate in deg/s (or provide omega_test_rad_s)
#   omega_test_rad_s      : angular rate in rad/s (takes precedence if provided)
#   omega_ref_rad_s       : reference rad/s for rate law (default 1.0)
#   rate_exp_n            : exponent n in (omega/omega_ref)**n (default 0.0)
#
# Required DATA columns in fcd_df:
#   shear_mod50_psi (QLS), thickness_in
# Optional:
#   loaded_area_in2, perimeter_in, mean_radius_in (row overrides)
#   spring_rate_meas -> will produce 'residual'

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


def _ensure_float(val: Any, name: str) -> float:
    try:
        return float(val)
    except Exception as e:
        raise ValueError(f"Expected a numeric value for '{name}', got {val!r}") from e


def _gent_lindley_factor(S: np.ndarray) -> np.ndarray:
    """Gent–Lindley stiffening factor f(S) = 1 + 2 S^2 (vectorized)."""
    return 1.0 + 2.0 * np.square(S)


def _shape_factor(A: np.ndarray, P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shape factor S = A / (2 P t) (vectorized). Guard against divide-by-zero."""
    denom = 2.0 * P * t
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(denom != 0.0, A / denom, np.nan)
    return S


def _compute_k_lin(G_eff: np.ndarray, A: np.ndarray, t: np.ndarray, fS: np.ndarray) -> np.ndarray:
    """Linear bonded-rubber shear stiffness k_lin = (G_eff * A / t) * f(S) in lbf/in (vectorized)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k_lin = np.where(t != 0.0, (G_eff * A / t) * fS, np.nan)
    return k_lin


def _rate_correction(K_pred: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    """Optional empirical rate correction: K *= (omega/omega_ref)**n."""
    if not bool(settings.get("use_rate_correction", False)):
        return K_pred

    n = _ensure_float(settings.get("rate_exp_n", 0.0), "rate_exp_n")
    omega_ref = _ensure_float(settings.get("omega_ref_rad_s", 1.0), "omega_ref_rad_s")

    if settings.get("omega_test_rad_s") is not None:
        omega = _ensure_float(settings.get("omega_test_rad_s"), "omega_test_rad_s")
    else:
        rate_deg_s = settings.get("test_rate_deg_s", None)
        if rate_deg_s is None:
            # No rate info; do nothing
            return K_pred
        omega = _ensure_float(rate_deg_s, "test_rate_deg_s") * (np.pi / 180.0)

    if omega <= 0.0 or omega_ref <= 0.0:
        return K_pred

    factor = (omega / omega_ref) ** n
    return K_pred * factor


def compute_baseline(fcd_df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    """Compute quasi-static baseline and residuals.

    Parameters
    ----------
    fcd_df : DataFrame
        Flexseal measurement table with (at minimum):
        - 'shear_mod50_psi' : effective small-strain shear modulus (QLS at 50 psi)
        - 'thickness_in'    : bondline thickness
        Optional:
        - 'loaded_area_in2', 'perimeter_in', 'mean_radius_in' (if absent, taken from settings)
        - 'spring_rate_meas': measured rotational spring rate (lbf-in/deg)

    settings : dict
        Must contain geometry defaults:
        - 'loaded_area_in2', 'perimeter_in', 'mean_radius_in'
        Optional:
        - 'K_parasitic_mean' (default 0.0), 'use_rate_correction', 'rate_exp_n',
          'omega_ref_rad_s', 'test_rate_deg_s' or 'omega_test_rad_s'.

    Returns
    -------
    DataFrame
        Original columns plus:
        - 'A_in2', 'P_in', 't_in', 'r_eff_in'
        - 'S', 'f_S', 'k_lin_lbf_per_in'
        - 'K_pred' [lbf-in/deg] (with optional parasitic and rate correction)
        - 'residual' if 'spring_rate_meas' is present
    """
    # Copy to avoid mutating caller
    df = fcd_df.copy()

    # Pull/validate settings geometry
    A_default = _ensure_float(settings.get("loaded_area_in2"), "loaded_area_in2")
    P_default = _ensure_float(settings.get("perimeter_in"), "perimeter_in")
    r_default = _ensure_float(settings.get("mean_radius_in"), "mean_radius_in")
    K_p = float(settings.get("K_parasitic_mean", 0.0))

    # Column mapping / fallbacks
    A = df.get("loaded_area_in2", pd.Series(A_default, index=df.index, dtype=float)).astype(float)
    P = df.get("perimeter_in", pd.Series(P_default, index=df.index, dtype=float)).astype(float)
    r_eff = df.get("mean_radius_in", pd.Series(r_default, index=df.index, dtype=float)).astype(float)

    # Required material & thickness columns
    if "shear_mod50_psi" not in df.columns:
        raise KeyError("'shear_mod50_psi' is required in the data table (QLS at 50 psi).")
    if "thickness_in" not in df.columns:
        raise KeyError("'thickness_in' is required in the data table.")

    G_eff = df["shear_mod50_psi"].astype(float).values
    t_in = df["thickness_in"].astype(float).values
    A_vals = A.astype(float).values
    P_vals = P.astype(float).values
    r_vals = r_eff.astype(float).values

    # Shape factor and GL factor
    S = _shape_factor(A_vals, P_vals, t_in)
    f_S = _gent_lindley_factor(S)

    # Linear shear stiffness (lbf/in)
    k_lin = _compute_k_lin(G_eff, A_vals, t_in, f_S)

    # Convert to rotational spring rate in lbf-in/deg
    # K_rot (lbf-in/rad) = k_lin (lbf/in) * r_eff^2; then convert to per degree by (pi/180)
    K_rot_rad = k_lin * np.square(r_vals)          # lbf-in / rad
    K_pred = K_rot_rad * (np.pi / 180.0)           # lbf-in / deg

    # Add parasitic stiffness (parallel) if any
    if K_p != 0.0:
        K_pred = K_pred + float(K_p)

    # Optional empirical rate correction
    K_pred = _rate_correction(K_pred, settings)

    # Assemble outputs
    df_out = df.copy()
    df_out["A_in2"] = A_vals
    df_out["P_in"] = P_vals
    df_out["t_in"] = t_in
    df_out["r_eff_in"] = r_vals
    df_out["S"] = S
    df_out["f_S"] = f_S
    df_out["k_lin_lbf_per_in"] = k_lin
    df_out["K_pred"] = K_pred

    # Residual if measurement exists
    if "spring_rate_meas" in df_out.columns:
        sr = pd.to_numeric(df_out["spring_rate_meas"], errors="coerce").values
        df_out["residual"] = sr - K_pred

    return df_out
