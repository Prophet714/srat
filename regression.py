import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.utils import resample
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class FitResult:
    coef_table: pd.DataFrame
    var_decomp: pd.DataFrame
    model_obj: Optional[Any]


def fit_hierarchical_residual_model(
    df: pd.DataFrame,
    settings: dict,
    *,
    response_col: str = "residual"
) -> FitResult:
    """
    Fit a residual model. Defaults to OLS (robust for small datasets).
    If settings['use_mixedlm'] is True and grouping columns exist,
    runs MixedLM instead. Adds bootstrap confidence ranges for OLS.
    """

    if response_col not in df.columns:
        return FitResult(pd.DataFrame(), pd.DataFrame(), None)

    work = df.dropna(subset=[response_col]).copy()
    if work.empty:
        return FitResult(pd.DataFrame(), pd.DataFrame(), None)

    candidates = [
        "shear_mod50_psi", "hardnessA", "thickness_in",
        "age_days", "cure_torque", "t90_min"
    ]
    predictors = [c for c in candidates if c in work.columns]
    if not predictors:
        return FitResult(pd.DataFrame(), pd.DataFrame(), None)

    # shrink predictors if n is very small
    n = len(work)
    max_pred = max(1, min(len(predictors), n - 2))
    if len(predictors) > max_pred:
        corr = work[predictors].corrwith(work[response_col]).abs().sort_values(ascending=False)
        predictors = list(corr.index[:max_pred])

    formula = f"{response_col} ~ {' + '.join(predictors)}"

    # ---------- choose model ----------
    use_mixedlm = bool(settings.get("use_mixedlm", False))

    if use_mixedlm and ({"lot", "roll"} & set(work.columns)):
        # ===== MixedLM path =====
        if {"lot", "roll"} <= set(work.columns):
            work["lot_roll"] = work["lot"].astype(str) + "_" + work["roll"].astype(str)
            group_var = "lot_roll"
        else:
            group_var = "lot" if "lot" in work.columns else "roll"

        model = smf.mixedlm(formula, work, groups=work[group_var])
        fit = model.fit(reml=True)
        tbl = fit.summary().tables[1]

        if hasattr(tbl, "data"):  # old SimpleTable
            coef_df = pd.DataFrame(tbl.data[1:], columns=tbl.data[0])
            coef_df.rename(columns={
                "Coef.": "estimate", "[0.025": "ci_low",
                "0.975]": "ci_high", "P>|z|": "p_value"
            }, inplace=True)
            coef_df["name"] = coef_df.index
            for col in ["estimate", "ci_low", "ci_high", "p_value"]:
                coef_df[col] = pd.to_numeric(coef_df[col], errors="coerce")
        else:
            coef_df = pd.DataFrame(tbl).copy()
            rename_map = {
                "Coef.": "estimate", "[0.025": "ci_low",
                "0.975]": "ci_high", "P>|z|": "p_value"
            }
            coef_df.rename(columns={c: rename_map.get(c, c) for c in coef_df.columns}, inplace=True)
            for col in ["estimate", "ci_low", "ci_high", "p_value"]:
                if col in coef_df.columns:
                    coef_df[col] = pd.to_numeric(coef_df[col], errors="coerce")
            if "name" not in coef_df.columns:
                coef_df.insert(0, "name", coef_df.index.astype(str))

        model_obj = fit

    else:
        # ===== OLS path =====
        model = smf.ols(formula, data=work)
        fit = model.fit()
        ci = fit.conf_int(alpha=0.05)
        coef_df = pd.DataFrame({
            "name": fit.params.index,
            "estimate": fit.params.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "p_value": fit.pvalues.values
        })

        # --- Bootstrap confidence ranges ---
        boot_params = []
        for _ in range(1000):
            samp = resample(work, replace=True, random_state=None)
            try:
                boot_fit = smf.ols(formula, data=samp).fit()
                boot_params.append(boot_fit.params)
            except Exception:
                continue
        if boot_params:
            boot_df = pd.DataFrame(boot_params)
            coef_df["boot_median"]  = boot_df.median()
            coef_df["boot_low_5"]   = boot_df.quantile(0.05)
            coef_df["boot_high_95"] = boot_df.quantile(0.95)

        model_obj = fit

    # ---------- variance decomposition ----------
    base_r2 = smf.ols(formula, data=work).fit().rsquared
    var_rows = []
    for p in predictors:
        reduced = [x for x in predictors if x != p]
        red_formula = f"{response_col} ~ {' + '.join(reduced)}" if reduced else f"{response_col} ~ 1"
        r2_reduced = smf.ols(red_formula, data=work).fit().rsquared
        part_r2 = max(base_r2 - r2_reduced, 0.0) * 100.0
        var_rows.append({"factor": p, "variance_share": part_r2})
    var_df = pd.DataFrame(var_rows).sort_values("variance_share", ascending=False)

    return FitResult(
        coef_table=coef_df.reset_index(drop=True),
        var_decomp=var_df.reset_index(drop=True),
        model_obj=model_obj
    )
