"""
Optional time-dependent analysis for SRAT.

This version *prompts* for a second input-deck path.  If the user
presses Enter (or the file is missing), the module does nothing.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Dict, Any


def prompt_and_load_time_deck() -> pd.DataFrame | None:
    """
    Ask the user for a path to the time-dependent input deck.
    Return a DataFrame or None if the user skips or the file is invalid.
    """
    path = input(
        "\n[Time analysis] Enter path to time-dependent input deck "
        "(or press Enter to skip): "
    ).strip()

    if not path:
        print("→ No time deck provided; skipping time analysis.")
        return None
    if not os.path.isfile(path):
        print(f"→ File not found: {path}.  Skipping time analysis.")
        return None

    try:
        df = pd.read_csv(path, comment="#")
        print(f"→ Loaded {len(df)} rows from {path}")
        return df
    except Exception as exc:
        print(f"→ Could not read {path}: {exc}.  Skipping time analysis.")
        return None


def analyze_time_effects(
    base_df: pd.DataFrame,
    settings: Dict[str, Any],
    *,
    response_col: str = "residual",
    date_col: str = "date",
    cure_time_col: str = "cure_time_min",
    cure_temp_col: str = "cure_temp_F"
) -> Dict[str, pd.DataFrame]:
    """
    If the user provides a time-deck, analyse residual vs. date
    and cure metrics.  Otherwise return {}.
    """
    outputs: Dict[str, pd.DataFrame] = {}

    time_df = prompt_and_load_time_deck()
    if time_df is None:
        return outputs

    # merge baseline residuals into the time_df on any common key columns
    merged = pd.merge(time_df, base_df, how="inner")

    if response_col not in merged:
        print("→ Residual column missing in merged data; skipping time analysis.")
        return outputs

    work = merged.dropna(subset=[response_col]).copy()
    if work.empty:
        print("→ No residual values available; skipping time analysis.")
        return outputs

    # ----- trend vs. date ----------------------------------------------------
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        start_date = work[date_col].min()
        work["days_since_start"] = (work[date_col] - start_date).dt.days
        if work["days_since_start"].notna().sum() > 2:
            model = smf.ols(f"{response_col} ~ days_since_start", data=work).fit()
            ci = model.conf_int()
            outputs["time_trend"] = pd.DataFrame({
                "name": model.params.index,
                "estimate": model.params.values,
                "ci_low": ci[0].values,
                "ci_high": ci[1].values,
                "p_value": model.pvalues.values
            })

    # ----- cure metrics ------------------------------------------------------
    predictors = [c for c in (cure_time_col, cure_temp_col) if c in work.columns]
    if predictors:
        formula = f"{response_col} ~ {' + '.join(predictors)}"
        model = smf.ols(formula, data=work).fit()
        ci = model.conf_int()
        outputs["cure_metrics"] = pd.DataFrame({
            "name": model.params.index,
            "estimate": model.params.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "p_value": model.pvalues.values
        })

    return outputs
