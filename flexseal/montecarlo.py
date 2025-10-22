import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class MCOutput:
    samples: pd.DataFrame     # inputs + K_sim
    summary: pd.DataFrame     # mean, std, CI
    sensitivity: pd.DataFrame # factor, sobol_total (if computed)

def run_mc_and_sensitivity(baseline_df: pd.DataFrame, fit_result, settings: dict) -> MCOutput:
    n = int(settings["iterations"])
    rng = np.random.default_rng()

    # Sample inputs from deck means/stds
    thickness = rng.normal(settings["mean_thickness"], settings["std_thickness"], n)
    G50 = rng.normal(settings["mean_shear_mod50"], settings["std_shear_mod50"], n)
    hardness = rng.normal(settings["mean_hardnessA"], settings["std_hardnessA"], n)

    # Simple surrogate for demo: K_sim = (G50 / thickness) + alpha*hardness
    # TODO: replace with physics + fitted regression coefficients
    alpha = 0.0
    K_sim = (G50 / thickness) + alpha * hardness

    samples = pd.DataFrame({
        "thickness_in": thickness,
        "shear_mod50_psi": G50,
        "hardnessA": hardness,
        "K_sim": K_sim, 
    })
    mean = float(np.mean(K_sim))
    std = float(np.std(K_sim, ddof=1))
    ci = (mean - 1.96*std/np.sqrt(n), mean + 1.96*std/np.sqrt(n))
    summary = pd.DataFrame([{"mean": mean, "std": std, "ci_low": ci[0], "ci_high": ci[1]}])

    # Sensitivity placeholder (wire to SALib later)
    sensitivity = pd.DataFrame([
        {"factor":"shear_mod50_psi","sobol_total":np.nan},
        {"factor":"thickness_in","sobol_total":np.nan},
        {"factor":"hardnessA","sobol_total":np.nan},
    ])

    return MCOutput(samples=samples, summary=summary, sensitivity=sensitivity)
