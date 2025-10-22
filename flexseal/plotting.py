import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
import os

def make_summary_plots(baseline_df: pd.DataFrame, fit_result: Any, mc_out: Any, settings: dict):
    # Use the same run folder if export.py created it
    outdir = settings.get("current_run_folder", settings.get("outdir", "results"))

    if mc_out and hasattr(mc_out, "samples"):
        ax = mc_out.samples["K_sim"].plot(kind="hist", bins=40, alpha=0.7)
        ax.set_title("Simulated Spring Rate Distribution")
        ax.set_xlabel("K_sim")
        fig = ax.get_figure()

        fig.savefig(os.path.join(outdir, "mc_hist.png"), dpi=150)
        plt.close(fig)

        return {"mc_hist": os.path.join(outdir, "mc_hist.png")}
    return {}
