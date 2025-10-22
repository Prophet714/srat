"""
flexseal package

Convenience imports so callers can just do:
    from flexseal import compute_baseline,
                        fit_hierarchical_residual_model,
                        run_mc_and_sensitivity,
                        make_summary_plots,
                        write_results
"""

# Import only the top-level API functions directly, not during
# module initialisation of the submodules that depend on each other.
# This prevents circular-import errors.

from .regression import fit_hierarchical_residual_model
from .plotting import make_summary_plots
from .export import write_results

# physics_model is optional – expose compute_baseline if it exists
try:
    from .physics_model import compute_baseline
except ImportError:
    compute_baseline = None

# Monte-Carlo routine can be imported lazily so it doesn’t trigger
# a circular import at package import time.  Access it as:
#     from flexseal import run_mc_and_sensitivity
# or just import from flexseal.montecarlo where needed.
try:
    from .montecarlo import run_mc_and_sensitivity
except ImportError:
    run_mc_and_sensitivity = None
