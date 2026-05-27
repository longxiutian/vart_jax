"""JAX/NumPyro implementation of single-tree Variational Regression Trees."""

from vart_jax.model import FitResult, fit_vart, posterior_predictive, vart_guide, vart_model
from vart_jax.simulation import simulate_piecewise_regression
from vart_jax.tree import VartTreeSpec, make_tree_arrays

__all__ = [
    "FitResult",
    "VartTreeSpec",
    "fit_vart",
    "make_tree_arrays",
    "posterior_predictive",
    "simulate_piecewise_regression",
    "vart_guide",
    "vart_model",
]
