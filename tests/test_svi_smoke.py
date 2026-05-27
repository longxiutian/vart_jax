import jax.numpy as jnp
from jax import random

from vart_jax.model import fit_vart, posterior_predictive, standardize_train_test
from vart_jax.simulation import simulate_piecewise_regression


def test_svi_smoke_tiny_dataset():
    sim = simulate_piecewise_regression(random.PRNGKey(0), n_train=32, n_test=12, n_features=4, noise_scale=0.2)
    X_train, X_test, _, _ = standardize_train_test(sim.X_train, sim.X_test)
    fit = fit_vart(random.PRNGKey(1), X_train, sim.y_train, depth=2, steps=8, learning_rate=0.02, temperature=0.7)
    assert fit.losses.shape == (8,)
    assert jnp.all(jnp.isfinite(fit.losses))
    pred = posterior_predictive(random.PRNGKey(2), fit.params, X_test, depth=2, temperature=0.7, num_samples=5)
    assert pred["mean"].shape == (12,)
    assert pred["lower"].shape == (12,)
    assert pred["upper"].shape == (12,)
    assert jnp.all(jnp.isfinite(pred["mean"]))
