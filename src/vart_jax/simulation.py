from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random


@dataclass(frozen=True)
class SimData:
    X_train: jax.Array
    y_train: jax.Array
    f_train: jax.Array
    region_train: jax.Array
    X_test: jax.Array
    y_test: jax.Array
    f_test: jax.Array
    region_test: jax.Array


def _piecewise_mean(X: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    region = jnp.where(x0 <= -0.35, 0, jnp.where(x1 <= 0.25, 1, 2))
    f0 = -1.5 + 1.10 * x1 - 0.25 * x2
    f1 = 0.55 - 1.15 * x0 + 0.65 * x2
    f2 = 1.75 + 0.70 * x0 - 0.40 * x1 + 0.15 * x3
    f = jnp.choose(region, (f0, f1, f2))
    return f, region


def simulate_piecewise_regression(
    rng_key: jax.Array,
    *,
    n_train: int = 400,
    n_test: int = 200,
    n_features: int = 5,
    noise_scale: float = 0.25,
) -> SimData:
    """Generate a piecewise-linear regression problem with known latent regions."""

    if n_features < 4:
        raise ValueError("n_features must be at least 4")
    n = n_train + n_test
    x_key, noise_key = random.split(rng_key)
    X = random.normal(x_key, shape=(n, n_features))
    f, region = _piecewise_mean(X)
    y = f + noise_scale * random.normal(noise_key, shape=(n,))
    return SimData(
        X_train=X[:n_train],
        y_train=y[:n_train],
        f_train=f[:n_train],
        region_train=region[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
        f_test=f[n_train:],
        region_test=region[n_train:],
    )
