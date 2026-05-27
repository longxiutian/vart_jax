from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from vart_jax.tree import VartTreeSpec, log_leaf_mixture_weights


def leaf_mixture_distribution(
    X: jnp.ndarray,
    split: jnp.ndarray,
    beta: jnp.ndarray,
    theta: jnp.ndarray,
    leaf_scale: jnp.ndarray,
    arrays: VartTreeSpec,
) -> dist.MixtureSameFamily:
    """Vectorized Normal mixture over VaRT candidate leaves."""

    log_weights = log_leaf_mixture_weights(X, split, beta, arrays)
    means = X @ theta.T
    mixing = dist.CategoricalLogits(logits=log_weights)
    components = dist.Normal(means, leaf_scale[None, :])
    return dist.MixtureSameFamily(mixing, components)


@jax.jit
def mixture_log_likelihood(
    X: jnp.ndarray,
    y: jnp.ndarray,
    split: jnp.ndarray,
    beta: jnp.ndarray,
    theta: jnp.ndarray,
    leaf_scale: jnp.ndarray,
    arrays: VartTreeSpec,
) -> jnp.ndarray:
    """Observation-wise finite-mixture log likelihood."""

    return leaf_mixture_distribution(X, split, beta, theta, leaf_scale, arrays).log_prob(y)


@jax.jit
def mixture_moments(
    X: jnp.ndarray,
    split: jnp.ndarray,
    beta: jnp.ndarray,
    theta: jnp.ndarray,
    leaf_scale: jnp.ndarray,
    arrays: VartTreeSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and variance of the leaf-mixture predictive distribution."""

    mixture = leaf_mixture_distribution(X, split, beta, theta, leaf_scale, arrays)
    return mixture.mean, jnp.maximum(mixture.variance, 1e-8)
