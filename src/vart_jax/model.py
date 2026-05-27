from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.optim import Adam

from vart_jax.likelihood import leaf_mixture_distribution, mixture_moments
from vart_jax.tree import make_tree_arrays


@dataclass(frozen=True)
class FitResult:
    params: dict
    losses: jnp.ndarray
    depth: int
    temperature: float
    train_mean: jnp.ndarray
    train_scale: jnp.ndarray


def vart_model(
    X: jnp.ndarray,
    y: jnp.ndarray | None = None,
    *,
    depth: int = 3,
    temperature: float = 0.5,
    split_prior_prob: float = 0.65,
    beta_prior_scale: float = 1.0,
    theta_prior_scale: float = 1.5,
    leaf_scale_prior_loc: float = -0.7,
    leaf_scale_prior_scale: float = 0.5,
) -> None:
    """Single-tree VaRT model with a relaxed Bernoulli tree structure."""

    arrays = make_tree_arrays(depth)
    p = X.shape[1]

    split = numpyro.sample(
        "split",
        dist.RelaxedBernoulli(temperature, probs=jnp.full((arrays.num_internal,), split_prior_prob)).to_event(1),
    )
    beta = numpyro.sample(
        "beta",
        dist.Normal(0.0, beta_prior_scale).expand((arrays.num_internal, p)).to_event(2),
    )
    theta = numpyro.sample(
        "theta",
        dist.Normal(0.0, theta_prior_scale).expand((arrays.num_candidates, p)).to_event(2),
    )
    leaf_scale = numpyro.sample(
        "leaf_scale",
        dist.LogNormal(leaf_scale_prior_loc, leaf_scale_prior_scale).expand((arrays.num_candidates,)).to_event(1),
    )

    y_dist = leaf_mixture_distribution(X, split, beta, theta, leaf_scale, arrays)
    with numpyro.plate("observations", X.shape[0]):
        numpyro.sample("y", y_dist, obs=y)


def vart_guide(
    X: jnp.ndarray,
    y: jnp.ndarray | None = None,
    *,
    depth: int = 3,
    temperature: float = 0.5,
    **_: object,
) -> None:
    """Mean-field guide for single-tree VaRT."""

    del y
    arrays = make_tree_arrays(depth)
    p = X.shape[1]

    split_logits = numpyro.param("q_split_logits", jnp.zeros((arrays.num_internal,)))
    numpyro.sample("split", dist.RelaxedBernoulliLogits(temperature, logits=split_logits).to_event(1))

    beta_loc = numpyro.param("q_beta_loc", jnp.zeros((arrays.num_internal, p)))
    beta_scale = numpyro.param(
        "q_beta_scale",
        jnp.full((arrays.num_internal, p), 0.25),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(2))

    theta_loc = numpyro.param("q_theta_loc", jnp.zeros((arrays.num_candidates, p)))
    theta_scale = numpyro.param(
        "q_theta_scale",
        jnp.full((arrays.num_candidates, p), 0.35),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("theta", dist.Normal(theta_loc, theta_scale).to_event(2))

    leaf_scale_loc = numpyro.param("q_leaf_scale_loc", jnp.full((arrays.num_candidates,), -0.7))
    leaf_scale_scale = numpyro.param(
        "q_leaf_scale_scale",
        jnp.full((arrays.num_candidates,), 0.20),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("leaf_scale", dist.LogNormal(leaf_scale_loc, leaf_scale_scale).to_event(1))


def standardize_train_test(X_train: jnp.ndarray, X_test: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = jnp.mean(X_train, axis=0)
    scale = jnp.std(X_train, axis=0)
    scale = jnp.where(scale < 1e-6, 1.0, scale)
    return (X_train - mean) / scale, (X_test - mean) / scale, mean, scale


def fit_vart(
    rng_key: jax.Array,
    X: jnp.ndarray,
    y: jnp.ndarray,
    *,
    depth: int = 3,
    steps: int = 2000,
    learning_rate: float = 0.03,
    temperature: float = 0.5,
    progress_every: int = 0,
) -> FitResult:
    """Fit VaRT with NumPyro SVI and return optimized variational params."""

    svi = SVI(
        vart_model,
        vart_guide,
        Adam(learning_rate),
        Trace_ELBO(num_particles=1),
    )
    state = svi.init(rng_key, X, y, depth=depth, temperature=temperature)
    losses = []
    for step in range(steps):
        state, loss = svi.update(state, X, y, depth=depth, temperature=temperature)
        losses.append(loss)
        if progress_every and (step + 1) % progress_every == 0:
            print(f"step {step + 1:5d} | loss {float(loss):.3f}")

    params = svi.get_params(state)
    train_mean = jnp.zeros((X.shape[1],))
    train_scale = jnp.ones((X.shape[1],))
    return FitResult(
        params=params,
        losses=jnp.asarray(losses),
        depth=depth,
        temperature=temperature,
        train_mean=train_mean,
        train_scale=train_scale,
    )


def _moments_for_sample(X: jnp.ndarray, arrays, split, beta, theta, leaf_scale):
    return mixture_moments(X, split, beta, theta, leaf_scale, arrays)


def posterior_predictive(
    rng_key: jax.Array,
    params: dict,
    X: jnp.ndarray,
    *,
    depth: int = 3,
    temperature: float = 0.5,
    num_samples: int = 200,
) -> dict[str, jnp.ndarray]:
    """Draw posterior predictive summaries from the variational guide."""

    arrays = make_tree_arrays(depth)
    predictive = Predictive(
        vart_guide,
        params=params,
        num_samples=num_samples,
        return_sites=("split", "beta", "theta", "leaf_scale"),
    )
    samples = predictive(rng_key, X, None, depth=depth, temperature=temperature)

    vmapped = jax.vmap(jax.tree_util.Partial(_moments_for_sample, X, arrays), in_axes=(0, 0, 0, 0))
    mean_samples, var_samples = vmapped(
        samples["split"],
        samples["beta"],
        samples["theta"],
        samples["leaf_scale"],
    )
    draw_key = random.fold_in(rng_key, 2026)
    y_draws = mean_samples + random.normal(draw_key, shape=mean_samples.shape) * jnp.sqrt(var_samples)
    return {
        "mean_samples": mean_samples,
        "var_samples": var_samples,
        "y_draws": y_draws,
        "mean": jnp.mean(mean_samples, axis=0),
        "lower": jnp.quantile(y_draws, 0.05, axis=0),
        "upper": jnp.quantile(y_draws, 0.95, axis=0),
    }
