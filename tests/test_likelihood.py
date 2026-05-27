import jax.numpy as jnp
import numpyro.distributions as dist

from vart_jax.likelihood import leaf_mixture_distribution, mixture_log_likelihood
from vart_jax.tree import make_tree_arrays


def _toy_inputs():
    arrays = make_tree_arrays(2)
    X = jnp.array([[1.0, -0.5], [1.0, 0.5]])
    y = jnp.array([0.1, -0.2])
    split = jnp.full((arrays.num_internal,), 0.5)
    beta = jnp.zeros((arrays.num_internal, 2))
    theta = jnp.zeros((arrays.num_candidates, 2))
    leaf_scale = jnp.ones((arrays.num_candidates,))
    return arrays, X, y, split, beta, theta, leaf_scale


def test_leaf_mixture_distribution_uses_numpyro_mixture_same_family():
    arrays, X, _, split, beta, theta, leaf_scale = _toy_inputs()
    mixture = leaf_mixture_distribution(X, split, beta, theta, leaf_scale, arrays)
    assert isinstance(mixture, dist.MixtureSameFamily)
    assert mixture.batch_shape == (2,)
    assert mixture.component_distribution.batch_shape == (2, arrays.num_candidates)


def test_mixture_log_likelihood_is_finite():
    arrays, X, y, split, beta, theta, leaf_scale = _toy_inputs()
    ll = mixture_log_likelihood(X, y, split, beta, theta, leaf_scale, arrays)
    assert ll.shape == (2,)
    assert jnp.all(jnp.isfinite(ll))
