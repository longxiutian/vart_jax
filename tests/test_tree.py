import jax.numpy as jnp

from vart_jax.tree import log_leaf_mixture_weights, log_path_probabilities, make_tree_arrays, structural_leaf_logits


def test_tree_arrays_depth_two_indexing():
    arrays = make_tree_arrays(2)
    assert arrays.b == 4
    assert arrays.num_internal == 3
    assert arrays.num_candidates == 7
    assert arrays.candidate_node_ids.tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_structural_leaf_logits_root_stop():
    arrays = make_tree_arrays(2)
    logits = structural_leaf_logits(jnp.array([1e-6, 0.5, 0.5]), arrays)
    weights = jnp.exp(logits)
    assert weights[0] > 0.999
    assert jnp.all(weights[1:] < 1e-3)


def test_path_probabilities_zero_beta_are_halves():
    arrays = make_tree_arrays(2)
    X = jnp.ones((1, 3))
    beta = jnp.zeros((arrays.num_internal, 3))
    probs = jnp.exp(log_path_probabilities(X, beta, arrays))[0]
    expected = jnp.array([1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(probs, expected, atol=1e-6)


def test_leaf_mixture_weights_are_normalized():
    arrays = make_tree_arrays(3)
    X = jnp.ones((5, 4))
    split = jnp.full((arrays.num_internal,), 0.7)
    beta = jnp.zeros((arrays.num_internal, 4))
    log_weights = log_leaf_mixture_weights(X, split, beta, arrays)
    assert jnp.allclose(jnp.sum(jnp.exp(log_weights), axis=1), 1.0, atol=1e-6)
