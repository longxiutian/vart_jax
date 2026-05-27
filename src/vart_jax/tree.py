from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.nn import log_sigmoid
from jax.scipy.special import logsumexp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VartTreeSpec:
    """Static complete-tree specification arrays for VaRT's truncated tree algebra.

    Depth is the number of binary decisions along a full terminal path. Thus
    `depth=3` has `b=8` terminal-layer nodes, `b-1=7` internal split nodes,
    and `2b-1=15` candidate leaf nodes because internal nodes may stop early.
    """

    depth: int
    b: int
    num_internal: int
    num_candidates: int
    candidate_node_ids: jnp.ndarray
    internal_node_ids: jnp.ndarray
    path_parents: jnp.ndarray
    path_signs: jnp.ndarray
    path_mask: jnp.ndarray
    ancestor_mask: jnp.ndarray
    internal_candidate_mask: jnp.ndarray
    terminal_candidate_mask: jnp.ndarray
    self_internal_index: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.candidate_node_ids,
            self.internal_node_ids,
            self.path_parents,
            self.path_signs,
            self.path_mask,
            self.ancestor_mask,
            self.internal_candidate_mask,
            self.terminal_candidate_mask,
            self.self_internal_index,
        )
        aux_data = (self.depth, self.b, self.num_internal, self.num_candidates)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        depth, b, num_internal, num_candidates = aux_data
        (
            candidate_node_ids,
            internal_node_ids,
            path_parents,
            path_signs,
            path_mask,
            ancestor_mask,
            internal_candidate_mask,
            terminal_candidate_mask,
            self_internal_index,
        ) = children
        return cls(
            depth=depth,
            b=b,
            num_internal=num_internal,
            num_candidates=num_candidates,
            candidate_node_ids=candidate_node_ids,
            internal_node_ids=internal_node_ids,
            path_parents=path_parents,
            path_signs=path_signs,
            path_mask=path_mask,
            ancestor_mask=ancestor_mask,
            internal_candidate_mask=internal_candidate_mask,
            terminal_candidate_mask=terminal_candidate_mask,
            self_internal_index=self_internal_index,
        )


@lru_cache(maxsize=None)
def make_tree_arrays(depth: int) -> VartTreeSpec:
    """Construct complete-tree bookkeeping arrays with broadcasted JAX ops.

    This is a static factory: it depends only on integer depth, not model
    parameters. It is cached by depth so SVI does not repeatedly rebuild the
    same structural arrays.
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")

    b = 2**depth
    num_internal = b - 1
    num_candidates = 2 * b - 1
    node_ids = jnp.arange(1, num_candidates + 1, dtype=jnp.int32)
    cols = jnp.arange(depth, dtype=jnp.int32)

    levels = jnp.floor(jnp.log2(node_ids.astype(jnp.float32))).astype(jnp.int32)
    path_mask_bool = cols[None, :] < levels[:, None]
    path_mask = path_mask_bool.astype(jnp.float32)

    parent_shift = jnp.maximum(levels[:, None] - cols[None, :], 0)
    path_parent_ids = node_ids[:, None] // (2**parent_shift)
    path_parents = jnp.where(path_mask_bool, path_parent_ids - 1, -1).astype(jnp.int32)

    bit_shift = jnp.maximum(levels[:, None] - cols[None, :] - 1, 0)
    branch_bits = (node_ids[:, None] // (2**bit_shift)) % 2
    path_signs = jnp.where(branch_bits == 0, 1.0, -1.0)
    path_signs = jnp.where(path_mask_bool, path_signs, 0.0).astype(jnp.float32)

    internal_indices = jnp.arange(num_internal, dtype=jnp.int32)
    ancestor_mask = jnp.any(path_parents[:, :, None] == internal_indices[None, None, :], axis=1)
    ancestor_mask = ancestor_mask.astype(jnp.float32)

    internal_candidate_mask = (node_ids <= num_internal).astype(jnp.float32)
    terminal_candidate_mask = (node_ids > num_internal).astype(jnp.float32)
    self_internal_index = jnp.where(node_ids <= num_internal, node_ids - 1, -1).astype(jnp.int32)

    return VartTreeSpec(
        depth=depth,
        b=b,
        num_internal=num_internal,
        num_candidates=num_candidates,
        candidate_node_ids=node_ids,
        internal_node_ids=jnp.arange(1, num_internal + 1, dtype=jnp.int32),
        path_parents=path_parents,
        path_signs=path_signs,
        path_mask=path_mask,
        ancestor_mask=ancestor_mask,
        internal_candidate_mask=internal_candidate_mask,
        terminal_candidate_mask=terminal_candidate_mask,
        self_internal_index=self_internal_index,
    )


@jax.jit
def structural_leaf_logits(split: jnp.ndarray, arrays: VartTreeSpec, eps: float = 1e-6) -> jnp.ndarray:
    """Return log structural leaf indicators for every candidate leaf node."""

    split = jnp.clip(split, eps, 1.0 - eps)
    log_split = jnp.log(split)
    log_ancestor_active = arrays.ancestor_mask @ log_split

    safe_self = jnp.maximum(arrays.self_internal_index, 0)
    stop_logit = jnp.log1p(-split[safe_self])
    internal_log = log_ancestor_active + stop_logit
    terminal_log = log_ancestor_active
    return jnp.where(arrays.internal_candidate_mask > 0, internal_log, terminal_log)


@jax.jit
def internal_node_logits(split: jnp.ndarray, arrays: VartTreeSpec, eps: float = 1e-6) -> jnp.ndarray:
    """Return log indicators that internal candidate nodes are active split nodes."""

    split = jnp.clip(split, eps, 1.0 - eps)
    log_split = jnp.log(split)
    log_ancestor_active = arrays.ancestor_mask[: arrays.num_internal] @ log_split
    return log_ancestor_active + log_split


@jax.jit
def log_path_probabilities(X: jnp.ndarray, beta: jnp.ndarray, arrays: VartTreeSpec) -> jnp.ndarray:
    """Log probability that each row reaches every candidate node.

    Left branches use sigmoid(beta_i' x); right branches use sigmoid(-beta_i' x).
    """

    scores = X @ beta.T
    safe_parents = jnp.maximum(arrays.path_parents, 0)
    gathered = jnp.take(scores, safe_parents, axis=1)
    branch_log_probs = log_sigmoid(gathered * arrays.path_signs[None, :, :])
    return jnp.sum(branch_log_probs * arrays.path_mask[None, :, :], axis=-1)


@jax.jit
def log_leaf_mixture_weights(
    X: jnp.ndarray,
    split: jnp.ndarray,
    beta: jnp.ndarray,
    arrays: VartTreeSpec,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Normalized log mixture weights over candidate leaves for each observation."""

    logits = structural_leaf_logits(split, arrays, eps=eps)[None, :] + log_path_probabilities(X, beta, arrays)
    return logits - logsumexp(logits, axis=1, keepdims=True)
