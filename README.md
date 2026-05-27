# vart-jax

A compact single-tree implementation of **VaRT: Variational Regression Trees** in JAX and NumPyro.

The v1 implementation focuses on the paper's core object: a truncated stochastic regression tree with relaxed Bernoulli split indicators, Gaussian split and leaf parameters, and a NumPyro `MixtureSameFamily` likelihood inside SVI.

## Quick start

```bash
uv run python -m vart_jax.check_gpu
uv run pytest
uv run python main.py --depth 3 --steps 2000
# Equivalent installed script:
uv run vart-sim --depth 3 --steps 2000
```

Simulation outputs are written to:

- `data/vart_sim.duckdb`
- `reports/loss_trace.png`
- `reports/predictions_true_vs_estimated.png`

## Model sketch

For a truncation depth `d`, let `b = 2**d`. Internal nodes are indexed `1..b-1`; candidate leaf nodes are indexed `1..2b-1`, because any internal node may stop early and become a leaf. Split indicators are represented by a continuous Concrete/relaxed Bernoulli variable during SVI. Given split-rule vectors `beta`, candidate leaf linear parameters `theta`, and leaf scales, the likelihood is the normalized finite mixture

```text
p(y_n | x_n) = sum_i w_i(x_n, s, beta) Normal(y_n | x_n' theta_i, sigma_i),
```

where `w_i` combines structural leaf indicators with stochastic path probabilities.

## CLI style

The CLI is implemented with standard-library `argparse` in `src/vart_jax/main.py`. The repository root `main.py` delegates to that package entrypoint, so `uv run python main.py ...` and `uv run vart-sim ...` exercise the same code.
