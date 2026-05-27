from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
from jax import random

from vart_jax.check_gpu import main as check_gpu_main
from vart_jax.model import fit_vart, posterior_predictive, standardize_train_test
from vart_jax.reporting import write_reports
from vart_jax.simulation import simulate_piecewise_regression
from vart_jax.storage import write_simulation_duckdb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vart-sim",
        description="Run a single-tree VaRT regression simulation with JAX/NumPyro.",
    )
    parser.add_argument("command", nargs="?", choices=("check-gpu",), help="Optional utility command.")
    parser.add_argument("--seed", type=int, default=20260527, help="PRNG seed.")
    parser.add_argument("--depth", type=int, default=3, help="Full tree truncation depth.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of SVI updates.")
    parser.add_argument("--n-train", type=int, default=4000, help="Number of training rows.")
    parser.add_argument("--n-test", type=int, default=2000, help="Number of test rows.")
    parser.add_argument("--n-features", type=int, default=5, help="Number of simulated covariates.")
    parser.add_argument("--noise-scale", type=float, default=0.25, help="Simulation noise scale.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Adam learning rate.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Relaxed Bernoulli temperature.")
    parser.add_argument("--posterior-samples", type=int, default=200, help="Variational posterior predictive samples.")
    parser.add_argument("--db-path", type=Path, default=Path("data/vart_sim.duckdb"), help="DuckDB output path.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="Report output directory.")
    parser.add_argument(
        "--require-gpu",
        dest="require_gpu",
        action="store_true",
        default=True,
        help="Fail unless JAX is GPU-backed. This is the default.",
    )
    parser.add_argument(
        "--no-require-gpu",
        dest="require_gpu",
        action="store_false",
        help="Skip the GPU-backed JAX check.",
    )
    return parser


def run_simulation(args: argparse.Namespace) -> dict[str, float]:
    if args.require_gpu:
        check_gpu_main()

    rng = random.PRNGKey(args.seed)
    sim_key, fit_key, pred_key = random.split(rng, 3)
    sim = simulate_piecewise_regression(
        sim_key,
        n_train=args.n_train,
        n_test=args.n_test,
        n_features=args.n_features,
        noise_scale=args.noise_scale,
    )
    X_train, X_test, _, _ = standardize_train_test(sim.X_train, sim.X_test)
    fit = fit_vart(
        fit_key,
        X_train,
        sim.y_train,
        depth=args.depth,
        steps=args.steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        progress_every=max(args.steps // 10, 1),
    )
    pred = posterior_predictive(
        pred_key,
        fit.params,
        X_test,
        depth=args.depth,
        temperature=args.temperature,
        num_samples=args.posterior_samples,
    )

    rmse_y = float(jnp.sqrt(jnp.mean((pred["mean"] - sim.y_test) ** 2)))
    rmse_f = float(jnp.sqrt(jnp.mean((pred["mean"] - sim.f_test) ** 2)))
    coverage_90 = float(jnp.mean((sim.y_test >= pred["lower"]) & (sim.y_test <= pred["upper"])))
    final_loss = float(fit.losses[-1]) if len(fit.losses) else float("nan")
    metrics = {
        "rmse_y": rmse_y,
        "rmse_f": rmse_f,
        "coverage_90": coverage_90,
        "final_loss": final_loss,
    }
    config = {
        "seed": args.seed,
        "depth": args.depth,
        "steps": args.steps,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "n_features": args.n_features,
        "noise_scale": args.noise_scale,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "posterior_samples": args.posterior_samples,
    }

    write_simulation_duckdb(args.db_path, sim=sim, losses=fit.losses, metrics=metrics, predictions=pred, config=config)
    write_reports(
        args.report_dir,
        losses=fit.losses,
        y_true=sim.y_test,
        f_true=sim.f_test,
        pred_mean=pred["mean"],
        lower=pred["lower"],
        upper=pred["upper"],
    )

    print(f"Final loss: {final_loss:.3f}")
    print(f"RMSE y: {rmse_y:.3f} | RMSE f: {rmse_f:.3f} | 90% coverage: {coverage_90:.3f}")
    print(f"DuckDB: {args.db_path}")
    print(f"Reports: {args.report_dir}")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-gpu":
        check_gpu_main()
        return 0
    run_simulation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
