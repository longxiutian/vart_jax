from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_reports(report_dir: Path, *, losses, y_true, f_true, pred_mean, lower, upper) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    losses_np = np.asarray(losses, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(losses_np)), losses_np, lw=1.5)
    ax.set_title("VaRT SVI loss")
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    fig.tight_layout()
    fig.savefig(report_dir / "loss_trace.png", dpi=160)
    plt.close(fig)

    y_np = np.asarray(y_true)
    f_np = np.asarray(f_true)
    pred_np = np.asarray(pred_mean)
    lower_np = np.asarray(lower)
    upper_np = np.asarray(upper)
    order = np.argsort(f_np)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(f_np, pred_np, s=18, alpha=0.75)
    lo = min(float(f_np.min()), float(pred_np.min()))
    hi = max(float(f_np.max()), float(pred_np.max()))
    axes[0].plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
    axes[0].set_title("True mean vs posterior mean")
    axes[0].set_xlabel("true f(x)")
    axes[0].set_ylabel("posterior mean")

    axes[1].plot(f_np[order], color="black", lw=1.5, label="true f(x)")
    axes[1].plot(pred_np[order], color="tab:blue", lw=1.5, label="posterior mean")
    axes[1].fill_between(np.arange(len(order)), lower_np[order], upper_np[order], color="tab:blue", alpha=0.18, label="90% pred interval")
    axes[1].scatter(np.arange(len(order)), y_np[order], s=8, color="tab:gray", alpha=0.45, label="y")
    axes[1].set_title("Sorted test predictions")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "predictions_true_vs_estimated.png", dpi=160)
    plt.close(fig)
