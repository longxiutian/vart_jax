from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from vart_jax.simulation import SimData


def _features_frame(X, prefix: str = "x") -> pd.DataFrame:
    return pd.DataFrame(np.asarray(X), columns=[f"{prefix}{j}" for j in range(X.shape[1])])


def write_simulation_duckdb(
    db_path: Path,
    *,
    sim: SimData,
    losses,
    metrics: dict[str, float | int],
    predictions: dict,
    config: dict[str, float | int],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        config_df = pd.DataFrame([config])
        metrics_df = pd.DataFrame([metrics])
        loss_df = pd.DataFrame({"step": np.arange(len(losses)), "loss": np.asarray(losses, dtype=float)})

        train_df = _features_frame(sim.X_train)
        train_df.insert(0, "split", "train")
        train_df.insert(1, "row_id", np.arange(len(train_df)))
        train_df["y"] = np.asarray(sim.y_train)
        train_df["f_true"] = np.asarray(sim.f_train)
        train_df["region"] = np.asarray(sim.region_train)

        test_df = _features_frame(sim.X_test)
        test_df.insert(0, "split", "test")
        test_df.insert(1, "row_id", np.arange(len(test_df)))
        test_df["y"] = np.asarray(sim.y_test)
        test_df["f_true"] = np.asarray(sim.f_test)
        test_df["region"] = np.asarray(sim.region_test)

        pred_df = pd.DataFrame(
            {
                "row_id": np.arange(len(predictions["mean"])),
                "pred_mean": np.asarray(predictions["mean"]),
                "pred_lower": np.asarray(predictions["lower"]),
                "pred_upper": np.asarray(predictions["upper"]),
                "y": np.asarray(sim.y_test),
                "f_true": np.asarray(sim.f_test),
                "region": np.asarray(sim.region_test),
            }
        )

        con.register("config_df", config_df)
        con.register("metrics_df", metrics_df)
        con.register("loss_df", loss_df)
        con.register("observations_df", pd.concat([train_df, test_df], ignore_index=True))
        con.register("predictions_df", pred_df)
        con.execute("CREATE OR REPLACE TABLE config AS SELECT * FROM config_df")
        con.execute("CREATE OR REPLACE TABLE metrics AS SELECT * FROM metrics_df")
        con.execute("CREATE OR REPLACE TABLE loss_trace AS SELECT * FROM loss_df")
        con.execute("CREATE OR REPLACE TABLE observations AS SELECT * FROM observations_df")
        con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM predictions_df")
    finally:
        con.close()
