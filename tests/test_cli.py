from pathlib import Path

import duckdb

from vart_jax.main import main


def test_cli_writes_duckdb_and_reports(tmp_path: Path):
    db_path = tmp_path / "vart_sim.duckdb"
    report_dir = tmp_path / "reports"
    exit_code = main(
        [
            "--seed", "3",
            "--depth", "2",
            "--steps", "6",
            "--n-train", "32",
            "--n-test", "12",
            "--n-features", "4",
            "--posterior-samples", "5",
            "--db-path", str(db_path),
            "--report-dir", str(report_dir),
            "--no-require-gpu",
        ]
    )
    assert exit_code == 0
    assert db_path.exists()
    assert (report_dir / "loss_trace.png").exists()
    assert (report_dir / "predictions_true_vs_estimated.png").exists()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert {"config", "metrics", "loss_trace", "observations", "predictions"}.issubset(tables)
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 12
    finally:
        con.close()
