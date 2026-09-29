from __future__ import annotations

import json

import pytest
import yaml

from demand_inventory_lab.cli import main


def test_smoke_runs_offline_and_writes_manifest(tmp_path, capsys):
    out = tmp_path / "smoke"
    code = main(["smoke", "--out", str(out)])
    smoke_out = capsys.readouterr().out
    assert code == 0, smoke_out
    assert "smoke ok" in smoke_out
    manifest = yaml.safe_load((out / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["project_id"] == "demand-inventory-lab"
    assert manifest["status"] == "completed" and manifest["mode"] == "smoke"
    assert manifest["evidence_status"] == "demo" and manifest["git_commit"] is None
    grid = json.loads((out / "scenario_grid.json").read_text(encoding="utf-8"))
    assert grid["mode"] == "synthetic-demo" and grid["grid"]
    summary = json.loads((out / "backtest_summary.json").read_text(encoding="utf-8"))
    assert set(summary["metrics"]) == {"seasonal_naive", "holt_winters", "lag_gbm"}
    # evaluate recomputes the same summary from the saved records
    code = main(["evaluate", "--records", str(out / "backtest_records.csv")])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["lag_gbm"]["overall"]["mae"] == pytest.approx(summary["metrics"]["lag_gbm"]["overall"]["mae"])


def test_missing_inputs_exit_with_code_2(tmp_path, capsys):
    assert main(["evaluate", "--records", str(tmp_path / "nope.csv")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["backtest", "--fixture", str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2
    assert main(["simulate", "--fixture", str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2


def test_backtest_rejects_unknown_category_with_exit_1(tmp_path, capsys):
    code = main(["backtest", "--category", "NOPE", "--out", str(tmp_path), "--min-train", "96", "--step", "12"])
    assert code == 1
    assert "available" in capsys.readouterr().err
