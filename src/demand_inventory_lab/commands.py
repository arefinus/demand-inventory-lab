"""Command implementations shared by the CLI. Return values are plain dicts so the
CLI layer decides what to print; nothing here prints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .errors import MissingDataError, SchemaError
from .forecast.pipeline import backtest_config_from_dict, build_models, final_forecast, gbm_config_from_dict
from .forecast.backtest import rolling_origin_backtest
from .inventory.policy import CostAssumptions, LeadTime, Policy, Shock
from .inventory.scenarios import ScenarioGridConfig, run_scenario_grid, shock_comparison
from .io import load_mrts_table, load_sku_table, load_yaml, select_series, select_sku_demand
from .manifest import RunManifest, make_run_id, sha256_of_file, sha256_of_mapping, utc_now, write_manifest

DEFAULT_CATEGORY = "SYN-RET-01"
SUMMARY_SCHEMA_VERSION = 1


def _write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _clean(obj):
    """Replace NaN floats with None so JSON stays strict."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def run_backtest(
    out_dir: Path,
    fixture: Path = paths.MRTS_FIXTURE,
    category: str = DEFAULT_CATEGORY,
    config_path: Path = paths.CONFIG_DIR / "backtest.yaml",
    overrides: dict | None = None,
) -> dict:
    cfg = load_yaml(config_path)
    cfg.update(overrides or {})
    bt_cfg = backtest_config_from_dict(cfg)
    gbm_cfg = gbm_config_from_dict(cfg, bt_cfg.max_horizon, bt_cfg.season_length)
    table = load_mrts_table(fixture)
    y, dates = select_series(table, category)
    models = build_models(bt_cfg, gbm_cfg)
    result = rolling_origin_backtest(y, dates, models, bt_cfg)
    forecast = final_forecast(y, dates, models, result)

    records_path = out_dir / "backtest_records.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.records.to_csv(records_path, index=False)
    summary = {
        "schemaVersion": SUMMARY_SCHEMA_VERSION,
        "mode": "synthetic-demo",
        "series": {"fixture": fixture.name, "category_code": category, "n_obs": int(result.n_obs)},
        "config": bt_cfg.to_dict(),
        "gbm_config": gbm_cfg.to_dict(),
        "n_folds": len(result.origins()),
        "metrics_note": "MAE, WAPE, MASE and interval coverage; MAPE omitted by design (see docs/method.md).",
        "metrics": _clean(result.metrics),
    }
    _write_json(out_dir / "backtest_summary.json", summary)
    _write_json(
        out_dir / "forecast.json",
        {
            "schemaVersion": SUMMARY_SCHEMA_VERSION,
            "mode": "synthetic-demo",
            "series": summary["series"],
            "origin": dates[-1].strftime("%Y-%m"),
            "horizon": bt_cfg.max_horizon,
            "interval": {"alpha": bt_cfg.interval_alpha, "method": "empirical residual quantiles per horizon from the backtest"},
            "models": forecast,
        },
    )
    return {"summary": summary, "records_path": records_path, "result": result, "config_hash": sha256_of_mapping(cfg)}


def load_costs(config_path: Path = paths.CONFIG_DIR / "assumptions.yaml") -> CostAssumptions:
    return CostAssumptions.from_mapping(load_yaml(config_path))


def run_simulation(
    out_dir: Path,
    fixture: Path = paths.SKU_FIXTURE,
    grid_config_path: Path = paths.CONFIG_DIR / "scenario_grid.yaml",
    assumptions_path: Path = paths.CONFIG_DIR / "assumptions.yaml",
    overrides: dict | None = None,
) -> dict:
    gcfg = load_yaml(grid_config_path)
    gcfg.update(overrides or {})
    costs = load_costs(assumptions_path)
    table = load_sku_table(fixture)
    demand = select_sku_demand(table, gcfg["sku_id"], gcfg["store_id"])
    grid_cfg = ScenarioGridConfig.from_mapping(gcfg)
    grid = run_scenario_grid(demand, grid_cfg, costs)
    grid["series"] = {"fixture": fixture.name, "sku_id": gcfg["sku_id"], "store_id": gcfg["store_id"]}
    _write_json(out_dir / "scenario_grid.json", grid)

    sh = gcfg.get("shock") or {}
    comparison = None
    if sh:
        shock = Shock(int(sh["start"]), int(sh["end"]), float(sh["multiplier"]))
        mid = grid_cfg.reorder_points[len(grid_cfg.reorder_points) // 2]
        policy = Policy(reorder_point=mid, order_up_to=mid + grid_cfg.order_quantity, review_period=grid_cfg.review_period)
        comparison = shock_comparison(demand, policy, LeadTime(fixed=grid_cfg.lead_times[0]), costs, shock, seed=grid_cfg.seed)
        _write_json(out_dir / "shock_comparison.json", comparison)
    return {"grid": grid, "comparison": comparison, "config_hash": sha256_of_mapping({**gcfg, "costs": costs.to_dict()})}


def evaluate_records(records_path: Path) -> dict:
    """Recompute the metric summary from a saved backtest records table."""
    from .forecast.backtest import RECORD_COLUMNS, summarise

    if not records_path.exists():
        raise MissingDataError(f"{records_path} not found. Run `python -m demand_inventory_lab backtest` first.")
    df = pd.read_csv(records_path)
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{records_path} is missing columns {missing}.")
    if df.empty:
        raise SchemaError(f"{records_path} has no rows.")
    return _clean(summarise(df))


def check_outputs(out_dir: Path) -> list[str]:
    """Schema checks used by smoke. Returns a list of problems (empty means ok)."""
    problems: list[str] = []
    for name in ("backtest_summary.json", "forecast.json", "scenario_grid.json"):
        p = out_dir / name
        if not p.exists():
            problems.append(f"missing {p}")
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("schemaVersion") != SUMMARY_SCHEMA_VERSION or obj.get("mode") != "synthetic-demo":
            problems.append(f"{name}: schemaVersion/mode mismatch")
        if name == "scenario_grid.json":
            grid = obj.get("grid", [])
            if not grid:
                problems.append("scenario_grid.json: empty grid")
            for cell in grid:
                for k in ("reorder_point", "lead_time", "fill_rate", "holding_cost"):
                    if k not in cell:
                        problems.append(f"scenario_grid.json: cell missing {k}")
                        break
        if name == "backtest_summary.json":
            for m in ("seasonal_naive", "holt_winters", "lag_gbm"):
                if m not in obj.get("metrics", {}):
                    problems.append(f"backtest_summary.json: model {m} missing")
    if not (out_dir / "backtest_records.csv").exists():
        problems.append("missing backtest_records.csv")
    return problems


def write_run_manifest(out_dir: Path, mode: str, status: str, started: str, config_hash: str, seed: int, notes: tuple[str, ...] = ()) -> Path:
    data = {
        "source_id": "synthetic-fixture-v1",
        "version": "v1",
        "split_manifest_hash": sha256_of_file(paths.MRTS_FIXTURE) if paths.MRTS_FIXTURE.exists() else None,
        "sku_fixture_hash": sha256_of_file(paths.SKU_FIXTURE) if paths.SKU_FIXTURE.exists() else None,
        "sample_counts": {},
    }
    if paths.MRTS_FIXTURE.exists():
        data["sample_counts"]["mrts_rows"] = int(sum(1 for _ in paths.MRTS_FIXTURE.open(encoding="utf-8")) - 1)
    if paths.SKU_FIXTURE.exists():
        data["sample_counts"]["sku_rows"] = int(sum(1 for _ in paths.SKU_FIXTURE.open(encoding="utf-8")) - 1)
    m = RunManifest(
        run_id=make_run_id(mode, started),
        mode=mode,
        status=status,
        started_at=started,
        finished_at=utc_now(),
        configuration_hash=config_hash,
        seed=seed,
        data=data,
        metrics_file=_relative(out_dir / "backtest_summary.json"),
        predictions_file=_relative(out_dir / "forecast.json"),
        notes=notes,
    )
    return write_manifest(out_dir / "run_manifest.yaml", m)


def _relative(p: Path) -> str | None:
    """Repository-relative POSIX path, or the bare file name when the output sits
    outside the repository. Absolute machine paths never enter a committed file."""
    if not p.exists():
        return None
    try:
        return p.resolve().relative_to(paths.REPO_ROOT).as_posix()
    except ValueError:
        return p.name
