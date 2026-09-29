from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from demand_inventory_lab import paths
from demand_inventory_lab.inventory import CostAssumptions, LeadTime, Policy, ScenarioGridConfig, Shock, run_scenario_grid, shock_comparison
from demand_inventory_lab.io import load_sku_table, select_sku_demand

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_fixture  # noqa: E402
import make_sku_fixture  # noqa: E402

COSTS = CostAssumptions(unit_cost=10.0, holding_rate_per_period=0.005, order_cost=50.0, stockout_penalty_per_unit=4.0)


@pytest.fixture(scope="module")
def sku_demand():
    return select_sku_demand(load_sku_table(paths.SKU_FIXTURE), "SYN-SKU-0001", "SYN-STORE-01")


def test_grid_has_one_cell_per_reorder_point_lead_time_pair(sku_demand):
    cfg = ScenarioGridConfig(reorder_points=(40, 80, 120), lead_times=(1, 2), order_quantity=100)
    out = run_scenario_grid(sku_demand, cfg, COSTS)
    assert out["schemaVersion"] == 1 and out["mode"] == "synthetic-demo"
    assert len(out["grid"]) == 6
    pairs = {(c["reorder_point"], c["lead_time"]) for c in out["grid"]}
    assert pairs == {(s, l) for s in (40, 80, 120) for l in (1, 2)}
    for c in out["grid"]:
        assert {"fill_rate", "holding_cost", "stockout_periods", "order_count", "period_service_level"} <= set(c)


def test_scenario_output_is_reproducible(sku_demand):
    cfg = ScenarioGridConfig(reorder_points=(40, 120), lead_times=(1, 3), order_quantity=100, seed=42)
    a = json.dumps(run_scenario_grid(sku_demand, cfg, COSTS), sort_keys=True)
    b = json.dumps(run_scenario_grid(sku_demand, cfg, COSTS), sort_keys=True)
    assert a == b


def test_fill_rate_does_not_decrease_with_reorder_point_for_fixed_lead_time(sku_demand):
    cfg = ScenarioGridConfig(reorder_points=(20, 60, 100, 140, 180), lead_times=(2,), order_quantity=100)
    grid = run_scenario_grid(sku_demand, cfg, COSTS)["grid"]
    rates = [c["fill_rate"] for c in sorted(grid, key=lambda c: c["reorder_point"])]
    assert all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))
    holding = [c["holding_cost"] for c in sorted(grid, key=lambda c: c["reorder_point"])]
    assert holding[-1] > holding[0]


def test_shock_comparison_reports_more_demand_when_shocked(sku_demand):
    pol = Policy(reorder_point=100, order_up_to=220)
    out = shock_comparison(sku_demand, pol, LeadTime(fixed=2), COSTS, Shock(60, 68, 1.5))
    assert out["shocked"]["units_demanded"] > out["baseline"]["units_demanded"]
    assert out["shock"] == {"start": 60, "end": 68, "multiplier": 1.5}


def test_scenario_config_validation():
    with pytest.raises(Exception):
        ScenarioGridConfig(reorder_points=(), lead_times=(1,), order_quantity=10)
    with pytest.raises(Exception):
        ScenarioGridConfig(reorder_points=(1,), lead_times=(1,), order_quantity=0)


def test_mrts_fixture_generator_is_deterministic_and_fictional():
    a = make_fixture.build(42)
    b = make_fixture.build(42)
    assert a.equals(b)
    assert len(a) == 360 and a["category_code"].nunique() == 3
    assert a["category_code"].str.startswith("SYN-").all()
    assert (a["cell_value"] >= 0).all()
    assert not make_fixture.build(7).equals(a)


def test_sku_fixture_generator_is_deterministic_and_fictional():
    a = make_sku_fixture.build(42)
    assert a.equals(make_sku_fixture.build(42))
    assert a["sku_id"].str.startswith("SYN-SKU-").all() and a["store_id"].str.startswith("SYN-STORE-").all()
    assert (a["demand"] >= 0).all()
    assert a.groupby(["sku_id", "store_id"]).size().eq(104).all()


def test_committed_fixture_matches_generator_and_has_fixture_md():
    on_disk = load_sku_table(paths.SKU_FIXTURE)
    fresh = make_sku_fixture.build(42)
    np.testing.assert_array_equal(on_disk["demand"].to_numpy(), fresh["demand"].to_numpy())
    md = (paths.FIXTURE_DIR / "FIXTURE.md").read_text(encoding="utf-8")
    assert "Authored synthetic demonstration data" in md and "seed `42`" in md
