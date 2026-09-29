from __future__ import annotations

import numpy as np
import pytest

from demand_inventory_lab.errors import ConfigError
from demand_inventory_lab.inventory import CostAssumptions, LeadTime, Policy, Shock, reorder_point, safety_stock, simulate

COSTS = CostAssumptions(unit_cost=10.0, holding_rate_per_period=0.01, order_cost=50.0, stockout_penalty_per_unit=4.0)
BACKORDER = CostAssumptions(unit_cost=10.0, holding_rate_per_period=0.01, order_cost=50.0, stockout_penalty_per_unit=4.0, unmet_demand="backorder")


def _demand(n=60, seed=1):
    return np.random.default_rng(seed).poisson(20, size=n)


def _check_identities(ledger):
    lhs = ledger["on_hand_end"]
    rhs = ledger["on_hand_open"] + ledger["received"] - ledger["backorders_cleared"] - ledger["filled"]
    assert (lhs == rhs).all()
    assert (ledger["on_hand_open"].iloc[1:].to_numpy() == ledger["on_hand_end"].iloc[:-1].to_numpy()).all()
    prev_on_order = ledger["on_order_end"].shift(1).fillna(0).to_numpy()
    assert (ledger["on_order_end"].to_numpy() == prev_on_order + ledger["ordered_qty"].to_numpy() - ledger["received"].to_numpy()).all()
    assert (ledger["filled"] + ledger["unmet"] == ledger["demand"]).all()


@pytest.mark.parametrize("lt", [0, 1, 3])
def test_balance_identity_holds_every_period(lt):
    res = simulate(_demand(), Policy(reorder_point=40, order_up_to=120), LeadTime(fixed=lt), COSTS)
    _check_identities(res.ledger)


def test_balance_identity_holds_in_backorder_mode():
    res = simulate(_demand(seed=3), Policy(reorder_point=20, order_up_to=60), LeadTime(fixed=2), BACKORDER)
    _check_identities(res.ledger)
    assert (res.ledger["backorders_end"] >= 0).all()
    assert res.ledger["backorders_cleared"].sum() > 0


def test_lead_time_zero_arrives_in_the_same_period():
    res = simulate([20, 20], Policy(reorder_point=10, order_up_to=50), LeadTime(fixed=0), COSTS, initial_on_hand=0)
    first = res.ledger.iloc[0]
    assert first["ordered_qty"] == 50 and first["received"] == 50 and first["filled"] == 20 and first["unmet"] == 0


def test_lead_time_one_arrives_next_period():
    res = simulate([20, 20, 20], Policy(reorder_point=10, order_up_to=50), LeadTime(fixed=1), COSTS, initial_on_hand=0)
    led = res.ledger
    assert led.loc[0, "received"] == 0 and led.loc[0, "unmet"] == 20
    assert led.loc[1, "received"] == 50 and led.loc[1, "unmet"] == 0


def test_lead_time_beyond_horizon_never_arrives():
    res = simulate(_demand(n=10), Policy(reorder_point=10, order_up_to=50), LeadTime(fixed=99), COSTS, initial_on_hand=0)
    assert res.ledger["received"].sum() == 0
    assert res.units_filled == 0 and res.fill_rate == 0.0
    assert res.ledger["on_order_end"].iloc[-1] > 0


def test_lead_time_distribution_samples_allowed_values_deterministically():
    lt = LeadTime(distribution=((1, 0.5), (3, 0.5)))
    a = simulate(_demand(), Policy(reorder_point=40, order_up_to=120), lt, COSTS, seed=5)
    b = simulate(_demand(), Policy(reorder_point=40, order_up_to=120), lt, COSTS, seed=5)
    assert a.ledger.equals(b.ledger)
    sampled = set(a.ledger.loc[a.ledger["ordered_qty"] > 0, "lead_time"].tolist())
    assert sampled <= {1, 3} and len(sampled) == 2
    assert lt.mean() == 2.0


def test_shock_multiplies_demand_only_inside_window():
    d = np.full(20, 10)
    shocked = Shock(start=5, end=10, multiplier=2.0).apply(d)
    assert shocked[:5].tolist() == [10] * 5 and shocked[5:10].tolist() == [20] * 5 and shocked[10:].tolist() == [10] * 10
    res = simulate(d, Policy(reorder_point=10, order_up_to=60), LeadTime(fixed=1), COSTS, shock=Shock(5, 10, 2.0))
    assert res.ledger["demand"].tolist() == shocked.tolist()
    with pytest.raises(ConfigError):
        simulate(d, Policy(reorder_point=10, order_up_to=60), LeadTime(fixed=1), COSTS, shock=Shock(5, 30, 2.0))


def test_fill_rate_definition_and_undefined_when_no_demand():
    res = simulate(_demand(), Policy(reorder_point=30, order_up_to=80), LeadTime(fixed=2), COSTS)
    assert 0.0 <= res.fill_rate <= 1.0
    assert res.fill_rate == pytest.approx(res.units_filled / res.units_demanded)
    assert res.stockout_periods == int((res.ledger["unmet"] > 0).sum())
    zero = simulate(np.zeros(10, dtype=int), Policy(reorder_point=5, order_up_to=10), LeadTime(fixed=1), COSTS)
    assert zero.fill_rate is None and zero.stockout_periods == 0


def test_order_count_and_costs_follow_documented_assumptions():
    res = simulate(_demand(), Policy(reorder_point=40, order_up_to=120), LeadTime(fixed=2), COSTS)
    assert res.order_count == int((res.ledger["ordered_qty"] > 0).sum())
    assert res.ordering_cost == pytest.approx(res.order_count * COSTS.order_cost)
    assert res.holding_cost == pytest.approx(res.ledger["on_hand_end"].sum() * COSTS.unit_cost * COSTS.holding_rate_per_period)
    assert res.stockout_cost == pytest.approx(res.ledger["unmet"].sum() * COSTS.stockout_penalty_per_unit)
    assert res.total_cost == pytest.approx(res.holding_cost + res.ordering_cost + res.stockout_cost)


def test_sS_policy_orders_only_when_position_at_or_below_reorder_point():
    pol = Policy(reorder_point=40, order_up_to=120)
    res = simulate(_demand(), pol, LeadTime(fixed=2), COSTS)
    led = res.ledger
    prev_on_order = led["on_order_end"].shift(1).fillna(0)
    prev_bo = led["backorders_end"].shift(1).fillna(0)
    position = led["on_hand_open"] + prev_on_order - prev_bo
    ordered = led["ordered_qty"] > 0
    assert (position[ordered] <= pol.reorder_point).all()
    assert (position[~ordered] > pol.reorder_point).all()
    assert ((position + led["ordered_qty"])[ordered] == pol.order_up_to).all()


def test_review_period_restricts_order_timing():
    res = simulate(_demand(), Policy(reorder_point=200, order_up_to=260, review_period=3), LeadTime(fixed=1), COSTS, initial_on_hand=0)
    periods = res.ledger.loc[res.ledger["ordered_qty"] > 0, "period"]
    assert (periods % 3 == 0).all() and len(periods) > 0


def test_policy_and_lead_time_validation():
    with pytest.raises(ConfigError):
        Policy(reorder_point=50, order_up_to=50)
    with pytest.raises(ConfigError):
        Policy(reorder_point=50, kind="sQ")
    with pytest.raises(ConfigError):
        LeadTime()
    with pytest.raises(ConfigError):
        LeadTime(distribution=((1, 0.7), (2, 0.7)))
    with pytest.raises(ConfigError):
        CostAssumptions(1.0, 0.1, 1.0, 1.0, unmet_demand="ignore")
    with pytest.raises(ConfigError):
        simulate(np.array([]), Policy(reorder_point=1, order_up_to=2), LeadTime(fixed=1), COSTS)


def test_safety_stock_and_reorder_point_formulas():
    ss = safety_stock(z=1.0, demand_std=4.0, lead_time_mean=3.0, review_period=1)
    assert ss == pytest.approx(8.0)
    assert reorder_point(demand_mean=10.0, lead_time_mean=3.0, review_period=1, ss=ss) == 48
    with pytest.raises(ConfigError):
        safety_stock(1.0, -1.0, 1.0, 1)
