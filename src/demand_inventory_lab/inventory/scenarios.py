"""Scenario grid: reorder point x lead time -> service level and holding cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..errors import ConfigError
from .policy import CostAssumptions, LeadTime, Policy, Shock
from .simulator import simulate

SCHEMA_VERSION = 1
MODE = "synthetic-demo"


@dataclass(frozen=True)
class ScenarioGridConfig:
    reorder_points: tuple[int, ...]
    lead_times: tuple[int, ...]
    order_quantity: int
    review_period: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.reorder_points or not self.lead_times:
            raise ConfigError("scenario grid needs at least one reorder point and one lead time")
        if self.order_quantity <= 0:
            raise ConfigError("order_quantity must be > 0")

    @classmethod
    def from_mapping(cls, m: Mapping) -> "ScenarioGridConfig":
        return cls(
            reorder_points=tuple(int(x) for x in m["reorder_points"]),
            lead_times=tuple(int(x) for x in m["lead_times"]),
            order_quantity=int(m["order_quantity"]),
            review_period=int(m.get("review_period", 1)),
            seed=int(m.get("seed", 42)),
        )

    def to_dict(self) -> dict:
        return {
            "reorder_points": list(self.reorder_points),
            "lead_times": list(self.lead_times),
            "order_quantity": self.order_quantity,
            "review_period": self.review_period,
            "seed": self.seed,
        }


def _cell_seed(base: int, s: int, lt: int) -> int:
    return int(base * 1_000_003 + s * 1_009 + lt) % (2**32 - 1)


def run_scenario_grid(demand, cfg: ScenarioGridConfig, costs: CostAssumptions, shock: Shock | None = None) -> dict:
    """Deterministic for a given (demand, cfg, costs, shock). Each cell gets its own
    derived seed so adding a cell never changes another cell's result."""
    grid = []
    for s in cfg.reorder_points:
        for lt in cfg.lead_times:
            policy = Policy(reorder_point=s, order_up_to=s + cfg.order_quantity, review_period=cfg.review_period)
            res = simulate(demand, policy, LeadTime(fixed=lt), costs, seed=_cell_seed(cfg.seed, s, lt), shock=shock)
            grid.append(
                {
                    "reorder_point": s,
                    "order_up_to": s + cfg.order_quantity,
                    "lead_time": lt,
                    "fill_rate": res.fill_rate,
                    "period_service_level": round(res.period_service_level, 6),
                    "stockout_periods": res.stockout_periods,
                    "order_count": res.order_count,
                    "holding_cost": round(res.holding_cost, 4),
                    "ordering_cost": round(res.ordering_cost, 4),
                    "stockout_cost": round(res.stockout_cost, 4),
                    "total_cost": round(res.total_cost, 4),
                }
            )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "mode": MODE,
        "generated_by": "demand_inventory_lab.inventory.scenarios.run_scenario_grid",
        "n_periods": int(np.asarray(demand).size),
        "config": cfg.to_dict(),
        "cost_assumptions": costs.to_dict(),
        "shock": (None if shock is None else {"start": shock.start, "end": shock.end, "multiplier": shock.multiplier}),
        "grid": grid,
    }


def shock_comparison(demand, policy: Policy, lead_time: LeadTime, costs: CostAssumptions, shock: Shock, seed: int = 42) -> dict:
    base = simulate(demand, policy, lead_time, costs, seed=seed)
    shocked = simulate(demand, policy, lead_time, costs, seed=seed, shock=shock)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "mode": MODE,
        "policy": {"reorder_point": policy.reorder_point, "order_up_to": policy.order_up_to, "review_period": policy.review_period},
        "lead_time": lead_time.describe(),
        "shock": {"start": shock.start, "end": shock.end, "multiplier": shock.multiplier},
        "baseline": base.summary(),
        "shocked": shocked.summary(),
    }
