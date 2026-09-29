"""Replenishment policy, lead time, cost assumptions and shock definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..errors import ConfigError


@dataclass(frozen=True)
class Policy:
    """Periodic review. ``kind`` is 'sS' (order up to S when position <= s) or
    'sQ' (order a fixed quantity Q when position <= s)."""

    reorder_point: int
    order_up_to: int | None = None
    order_quantity: int | None = None
    review_period: int = 1
    kind: str = "sS"

    def __post_init__(self) -> None:
        if self.review_period < 1:
            raise ConfigError("review_period must be >= 1")
        if self.kind == "sS":
            if self.order_up_to is None or self.order_up_to <= self.reorder_point:
                raise ConfigError("(s, S) policy needs order_up_to > reorder_point")
        elif self.kind == "sQ":
            if self.order_quantity is None or self.order_quantity <= 0:
                raise ConfigError("(s, Q) policy needs order_quantity > 0")
        else:
            raise ConfigError(f"unknown policy kind {self.kind!r}; use 'sS' or 'sQ'")

    def order_size(self, inventory_position: int) -> int:
        if inventory_position > self.reorder_point:
            return 0
        if self.kind == "sS":
            return max(0, int(self.order_up_to) - inventory_position)
        return int(self.order_quantity)

    def initial_on_hand(self) -> int:
        return int(self.order_up_to) if self.kind == "sS" else self.reorder_point + int(self.order_quantity)


@dataclass(frozen=True)
class LeadTime:
    """Fixed lead time, or a discrete distribution given as ((value, prob), ...)."""

    fixed: int | None = None
    distribution: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self) -> None:
        if (self.fixed is None) == (self.distribution is None):
            raise ConfigError("LeadTime needs exactly one of fixed or distribution")
        if self.fixed is not None and self.fixed < 0:
            raise ConfigError("fixed lead time must be >= 0")
        if self.distribution is not None:
            vals = [v for v, _ in self.distribution]
            probs = [p for _, p in self.distribution]
            if any(v < 0 for v in vals) or any(p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-9):
                raise ConfigError("lead time distribution needs non-negative values and probabilities summing to 1")

    def sample(self, rng: np.random.Generator) -> int:
        if self.fixed is not None:
            return int(self.fixed)
        vals = np.array([v for v, _ in self.distribution])
        probs = np.array([p for _, p in self.distribution])
        return int(rng.choice(vals, p=probs))

    def mean(self) -> float:
        if self.fixed is not None:
            return float(self.fixed)
        return float(sum(v * p for v, p in self.distribution))

    def describe(self) -> dict:
        return {"fixed": self.fixed, "distribution": [list(x) for x in self.distribution] if self.distribution else None}


@dataclass(frozen=True)
class CostAssumptions:
    unit_cost: float
    holding_rate_per_period: float
    order_cost: float
    stockout_penalty_per_unit: float
    unmet_demand: str = "lost_sales"

    def __post_init__(self) -> None:
        if self.unmet_demand not in ("lost_sales", "backorder"):
            raise ConfigError("unmet_demand must be 'lost_sales' or 'backorder'")
        if min(self.unit_cost, self.holding_rate_per_period, self.order_cost, self.stockout_penalty_per_unit) < 0:
            raise ConfigError("cost assumptions must be non-negative")

    @property
    def holding_cost_per_unit_period(self) -> float:
        return self.unit_cost * self.holding_rate_per_period

    @classmethod
    def from_mapping(cls, m: Mapping) -> "CostAssumptions":
        try:
            return cls(
                unit_cost=float(m["unit_cost"]),
                holding_rate_per_period=float(m["holding_rate_per_period"]),
                order_cost=float(m["order_cost"]),
                stockout_penalty_per_unit=float(m["stockout_penalty_per_unit"]),
                unmet_demand=str(m.get("unmet_demand", "lost_sales")),
            )
        except KeyError as exc:
            raise ConfigError(f"assumptions.yaml is missing key {exc}") from exc

    def to_dict(self) -> dict:
        return {
            "unit_cost": self.unit_cost,
            "holding_rate_per_period": self.holding_rate_per_period,
            "order_cost": self.order_cost,
            "stockout_penalty_per_unit": self.stockout_penalty_per_unit,
            "unmet_demand": self.unmet_demand,
        }


@dataclass(frozen=True)
class Shock:
    """Multiply demand by ``multiplier`` for periods start <= t < end."""

    start: int
    end: int
    multiplier: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or self.multiplier < 0:
            raise ConfigError("Shock needs 0 <= start < end and multiplier >= 0")

    def apply(self, demand: np.ndarray) -> np.ndarray:
        out = np.asarray(demand, dtype=float).copy()
        out[self.start : self.end] = np.round(out[self.start : self.end] * self.multiplier)
        return out.astype(int)


def safety_stock(z: float, demand_std: float, lead_time_mean: float, review_period: int) -> float:
    """z * sigma_demand * sqrt(L + R) under independent per-period demand."""
    if demand_std < 0 or lead_time_mean < 0 or review_period < 1:
        raise ConfigError("demand_std and lead_time_mean must be >= 0, review_period >= 1")
    return float(z * demand_std * math.sqrt(lead_time_mean + review_period))


def reorder_point(demand_mean: float, lead_time_mean: float, review_period: int, ss: float) -> int:
    """Expected demand over the protection interval plus safety stock, rounded up."""
    return int(math.ceil(demand_mean * (lead_time_mean + review_period) + ss))
