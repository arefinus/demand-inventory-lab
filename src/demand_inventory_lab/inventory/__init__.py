"""Synthetic SKU demand and a periodic-review replenishment simulator."""

from __future__ import annotations

from .demand import DemandParams, generate_demand
from .policy import CostAssumptions, LeadTime, Policy, Shock, reorder_point, safety_stock
from .scenarios import ScenarioGridConfig, run_scenario_grid, shock_comparison
from .simulator import SimulationResult, simulate

__all__ = [
    "CostAssumptions",
    "DemandParams",
    "LeadTime",
    "Policy",
    "ScenarioGridConfig",
    "Shock",
    "SimulationResult",
    "generate_demand",
    "reorder_point",
    "run_scenario_grid",
    "safety_stock",
    "shock_comparison",
    "simulate",
]
