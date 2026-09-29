"""Synthetic integer demand generator (negative binomial with seasonality)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemandParams:
    mean: float = 40.0
    dispersion: float = 8.0          # gamma shape; smaller means more variable demand
    seasonal_amplitude: float = 0.2  # relative amplitude of the yearly cycle
    season_length: int = 52
    trend_per_period: float = 0.0    # multiplicative drift per period

    def __post_init__(self) -> None:
        if self.mean < 0 or self.dispersion <= 0 or self.season_length < 1:
            raise ValueError("mean must be >= 0, dispersion > 0, season_length >= 1")


def generate_demand(rng: np.random.Generator, n_periods: int, params: DemandParams) -> np.ndarray:
    """Integer demand per period. Mean follows a seasonal cycle with optional trend;
    the count is Poisson with a gamma-mixed rate, so variance exceeds the mean."""
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    t = np.arange(n_periods)
    season = 1.0 + params.seasonal_amplitude * np.sin(2.0 * np.pi * t / params.season_length)
    trend = (1.0 + params.trend_per_period) ** t
    mu = np.maximum(params.mean * season * trend, 0.0)
    if params.mean == 0.0:
        return np.zeros(n_periods, dtype=int)
    rate = rng.gamma(shape=params.dispersion, scale=mu / params.dispersion)
    return rng.poisson(rate).astype(int)
