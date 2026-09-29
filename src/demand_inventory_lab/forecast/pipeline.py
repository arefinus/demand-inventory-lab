"""Assembles the three baseline models and produces the final forecast with intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, ForecastFn, residuals_by_horizon, rolling_origin_backtest
from .features import FeatureSpec
from .gbm import GBMConfig, gbm_forecast_factory
from .holt_winters import holt_winters_forecast
from .seasonal_naive import seasonal_naive_forecast

MODEL_NAMES = ("seasonal_naive", "holt_winters", "lag_gbm")


def backtest_config_from_dict(cfg: Mapping) -> BacktestConfig:
    return BacktestConfig(
        season_length=int(cfg.get("season_length", 12)),
        max_horizon=int(cfg.get("max_horizon", 6)),
        min_train=int(cfg.get("min_train", 48)),
        step=int(cfg.get("step", 1)),
        interval_alpha=float(cfg.get("interval_alpha", 0.2)),
        min_residuals=int(cfg.get("min_residuals", 8)),
    )


def gbm_config_from_dict(cfg: Mapping, max_horizon: int, season_length: int) -> GBMConfig:
    g = cfg.get("gbm", {})
    spec = FeatureSpec(
        lags=tuple(int(k) for k in g.get("lags", range(1, 13))),
        rolling_windows=tuple(int(w) for w in g.get("rolling_windows", (3, 12))),
        season_length=season_length,
    )
    return GBMConfig(
        max_horizon=max_horizon,
        feature_spec=spec,
        max_iter=int(g.get("max_iter", 150)),
        learning_rate=float(g.get("learning_rate", 0.05)),
        max_leaf_nodes=int(g.get("max_leaf_nodes", 15)),
        random_state=int(g.get("random_state", 42)),
    )


def build_models(bt: BacktestConfig, gbm: GBMConfig) -> dict[str, ForecastFn]:
    m = bt.season_length
    return {
        "seasonal_naive": lambda y, h: seasonal_naive_forecast(y, h, m),
        "holt_winters": lambda y, h: holt_winters_forecast(y, h, m),
        "lag_gbm": gbm_forecast_factory(gbm),
    }


@dataclass(frozen=True)
class ForecastRow:
    step: int
    date: str | None
    point: float
    lo: float | None
    hi: float | None


def final_forecast(
    y,
    dates: pd.DatetimeIndex | None,
    models: Mapping[str, ForecastFn],
    backtest: BacktestResult,
) -> dict[str, list[dict]]:
    """Fit each model on the full series and attach intervals from the backtest
    residuals per horizon. All residuals precede the forecast origin, so this is
    still point in time."""
    arr = np.asarray(y, dtype=float).ravel()
    H = backtest.config.max_horizon
    alpha = backtest.config.interval_alpha
    future_dates = None
    if dates is not None:
        last = pd.Period(dates[-1], freq="M")
        future_dates = [(last + h).strftime("%Y-%m") for h in range(1, H + 1)]
    out: dict[str, list[dict]] = {}
    for name, fn in models.items():
        fc = np.asarray(fn(arr.copy(), H), dtype=float)
        res = residuals_by_horizon(backtest.records, name)
        rows = []
        for h in range(1, H + 1):
            pool = res.get(h, np.array([]))
            if pool.size >= backtest.config.min_residuals:
                lo_q, hi_q = np.quantile(pool, [alpha / 2, 1 - alpha / 2])
                lo, hi = float(fc[h - 1] + lo_q), float(fc[h - 1] + hi_q)
            else:
                lo, hi = None, None
            rows.append(
                {
                    "step": h,
                    "date": future_dates[h - 1] if future_dates else None,
                    "point": float(fc[h - 1]),
                    "lo": lo,
                    "hi": hi,
                    "n_residuals": int(pool.size),
                }
            )
        out[name] = rows
    return out
