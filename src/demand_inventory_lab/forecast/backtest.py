"""Rolling-origin backtest with point-in-time empirical intervals.

For each origin o (number of observations available), every model is fitted on
``y[:o]`` only and forecasts targets ``o .. o+H-1``. Intervals for a fold at
origin o and horizon h are formed from the out-of-sample residuals of earlier
folds whose target index is strictly below o, so no residual used to build an
interval comes from an observation the forecaster could not yet have seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ..errors import InsufficientHistoryError
from .metrics import interval_coverage, mae, seasonal_naive_denominator, wape

ForecastFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class BacktestConfig:
    season_length: int = 12
    max_horizon: int = 6
    min_train: int = 48
    step: int = 1
    interval_alpha: float = 0.2
    min_residuals: int = 8

    def __post_init__(self) -> None:
        if self.max_horizon < 1 or self.step < 1 or self.min_train < 1:
            raise ValueError("max_horizon, step and min_train must be >= 1")
        if not 0.0 < self.interval_alpha < 1.0:
            raise ValueError("interval_alpha must be in (0, 1)")
        if self.min_train <= self.season_length:
            raise ValueError("min_train must exceed season_length so the MASE denominator is defined")

    def to_dict(self) -> dict:
        return {
            "season_length": self.season_length,
            "max_horizon": self.max_horizon,
            "min_train": self.min_train,
            "step": self.step,
            "interval_alpha": self.interval_alpha,
            "min_residuals": self.min_residuals,
        }


@dataclass
class BacktestResult:
    records: pd.DataFrame
    config: BacktestConfig
    n_obs: int
    metrics: dict = field(default_factory=dict)

    def origins(self) -> list[int]:
        return sorted(self.records["origin"].unique().tolist())


RECORD_COLUMNS = [
    "model", "origin", "horizon", "target_index", "target_date", "actual", "forecast",
    "lo", "hi", "n_residuals", "mase_denominator",
]


def fold_origins(n: int, config: BacktestConfig) -> np.ndarray:
    if n <= config.min_train:
        raise InsufficientHistoryError(
            f"Backtest needs more than min_train={config.min_train} observations; got {n}."
        )
    return np.arange(config.min_train, n, config.step)


def _interval(pool: np.ndarray, point: float, alpha: float, min_residuals: int) -> tuple[float, float, int]:
    if pool.size < min_residuals:
        return float("nan"), float("nan"), int(pool.size)
    lo_q, hi_q = np.quantile(pool, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(point + lo_q), float(point + hi_q), int(pool.size)


def rolling_origin_backtest(
    y,
    dates: pd.DatetimeIndex | None,
    models: Mapping[str, ForecastFn],
    config: BacktestConfig,
) -> BacktestResult:
    arr = np.asarray(y, dtype=float).ravel()
    n = arr.size
    origins = fold_origins(n, config)
    if dates is not None and len(dates) != n:
        raise ValueError("dates must align with y")

    rows: list[dict] = []
    # residual history per (model, horizon): list of (target_index, residual)
    history: dict[tuple[str, int], list[tuple[int, float]]] = {}

    for o in origins:
        train = arr[:o]
        H = min(config.max_horizon, n - o)
        denom = seasonal_naive_denominator(train, config.season_length)
        for name, fn in models.items():
            fc = np.asarray(fn(train.copy(), H), dtype=float).ravel()
            if fc.shape != (H,):
                raise ValueError(f"model {name!r} returned shape {fc.shape}, expected ({H},)")
            for h in range(1, H + 1):
                t_idx = o + h - 1
                past = history.setdefault((name, h), [])
                pool = np.array([r for (ti, r) in past if ti < o], dtype=float)
                lo, hi, n_res = _interval(pool, fc[h - 1], config.interval_alpha, config.min_residuals)
                actual = float(arr[t_idx])
                rows.append(
                    {
                        "model": name,
                        "origin": int(o),
                        "horizon": h,
                        "target_index": int(t_idx),
                        "target_date": (dates[t_idx].strftime("%Y-%m") if dates is not None else None),
                        "actual": actual,
                        "forecast": float(fc[h - 1]),
                        "lo": lo,
                        "hi": hi,
                        "n_residuals": n_res,
                        "mase_denominator": denom,
                    }
                )
                past.append((t_idx, actual - float(fc[h - 1])))

    records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    result = BacktestResult(records=records, config=config, n_obs=n)
    result.metrics = summarise(records)
    return result


def _group_metrics(g: pd.DataFrame) -> dict:
    actual = g["actual"].to_numpy()
    fc = g["forecast"].to_numpy()
    denom = g["mase_denominator"].to_numpy()
    scaled = np.abs(actual - fc)[denom > 0] / denom[denom > 0]
    return {
        "n": int(len(g)),
        "mae": mae(actual, fc),
        "wape": wape(actual, fc),
        "mase": (float(np.mean(scaled)) if scaled.size else None),
        "mase_denominator_mean": float(np.mean(denom)),
        "interval_coverage": interval_coverage(actual, g["lo"].to_numpy(), g["hi"].to_numpy()),
        "n_with_interval": int(g["lo"].notna().sum()),
    }


def summarise(records: pd.DataFrame) -> dict:
    out: dict = {}
    for name, g in records.groupby("model", sort=True):
        out[name] = {
            "overall": _group_metrics(g),
            "by_horizon": {int(h): _group_metrics(gh) for h, gh in g.groupby("horizon", sort=True)},
        }
    return out


def residuals_by_horizon(records: pd.DataFrame, model: str) -> dict[int, np.ndarray]:
    g = records[records["model"] == model]
    return {int(h): (gh["actual"] - gh["forecast"]).to_numpy(dtype=float) for h, gh in g.groupby("horizon")}
