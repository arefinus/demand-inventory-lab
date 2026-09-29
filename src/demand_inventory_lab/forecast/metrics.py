"""Point and interval accuracy metrics.

Default metric set: MAE, WAPE, MASE, interval coverage.

MAPE is deliberately not in the default set. Its per-point relative error divides
by the actual value, so a single zero or near-zero month (common in intermittent
demand and in shocked series) makes the metric undefined or dominated by one
observation, and it penalises over-forecasts more than under-forecasts of the
same size. WAPE divides by the sum of actuals, which is defined whenever total
demand is positive and treats over- and under-forecasts symmetrically. MASE
scales by the in-sample seasonal-naive error, which makes series of different
scale comparable and reports the denominator explicitly.

Metrics that are undefined return ``None`` rather than ``inf`` or ``nan`` so the
undefined case is visible in JSON output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import EmptySeriesError


def _as_arrays(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y, dtype=float).ravel()
    b = np.asarray(yhat, dtype=float).ravel()
    if a.size == 0:
        raise EmptySeriesError("Cannot compute a metric on an empty series.")
    if a.shape != b.shape:
        raise ValueError(f"actual and forecast lengths differ: {a.shape} vs {b.shape}")
    return a, b


def mae(y, yhat) -> float:
    a, b = _as_arrays(y, yhat)
    return float(np.mean(np.abs(a - b)))


def wape(y, yhat) -> float | None:
    """Weighted absolute percentage error, sum|e| / sum|y|.

    Returns None when the denominator is zero (all actuals zero); the ratio is
    undefined in that case and must not be reported as a number.
    """
    a, b = _as_arrays(y, yhat)
    denom = float(np.sum(np.abs(a)))
    if denom <= 0.0:
        return None
    return float(np.sum(np.abs(a - b)) / denom)


@dataclass(frozen=True)
class MaseResult:
    value: float | None
    denominator: float
    season_length: int

    def to_dict(self) -> dict:
        return {"mase": self.value, "mase_denominator": self.denominator, "season_length": self.season_length}


def seasonal_naive_denominator(train, m: int) -> float:
    """Mean absolute in-sample seasonal-naive error over the training window."""
    t = np.asarray(train, dtype=float).ravel()
    if m < 1:
        raise ValueError("season_length must be >= 1")
    if t.size <= m:
        raise EmptySeriesError(
            f"MASE denominator needs more than season_length={m} training points; got {t.size}."
        )
    return float(np.mean(np.abs(t[m:] - t[:-m])))


def mase(y, yhat, train, m: int) -> MaseResult:
    """Mean absolute scaled error with an explicit seasonal-naive denominator.

    Returns value None when the denominator is zero (a training window that is
    exactly periodic or constant), keeping the denominator in the result so the
    reason is visible.
    """
    denom = seasonal_naive_denominator(train, m)
    err = mae(y, yhat)
    if denom <= 0.0:
        return MaseResult(None, denom, m)
    return MaseResult(float(err / denom), denom, m)


def interval_coverage(y, lo, hi) -> float | None:
    """Share of actuals inside [lo, hi]. Rows with a NaN bound are excluded; if no
    interval exists, returns None."""
    a = np.asarray(y, dtype=float).ravel()
    l = np.asarray(lo, dtype=float).ravel()
    h = np.asarray(hi, dtype=float).ravel()
    if not (a.shape == l.shape == h.shape):
        raise ValueError("actual, lo and hi must have equal lengths")
    mask = ~(np.isnan(l) | np.isnan(h))
    if not mask.any():
        return None
    inside = (a[mask] >= l[mask]) & (a[mask] <= h[mask])
    return float(np.mean(inside))


def mape(y, yhat) -> float | None:
    """Mean absolute percentage error. Not part of the default metric set (see the
    module docstring). Returns None when any actual is zero."""
    a, b = _as_arrays(y, yhat)
    if np.any(a == 0.0):
        return None
    return float(np.mean(np.abs((a - b) / a)))


DEFAULT_METRICS = ("mae", "wape", "mase", "interval_coverage")
