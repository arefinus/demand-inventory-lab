"""Holt-Winters additive exponential smoothing via statsmodels."""

from __future__ import annotations

import warnings

import numpy as np

from ..errors import EmptySeriesError, InsufficientHistoryError


def holt_winters_forecast(y, horizon: int, m: int, trend: str = "add", seasonal: str = "add") -> np.ndarray:
    """Fit additive trend + additive seasonality on ``y`` and forecast ``horizon`` steps.

    Requires at least two full seasons. A constant series (zero variance) is
    returned as a constant forecast without fitting, because the optimiser has
    nothing to estimate.
    """
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise EmptySeriesError("Holt-Winters needs at least two seasons of history; got an empty series.")
    if horizon < 1 or m < 2:
        raise ValueError("horizon must be >= 1 and season_length >= 2")
    if arr.size < 2 * m:
        raise InsufficientHistoryError(
            f"Holt-Winters needs at least 2 * season_length = {2 * m} observations; got {arr.size}."
        )
    if np.allclose(arr, arr[0]):
        return np.full(horizon, arr[0], dtype=float)

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            arr,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=m,
            initialization_method="estimated",
            damped_trend=True,
        )
        fit = model.fit(optimized=True)
        out = np.asarray(fit.forecast(horizon), dtype=float)
    if not np.all(np.isfinite(out)):
        raise InsufficientHistoryError("Holt-Winters produced non-finite forecasts; series is degenerate.")
    return out
