"""Seasonal naive forecast: the value observed one season earlier."""

from __future__ import annotations

import numpy as np

from ..errors import EmptySeriesError, InsufficientHistoryError


def seasonal_naive_forecast(y, horizon: int, m: int) -> np.ndarray:
    """Forecast ``horizon`` steps ahead of the end of ``y``.

    For step h (1-based) the forecast is the last observed value from the same
    season: ``y[n + h - 1 - m * ceil(h / m)]``. Horizons longer than one season
    wrap around to the most recent full season.
    """
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise EmptySeriesError("seasonal naive needs at least one season of history; got an empty series.")
    if m < 1 or horizon < 1:
        raise ValueError("season_length and horizon must be >= 1")
    if arr.size < m:
        raise InsufficientHistoryError(
            f"seasonal naive needs at least season_length={m} observations; got {arr.size}."
        )
    n = arr.size
    steps = np.arange(1, horizon + 1)
    cycles = np.ceil(steps / m).astype(int)
    idx = n + steps - 1 - m * cycles
    return arr[idx]
