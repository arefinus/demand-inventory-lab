from __future__ import annotations

import numpy as np
import pytest

from demand_inventory_lab.errors import EmptySeriesError, InsufficientHistoryError
from demand_inventory_lab.forecast.holt_winters import holt_winters_forecast


def test_returns_finite_forecast_of_requested_length(trending_seasonal):
    fc = holt_winters_forecast(trending_seasonal, horizon=6, m=12)
    assert fc.shape == (6,)
    assert np.all(np.isfinite(fc))
    # trend is positive, forecast should sit above the series mean of the last year
    assert fc.mean() > trending_seasonal[-12:].mean() - 15.0


def test_constant_series_returns_constant():
    fc = holt_winters_forecast(np.full(36, 7.0), horizon=4, m=12)
    np.testing.assert_allclose(fc, 7.0)


def test_insufficient_history_raises():
    with pytest.raises(InsufficientHistoryError):
        holt_winters_forecast(np.arange(20.0), horizon=3, m=12)
    with pytest.raises(EmptySeriesError):
        holt_winters_forecast(np.array([]), horizon=3, m=12)
