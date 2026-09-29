from __future__ import annotations

import numpy as np
import pytest

from demand_inventory_lab.errors import EmptySeriesError, InsufficientHistoryError
from demand_inventory_lab.forecast.seasonal_naive import seasonal_naive_forecast
from conftest import BASE_PATTERN, SEASON


def test_exact_on_periodic_series(periodic):
    fc = seasonal_naive_forecast(periodic, horizon=12, m=SEASON)
    np.testing.assert_allclose(fc, BASE_PATTERN)


def test_horizon_beyond_one_season_wraps_to_most_recent_season(periodic):
    fc = seasonal_naive_forecast(periodic, horizon=18, m=SEASON)
    np.testing.assert_allclose(fc[12:], fc[:6])
    np.testing.assert_allclose(fc[:12], BASE_PATTERN)


def test_seasonal_index_points_to_same_season():
    y = np.arange(1.0, 25.0)  # 24 months, values 1..24
    fc = seasonal_naive_forecast(y, horizon=3, m=12)
    # step 1 forecasts month 25 -> same season as month 13 (value 13)
    np.testing.assert_allclose(fc, [13.0, 14.0, 15.0])


def test_insufficient_history_raises():
    with pytest.raises(InsufficientHistoryError):
        seasonal_naive_forecast(np.arange(5.0), horizon=1, m=12)


def test_empty_series_raises():
    with pytest.raises(EmptySeriesError):
        seasonal_naive_forecast(np.array([]), horizon=1, m=12)


def test_zero_series_forecasts_zero():
    fc = seasonal_naive_forecast(np.zeros(24), horizon=6, m=12)
    assert np.all(fc == 0.0)
