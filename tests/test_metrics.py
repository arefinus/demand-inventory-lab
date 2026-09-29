from __future__ import annotations

import numpy as np
import pytest

from demand_inventory_lab.errors import EmptySeriesError
from demand_inventory_lab.forecast.metrics import (
    DEFAULT_METRICS,
    interval_coverage,
    mae,
    mape,
    mase,
    seasonal_naive_denominator,
    wape,
)


def test_mae_basic():
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_wape_basic():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)


def test_wape_is_undefined_when_all_actuals_are_zero():
    assert wape([0.0, 0.0, 0.0], [1.0, 0.0, 2.0]) is None


def test_mase_uses_explicit_seasonal_naive_denominator():
    train = [1.0, 2.0, 2.0, 4.0]  # m=2 seasonal diffs: |2-1|, |4-2| -> mean 1.5
    assert seasonal_naive_denominator(train, 2) == pytest.approx(1.5)
    res = mase([1.0], [3.0], train, m=2)
    assert res.denominator == pytest.approx(1.5)
    assert res.value == pytest.approx(2.0 / 1.5)
    assert res.to_dict()["season_length"] == 2


def test_mase_denominator_zero_returns_none_but_keeps_denominator():
    res = mase([1.0], [3.0], [1.0, 2.0, 1.0, 2.0], m=2)
    assert res.value is None
    assert res.denominator == 0.0


def test_mase_needs_more_than_one_season_of_training():
    with pytest.raises(EmptySeriesError):
        seasonal_naive_denominator([1.0, 2.0], m=2)


def test_interval_coverage_counts_only_rows_with_intervals():
    y = [1.0, 5.0, 3.0]
    lo = [0.0, np.nan, 4.0]
    hi = [2.0, np.nan, 6.0]
    assert interval_coverage(y, lo, hi) == pytest.approx(0.5)
    assert interval_coverage(y, [np.nan] * 3, [np.nan] * 3) is None


def test_mape_is_not_a_default_metric_and_is_undefined_at_zero():
    assert "mape" not in DEFAULT_METRICS
    assert mape([0.0, 1.0], [1.0, 1.0]) is None
    assert mape([2.0, 4.0], [1.0, 4.0]) == pytest.approx(0.25)


def test_metrics_reject_empty_series():
    with pytest.raises(EmptySeriesError):
        mae([], [])
