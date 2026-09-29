from __future__ import annotations

import numpy as np
import pytest

from demand_inventory_lab.errors import InsufficientHistoryError
from demand_inventory_lab.forecast.features import FeatureSpec, TrainingOnlyScaler, build_training_matrix, features_at
from demand_inventory_lab.forecast.gbm import GBMConfig, LagGBMForecaster

SPEC = FeatureSpec()


def test_features_at_t_use_only_observations_before_t(trending_seasonal):
    y = trending_seasonal.copy()
    t = 60
    before = features_at(y, t, SPEC)
    y_perturbed = y.copy()
    y_perturbed[t:] += 1_000.0  # anything at or after t must be invisible
    after = features_at(y_perturbed, t, SPEC)
    np.testing.assert_array_equal(before, after)


def test_features_at_rejects_too_short_history(trending_seasonal):
    with pytest.raises(InsufficientHistoryError):
        features_at(trending_seasonal, SPEC.min_history - 1, SPEC)


def test_training_rows_align_target_with_horizon(trending_seasonal):
    y = trending_seasonal
    for h in (1, 3, 6):
        X, target, level, origins = build_training_matrix(y, h, SPEC)
        np.testing.assert_allclose(target * level, y[origins + h - 1])
        assert X.shape[0] == target.shape[0] == origins.shape[0]


def test_training_rows_never_reach_past_series_end(trending_seasonal):
    n = trending_seasonal.size
    for h in (1, 6):
        _, _, _, origins = build_training_matrix(trending_seasonal, h, SPEC)
        assert origins.max() + h - 1 == n - 1
        assert origins.min() == SPEC.min_history


def test_scaler_statistics_do_not_change_when_test_rows_are_perturbed(trending_seasonal):
    train = trending_seasonal[:80]
    s1 = TrainingOnlyScaler().fit(train)
    perturbed = trending_seasonal.copy()
    perturbed[80:] *= 5.0
    s2 = TrainingOnlyScaler().fit(perturbed[:80])
    assert s1.fitted_statistics() == s2.fitted_statistics()
    with pytest.raises(RuntimeError):
        TrainingOnlyScaler().transform([1.0])


def test_gbm_fitted_statistics_and_predictions_ignore_post_training_values(trending_seasonal):
    y = trending_seasonal
    cfg = GBMConfig(max_horizon=3, max_iter=30)
    a = LagGBMForecaster(cfg).fit(y[:80])
    perturbed = y.copy()
    perturbed[80:] += 500.0
    b = LagGBMForecaster(cfg).fit(perturbed[:80])
    assert a.fitted_statistics() == b.fitted_statistics()
    np.testing.assert_allclose(a.predict(y[:80]), b.predict(y[:80]))


def test_gbm_is_deterministic_for_a_seed(trending_seasonal):
    cfg = GBMConfig(max_horizon=2, max_iter=30, random_state=7)
    p1 = LagGBMForecaster(cfg).fit(trending_seasonal).predict(trending_seasonal)
    p2 = LagGBMForecaster(cfg).fit(trending_seasonal).predict(trending_seasonal)
    np.testing.assert_array_equal(p1, p2)
    assert p1.shape == (2,)
    assert np.all(p1 >= 0.0)


def test_gbm_handles_zero_series_and_short_history():
    cfg = GBMConfig(max_horizon=2, max_iter=20)
    fc = LagGBMForecaster(cfg).fit(np.zeros(40)).predict(np.zeros(40))
    assert np.all(np.isfinite(fc)) and np.all(fc == 0.0)
    with pytest.raises(InsufficientHistoryError):
        LagGBMForecaster(cfg).fit(np.arange(15.0))
    with pytest.raises(RuntimeError):
        LagGBMForecaster(cfg).predict(np.zeros(40))
