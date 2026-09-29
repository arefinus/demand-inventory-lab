from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_inventory_lab.forecast.backtest import BacktestConfig, fold_origins, rolling_origin_backtest
from demand_inventory_lab.forecast.metrics import seasonal_naive_denominator
from demand_inventory_lab.forecast.pipeline import final_forecast
from demand_inventory_lab.forecast.seasonal_naive import seasonal_naive_forecast

CFG = BacktestConfig(season_length=12, max_horizon=6, min_train=36, step=3, interval_alpha=0.2, min_residuals=4)
SN = {"seasonal_naive": lambda y, h: seasonal_naive_forecast(y, h, 12)}


def test_forecaster_only_ever_sees_observations_before_the_origin(trending_seasonal):
    seen: list[tuple[int, np.ndarray]] = []

    def spy(y_train, h):
        seen.append((y_train.size, y_train.copy()))
        return np.zeros(h)

    rolling_origin_backtest(trending_seasonal, None, {"spy": spy}, CFG)
    assert seen, "spy was never called"
    for size, train in seen:
        np.testing.assert_array_equal(train, trending_seasonal[:size])
    origins = fold_origins(trending_seasonal.size, CFG)
    assert sorted(s for s, _ in seen) == sorted(origins.tolist())


def test_origins_respect_min_train_and_step(trending_seasonal):
    origins = fold_origins(trending_seasonal.size, CFG)
    assert origins[0] == CFG.min_train
    assert np.all(np.diff(origins) == CFG.step)
    assert origins[-1] < trending_seasonal.size


def test_targets_never_exceed_series_end(trending_seasonal):
    res = rolling_origin_backtest(trending_seasonal, None, SN, CFG)
    assert res.records["target_index"].max() == trending_seasonal.size - 1
    last = res.records[res.records["origin"] == res.origins()[-1]]
    assert last["horizon"].max() == min(CFG.max_horizon, trending_seasonal.size - res.origins()[-1])


def test_interval_residual_pool_uses_only_targets_observed_before_the_origin(trending_seasonal):
    res = rolling_origin_backtest(trending_seasonal, None, SN, CFG)
    rec = res.records
    for _, row in rec.iterrows():
        earlier = rec[(rec["horizon"] == row["horizon"]) & (rec["target_index"] < row["origin"])]
        assert row["n_residuals"] == len(earlier)


def test_intervals_absent_until_min_residuals_then_present(trending_seasonal):
    res = rolling_origin_backtest(trending_seasonal, None, SN, CFG)
    h1 = res.records[res.records["horizon"] == 1].sort_values("origin")
    assert h1["lo"].isna().iloc[: CFG.min_residuals].all()
    assert h1["lo"].notna().iloc[-1]
    with_iv = h1[h1["lo"].notna()]
    assert (with_iv["lo"] <= with_iv["hi"]).all()


def test_mase_denominator_recorded_per_fold_matches_training_window(trending_seasonal):
    res = rolling_origin_backtest(trending_seasonal, None, SN, CFG)
    for o, g in res.records.groupby("origin"):
        expected = seasonal_naive_denominator(trending_seasonal[:o], CFG.season_length)
        assert g["mase_denominator"].unique().tolist() == [pytest.approx(expected)]
    m = res.metrics["seasonal_naive"]
    assert set(m["by_horizon"].keys()) == set(range(1, CFG.max_horizon + 1))
    assert {"mae", "wape", "mase", "interval_coverage", "n"} <= set(m["overall"].keys())


def test_wape_reported_as_undefined_on_all_zero_series():
    res = rolling_origin_backtest(np.zeros(60), None, SN, CFG)
    assert res.metrics["seasonal_naive"]["overall"]["wape"] is None
    assert res.metrics["seasonal_naive"]["overall"]["mase"] is None
    assert res.metrics["seasonal_naive"]["overall"]["mae"] == 0.0


def test_config_rejects_min_train_not_exceeding_season_length():
    with pytest.raises(ValueError):
        BacktestConfig(season_length=12, min_train=12)
    with pytest.raises(ValueError):
        BacktestConfig(interval_alpha=1.5)


def test_final_forecast_continues_dates_and_attaches_intervals(trending_seasonal):
    dates = pd.date_range("2015-01-01", periods=trending_seasonal.size, freq="MS")
    res = rolling_origin_backtest(trending_seasonal, dates, SN, CFG)
    out = final_forecast(trending_seasonal, dates, SN, res)
    rows = out["seasonal_naive"]
    assert [r["step"] for r in rows] == list(range(1, CFG.max_horizon + 1))
    assert rows[0]["date"] == "2025-01"
    # Empirical residual intervals are not centred on the point forecast; a biased
    # model (seasonal naive on a trending series) has lo > point. Only lo <= hi holds.
    assert all(r["lo"] is not None and r["lo"] <= r["hi"] for r in rows)
    assert all(r["n_residuals"] >= CFG.min_residuals for r in rows)
