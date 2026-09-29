# Method

## 1. Forecasting baselines

All three models implement one interface, `forecast_fn(y_train, horizon) -> array`, so
the backtest treats them identically.

**Seasonal naive.** Step `h` ahead of `n` observations returns
`y[n + h - 1 - m * ceil(h / m)]`, the last observed value from the same season.
Horizons beyond one season wrap to the most recent full season. Requires `m`
observations.

**Holt-Winters.** `statsmodels.tsa.holtwinters.ExponentialSmoothing` with additive
trend (damped), additive seasonality, `seasonal_periods = m`, estimated initialisation.
Requires `2m` observations. A zero-variance series returns a constant without fitting;
a non-finite forecast raises instead of being silently replaced.

**Lag gradient boosting.** One `HistGradientBoostingRegressor` per horizon (direct
strategy). The feature row at origin `t` is built from `y[:t]` only:

- lags 1 to 12 and rolling means over 3 and 12 periods, each divided by the trailing
  12-period level at `t` (level falls back to 1.0 on a zero series);
- season index `t mod m`;
- the level itself.

The target is `y[t + h - 1] / level_t`, standardised by a `TrainingOnlyScaler` fitted on
training rows only, and the prediction is rescaled and floored at zero. The
per-row level normalisation lets a tree model follow a trend it cannot otherwise
extrapolate. Tests perturb `y[t:]` and check that the feature row at `t`, the scaler
statistics and the fitted forecaster's predictions do not change.

Settings live in `configs/backtest.yaml`: 100 iterations, learning rate 0.08, 15 leaf
nodes, minimum 5 samples per leaf, `random_state` 42. Fitting is pinned to one thread
because training windows have tens of rows and thread start-up otherwise dominates.

## 2. Rolling-origin backtest

Origins are `o = min_train, min_train + step, ...` while `o < n`. For each origin every
model is fitted on `y[:o]` and forecasts `y[o .. o+H-1]`, clipped at the series end. A
spy forecaster in the tests records what it was given and the test asserts it equals
`y[:o]` exactly for every fold.

**Intervals are point in time.** For a fold at origin `o` and horizon `h`, the residual
pool is the set of earlier out-of-sample residuals at horizon `h` whose target index is
strictly below `o`, so every residual comes from an observation the forecaster could
have seen. Bounds are `forecast + quantile(pool, alpha/2)` and
`forecast + quantile(pool, 1 - alpha/2)` with `alpha = 0.2`. Until the pool holds
`min_residuals = 8` values the bounds are reported as null, not fabricated, and coverage
is computed only over rows that have an interval. The pool size is stored in every
record. The final forecast from the last observation uses all backtest residuals per
horizon, which are all in its past.

Because the pool is empirical and one-sided bias is not removed, a biased model's
interval is not centred on its point forecast. That is intended: the band describes
where actuals fell relative to that model's forecasts.

## 3. Metrics

Per model, overall and by horizon:

- **MAE**: mean absolute error.
- **WAPE**: `sum|e| / sum|y|`; returned as null when the actuals sum to zero. A test
  covers the all-zero series.
- **MASE**: `MAE / d_o` where `d_o = mean |y_t - y_{t-m}|` over the training window of
  each fold. The per-fold denominator is written to every record and its mean is
  reported next to the MASE value. When a denominator is zero the fold's rows are
  excluded; when all are zero the metric is null.
- **Interval coverage**: share of actuals inside `[lo, hi]` over rows with an interval.

**Why MAPE is not a default metric.** MAPE divides each error by its own actual. One
zero month makes it undefined and near-zero months dominate the average, which is common
for intermittent SKU demand and for shocked series; it also penalises over-forecasts
more than under-forecasts of the same size. WAPE keeps a percentage interpretation
without those failure modes and MASE makes series of different scale comparable with a
denominator that is written down. `metrics.mape` exists for users who need it and
returns null when any actual is zero; it is not in `DEFAULT_METRICS`.

## 4. Replenishment simulator

Periodic review, integer units. Policy `(s, S)`: when the inventory position
`on_hand + on_order - backorders` is at or below `s` at a review, order up to `S`.
Policy `(s, Q)`: order a fixed `Q` instead. Review every `R` periods. Lead time is a
fixed integer or a discrete distribution sampled with the run seed.

Event order inside period `t`:

1. review and place an order due at `t + L` (only when `t mod R == 0`);
2. receive every order with due date `<= t` (lead time 0 arrives in the same period it
   was placed; lead time 1 arrives the next period);
3. clear backorders from available stock (backorder mode);
4. fill demand from on-hand stock; unmet demand is lost or backordered;
5. charge holding cost on end-of-period on-hand stock.

Balance identity tested on every ledger row:
`on_hand_end = on_hand_open + received - backorders_cleared - filled`, together with
on-hand continuity between periods and on-order accounting
`on_order_end = on_order_prev + ordered - received`.

Outputs per run: units demanded and filled, fill rate (immediate fill, null when no
demand), stockout periods, period service level `1 - stockouts / n`, order count,
holding cost, ordering cost, stockout cost and their total.

**Cost assumptions** come from `configs/assumptions.yaml` and are fictional demonstration
values in abstract currency units: unit cost 10, holding rate 0.005 of unit cost per
unit per week, 50 per order, 4 per unit of unmet demand, lost-sales mode. Changing that
file changes every cost in the outputs; nothing is hard-coded.

**Safety stock helpers**: `safety_stock = z * sigma * sqrt(L + R)` and
`reorder_point = ceil(mu * (L + R) + safety_stock)`, for users who want to derive `s`
from demand statistics rather than sweep it.

**Demand shock**: multiply demand by a factor over `[start, end)`. `shock_comparison`
runs one policy with and without the shock and reports both summaries.

## 5. Scenario grid

`run_scenario_grid` sweeps `reorder_points x lead_times` from `configs/scenario_grid.yaml`
with `S = s + order_quantity`. Each cell has its own derived seed. The output is
`{schemaVersion: 1, mode: "synthetic-demo", grid: [...]}` where each cell carries
`reorder_point, order_up_to, lead_time, fill_rate, period_service_level,
stockout_periods, order_count, holding_cost, ordering_cost, stockout_cost, total_cost`.
Tests check the cell count, that two runs produce identical JSON, and that fill rate is
non-decreasing in `s` for a fixed lead time on the fixture.

## 6. Separation of data worlds

The forecasting layer may consume the MRTS-shaped fixture or a fetched Census series.
The simulator consumes SKU-level synthetic demand only. **Aggregate Census data is never
blended with the SKU simulation.** See `docs/data.md`.
