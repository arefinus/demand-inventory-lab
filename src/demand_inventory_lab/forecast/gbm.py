"""Direct multi-horizon gradient boosting forecaster on lag features.

One HistGradientBoostingRegressor per horizon. Every transform (level
normalisation, target standardisation) is fitted on the training window only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..errors import InsufficientHistoryError
from .features import FeatureSpec, TrainingOnlyScaler, build_training_matrix, features_at


@dataclass(frozen=True)
class GBMConfig:
    max_horizon: int = 6
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    max_iter: int = 100
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 5
    random_state: int = 42

    def to_dict(self) -> dict:
        return {
            "max_horizon": self.max_horizon,
            "lags": list(self.feature_spec.lags),
            "rolling_windows": list(self.feature_spec.rolling_windows),
            "season_length": self.feature_spec.season_length,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }


class LagGBMForecaster:
    """fit() on a training series; predict() from the end of a history array."""

    def __init__(self, config: GBMConfig | None = None) -> None:
        self.config = config or GBMConfig()
        self._models: dict[int, HistGradientBoostingRegressor] = {}
        self._scalers: dict[int, TrainingOnlyScaler] = {}
        self._n_train: int | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self._models)

    def fit(self, y_train) -> "LagGBMForecaster":
        arr = np.asarray(y_train, dtype=float).ravel()
        spec = self.config.feature_spec
        needed = spec.min_history + self.config.max_horizon + self.config.min_samples_leaf
        if arr.size < needed:
            raise InsufficientHistoryError(
                f"GBM forecaster needs at least {needed} training observations for horizon "
                f"{self.config.max_horizon}; got {arr.size}."
            )
        self._models.clear()
        self._scalers.clear()
        # Training windows here have tens of rows; OpenMP thread start-up dominates
        # the fit time, so the histogram builder is pinned to one thread.
        with threadpool_limits(limits=1):
            for h in range(1, self.config.max_horizon + 1):
                X, target, _level, _origins = build_training_matrix(arr, h, spec)
                scaler = TrainingOnlyScaler().fit(target)
                model = HistGradientBoostingRegressor(
                    max_iter=self.config.max_iter,
                    learning_rate=self.config.learning_rate,
                    max_leaf_nodes=self.config.max_leaf_nodes,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                )
                model.fit(X, scaler.transform(target))
                self._models[h] = model
                self._scalers[h] = scaler
        self._n_train = int(arr.size)
        return self

    def predict(self, y_history, horizon: int | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        arr = np.asarray(y_history, dtype=float).ravel()
        H = horizon or self.config.max_horizon
        if H > self.config.max_horizon:
            raise ValueError(f"horizon {H} exceeds fitted max_horizon {self.config.max_horizon}")
        x = features_at(arr, arr.size, self.config.feature_spec).reshape(1, -1)
        level = float(x[0, -1])
        out = np.empty(H, dtype=float)
        for h in range(1, H + 1):
            z = self._models[h].predict(x)
            ratio = self._scalers[h].inverse_transform(z)[0]
            out[h - 1] = max(0.0, ratio * level)
        return out

    def fitted_statistics(self) -> dict[int, dict[str, float]]:
        return {h: s.fitted_statistics() for h, s in self._scalers.items()}


def gbm_forecast_factory(config: GBMConfig):
    """Adapter to the ``forecast_fn(y_train, horizon)`` interface used by the backtest."""

    def forecast_fn(y_train, horizon: int) -> np.ndarray:
        cfg = config if config.max_horizon == horizon else replace(config, max_horizon=horizon)
        return LagGBMForecaster(cfg).fit(y_train).predict(y_train, horizon)

    return forecast_fn
