"""Lag features for the gradient boosting forecaster.

Convention: ``features_at(y, t)`` uses only ``y[:t]`` (indices <= t-1) and is
the feature row for forecasting ``y[t + h - 1]`` at horizon h. Nothing here
looks at index t or later, which the tests check by perturbing ``y[t:]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import InsufficientHistoryError


@dataclass(frozen=True)
class FeatureSpec:
    lags: tuple[int, ...] = tuple(range(1, 13))
    rolling_windows: tuple[int, ...] = (3, 12)
    season_length: int = 12

    @property
    def min_history(self) -> int:
        return max(max(self.lags), max(self.rolling_windows))

    @property
    def names(self) -> tuple[str, ...]:
        lag_names = tuple(f"lag_{k}" for k in self.lags)
        roll_names = tuple(f"rollmean_{w}" for w in self.rolling_windows)
        return lag_names + roll_names + ("season_index", "level")


def trailing_level(y: np.ndarray, t: int, window: int) -> float:
    """Mean of the last ``window`` observations before t; 1.0 when that mean is not positive
    so that ratios fall back to raw values on zero series."""
    lvl = float(np.mean(y[t - window : t]))
    return lvl if lvl > 0.0 else 1.0


def features_at(y, t: int, spec: FeatureSpec) -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if t < spec.min_history:
        raise InsufficientHistoryError(f"features_at needs t >= {spec.min_history}; got t={t}.")
    if t > arr.size:
        raise InsufficientHistoryError(f"features_at needs t <= len(y)={arr.size}; got t={t}.")
    past = arr[:t]
    level = trailing_level(past, t, spec.season_length)
    lags = [past[t - k] / level for k in spec.lags]
    rolls = [float(np.mean(past[t - w : t])) / level for w in spec.rolling_windows]
    season_index = float(t % spec.season_length)
    return np.array(lags + rolls + [season_index, level], dtype=float)


def build_training_matrix(y, horizon: int, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows for the direct forecaster at ``horizon`` h.

    Returns (X, target_ratio, level, t) where the row built at origin t predicts
    y[t + h - 1] expressed as a ratio to the trailing level at t. Only origins
    whose target lies inside the series are included, so no row uses y[n:] or
    later.
    """
    arr = np.asarray(y, dtype=float).ravel()
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = arr.size
    origins = np.arange(spec.min_history, n - horizon + 1)
    if origins.size == 0:
        raise InsufficientHistoryError(
            f"Need at least {spec.min_history + horizon} observations to build horizon-{horizon} rows; got {n}."
        )
    X = np.vstack([features_at(arr, int(t), spec) for t in origins])
    level = X[:, -1]
    target = arr[origins + horizon - 1] / level
    return X, target, level, origins


@dataclass
class TrainingOnlyScaler:
    """Standardises the target ratio with statistics fitted on training rows only.

    Kept as a separate object so tests can prove the fitted mean and std do not
    change when rows after the training window are perturbed.
    """

    mean_: float | None = None
    std_: float | None = None

    def fit(self, values) -> "TrainingOnlyScaler":
        v = np.asarray(values, dtype=float).ravel()
        if v.size == 0:
            raise InsufficientHistoryError("Scaler needs at least one training value.")
        self.mean_ = float(np.mean(v))
        std = float(np.std(v))
        self.std_ = std if std > 0.0 else 1.0
        return self

    def transform(self, values) -> np.ndarray:
        self._check()
        return (np.asarray(values, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, values) -> np.ndarray:
        self._check()
        return np.asarray(values, dtype=float) * self.std_ + self.mean_

    def _check(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted; call fit() on training values first.")

    def fitted_statistics(self) -> dict[str, float]:
        self._check()
        return {"mean": float(self.mean_), "std": float(self.std_)}
