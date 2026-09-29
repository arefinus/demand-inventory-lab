from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_inventory_lab import paths  # noqa: E402
from demand_inventory_lab.io import load_mrts_table, select_series  # noqa: E402

SEASON = 12
BASE_PATTERN = np.array([10.0, 12.0, 15.0, 14.0, 13.0, 16.0, 18.0, 17.0, 14.0, 12.0, 11.0, 20.0])


@pytest.fixture
def periodic() -> np.ndarray:
    """Exactly periodic series, five seasons long."""
    return np.tile(BASE_PATTERN, 5)


@pytest.fixture
def trending_seasonal() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(120)
    return 100.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / SEASON) + rng.normal(0.0, 2.0, size=120)


@pytest.fixture(scope="session")
def mrts_series() -> tuple[np.ndarray, object]:
    table = load_mrts_table(paths.MRTS_FIXTURE)
    return select_series(table, "SYN-RET-01")


@pytest.fixture
def repo_root() -> Path:
    return ROOT
