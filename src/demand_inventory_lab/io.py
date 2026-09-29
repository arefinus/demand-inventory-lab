"""Fixture loading with schema validation. Fails fast with an actionable message."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .errors import MissingDataError, SchemaError

MRTS_COLUMNS = ("category_code", "data_type_code", "seasonally_adj", "time", "cell_value")
SKU_COLUMNS = ("sku_id", "store_id", "week_index", "week_start", "demand")


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise MissingDataError(f"Required file not found: {path}. {hint}")


def load_yaml(path: Path) -> dict:
    _require(path, "Check the configs/ directory of the repository.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a YAML mapping at the top level.")
    return data


def load_mrts_table(path: Path) -> pd.DataFrame:
    _require(path, "Generate it with: python tools/make_fixture.py --seed 42")
    df = pd.read_csv(path, dtype={"category_code": str, "data_type_code": str, "seasonally_adj": str})
    missing = [c for c in MRTS_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns {missing}; expected {list(MRTS_COLUMNS)}.")
    if df.empty:
        raise SchemaError(f"{path} contains no rows.")
    df["time"] = pd.PeriodIndex(df["time"].astype(str), freq="M").to_timestamp()
    df["cell_value"] = pd.to_numeric(df["cell_value"], errors="raise").astype(float)
    return df.sort_values(["category_code", "time"]).reset_index(drop=True)


def select_series(df: pd.DataFrame, category_code: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    sub = df[df["category_code"] == category_code]
    if sub.empty:
        available = sorted(df["category_code"].unique().tolist())
        raise SchemaError(f"category_code {category_code!r} not in table; available: {available}")
    sub = sub.sort_values("time")
    return sub["cell_value"].to_numpy(dtype=float), pd.DatetimeIndex(sub["time"])


def load_sku_table(path: Path) -> pd.DataFrame:
    _require(path, "Generate it with: python tools/make_sku_fixture.py --seed 42")
    df = pd.read_csv(path, dtype={"sku_id": str, "store_id": str})
    missing = [c for c in SKU_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns {missing}; expected {list(SKU_COLUMNS)}.")
    if (df["demand"] < 0).any():
        raise SchemaError(f"{path} contains negative demand values.")
    return df.sort_values(["sku_id", "store_id", "week_index"]).reset_index(drop=True)


def select_sku_demand(df: pd.DataFrame, sku_id: str, store_id: str) -> np.ndarray:
    sub = df[(df["sku_id"] == sku_id) & (df["store_id"] == store_id)].sort_values("week_index")
    if sub.empty:
        raise SchemaError(f"No rows for sku_id={sku_id!r} store_id={store_id!r} in SKU fixture.")
    return sub["demand"].to_numpy(dtype=int)
