"""Parse the Census API list-of-lists payload into a typed monthly table."""

from __future__ import annotations

import pandas as pd

from ..errors import SchemaError

REQUIRED_FIELDS = ("cell_value", "data_type_code", "time_slot_id", "category_code", "seasonally_adj", "time")


def parse_response(rows: list[list[str]]) -> pd.DataFrame:
    """``rows[0]`` is the header. Returns a frame sorted by time with numeric
    ``cell_value`` and a monthly ``time`` timestamp. Raises SchemaError when the
    header lacks required fields or values are not numeric."""
    if not rows or not isinstance(rows[0], list):
        raise SchemaError("Census payload is empty or not a list of rows.")
    header = [str(h) for h in rows[0]]
    missing = [f for f in REQUIRED_FIELDS if f not in header]
    if missing:
        raise SchemaError(f"Census payload header is missing fields {missing}; got {header}.")
    body = rows[1:]
    if not body:
        raise SchemaError("Census payload has a header but no data rows.")
    df = pd.DataFrame(body, columns=header)
    try:
        df["cell_value"] = pd.to_numeric(df["cell_value"].replace({"": None}), errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"cell_value contains non-numeric entries: {exc}") from exc
    try:
        df["time"] = pd.PeriodIndex(df["time"].astype(str), freq="M").to_timestamp()
    except Exception as exc:  # pandas raises several types here
        raise SchemaError(f"time column is not in YYYY-MM form: {exc}") from exc
    keep = ["category_code", "data_type_code", "seasonally_adj", "time", "cell_value"]
    return df[keep].sort_values(["category_code", "time"]).reset_index(drop=True)
