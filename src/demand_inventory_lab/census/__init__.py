"""Client for the US Census Monthly Retail Trade Survey (MRTS) time series API.

This is aggregate national data. It is used only as an optional real series for
the forecasting layer. It is never used as, or blended with, SKU demand in the
inventory simulator.
"""

from __future__ import annotations

from .cache import CensusCache, CacheEntry
from .client import BASE_URL, TERMS_URL, MRTSQuery, build_url, fetch_mrts
from .schema import parse_response

__all__ = [
    "BASE_URL",
    "TERMS_URL",
    "CacheEntry",
    "CensusCache",
    "MRTSQuery",
    "build_url",
    "fetch_mrts",
    "parse_response",
]
