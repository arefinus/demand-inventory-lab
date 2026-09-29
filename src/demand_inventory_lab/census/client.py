"""Query builder and fetcher for the MRTS time series endpoint.

The network call is isolated behind a ``transport`` callable so tests exercise
URL construction, parsing and caching without any network access. The default
transport uses ``requests`` and is only reached from the ``fetch`` CLI command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from ..errors import SchemaError
from .cache import CacheEntry, CensusCache, cache_key

BASE_URL = "https://api.census.gov/data/timeseries/eits/mrts"
TERMS_URL = "https://www.census.gov/data/developers/about/terms-of-service.html"
API_KEY_ENV = "CENSUS_API_KEY"

Transport = Callable[[str], list[list[str]]]


@dataclass(frozen=True)
class MRTSQuery:
    """A single MRTS request. Defaults describe the query shape, not a real dataset
    shipped here; ``category_code`` must be supplied by the user."""

    category_code: str
    data_type_code: str = "SM"
    seasonally_adj: str = "no"
    time_from: str = "2010"
    time_to: str | None = None

    def params(self) -> list[tuple[str, str]]:
        time_expr = f"from {self.time_from}" + (f" to {self.time_to}" if self.time_to else "")
        return [
            ("get", "cell_value,data_type_code,time_slot_id,category_code,seasonally_adj"),
            ("for", "us:*"),
            ("time", time_expr),
            ("category_code", self.category_code),
            ("data_type_code", self.data_type_code),
            ("seasonally_adj", self.seasonally_adj),
        ]


def build_url(query: MRTSQuery, api_key: str | None = None) -> str:
    params = query.params()
    if api_key:
        params.append(("key", api_key))
    return f"{BASE_URL}?{urlencode(params, safe=':*,')}"


def redacted_url(query: MRTSQuery) -> str:
    return build_url(query, api_key=None)


def api_key_from_env() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def _requests_transport(url: str) -> list[list[str]]:
    import requests

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise SchemaError(f"Census API returned HTTP {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    if not isinstance(payload, list):
        raise SchemaError("Census API payload is not a JSON list of rows.")
    return payload


def fetch_mrts(
    query: MRTSQuery,
    cache: CensusCache,
    transport: Transport | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> tuple[CacheEntry, bool]:
    """Return (entry, from_cache). Uses the cache unless ``force`` is set."""
    url_red = redacted_url(query)
    key = cache_key(url_red)
    if not force:
        hit = cache.get(key)
        if hit is not None:
            return hit, True
    fn = transport or _requests_transport
    rows = fn(build_url(query, api_key=api_key))
    if not rows or not isinstance(rows[0], list):
        raise SchemaError("Transport returned an empty or malformed payload.")
    return cache.put(key, url_red, rows), False


def default_cache(root: Path) -> CensusCache:
    return CensusCache(root)
