"""On-disk JSON cache for Census responses with retrieval date, payload hash and a
data-vintage note. The API key is never written to disk."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VINTAGE_NOTE = (
    "Census MRTS estimates are revised after first publication (advance, preliminary, "
    "revised, annual benchmark). This payload reflects the vintage available on the "
    "retrieval date; a later retrieval of the same query may return different values."
)


@dataclass(frozen=True)
class CacheEntry:
    key: str
    url_redacted: str
    retrieved_at: str
    sha256: str
    vintage_note: str
    rows: list[list[str]]

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "url": self.url_redacted,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "vintage_note": self.vintage_note,
            "rows": self.rows,
        }


def payload_sha256(rows: list[list[str]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def cache_key(url_redacted: str) -> str:
    return hashlib.sha256(url_redacted.encode("utf-8")).hexdigest()[:24]


class CensusCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        p = self.path_for(key)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        return CacheEntry(
            key=d["key"],
            url_redacted=d["url"],
            retrieved_at=d["retrieved_at"],
            sha256=d["sha256"],
            vintage_note=d["vintage_note"],
            rows=d["rows"],
        )

    def put(self, key: str, url_redacted: str, rows: list[list[str]]) -> CacheEntry:
        entry = CacheEntry(
            key=key,
            url_redacted=url_redacted,
            retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sha256=payload_sha256(rows),
            vintage_note=VINTAGE_NOTE,
            rows=rows,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path_for(key).open("w", encoding="utf-8") as fh:
            json.dump(entry.to_dict(), fh, indent=1)
        return entry
