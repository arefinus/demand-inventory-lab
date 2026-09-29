from __future__ import annotations

import json

import pytest

from demand_inventory_lab.census import TERMS_URL, CensusCache, MRTSQuery, build_url, fetch_mrts, parse_response
from demand_inventory_lab.census.client import API_KEY_ENV, api_key_from_env, redacted_url
from demand_inventory_lab.cli import main
from demand_inventory_lab.errors import SchemaError

HEADER = ["cell_value", "data_type_code", "time_slot_id", "category_code", "seasonally_adj", "time", "us"]
ROWS = [
    HEADER,
    ["120", "SM", "1", "SYN-RET-01", "no", "2020-02", "1"],
    ["100", "SM", "1", "SYN-RET-01", "no", "2020-01", "1"],
]


def test_build_url_contains_query_and_no_key_by_default():
    q = MRTSQuery(category_code="44X72", time_from="2015")
    url = build_url(q)
    assert url.startswith("https://api.census.gov/data/timeseries/eits/mrts?")
    assert "category_code=44X72" in url and "key=" not in url
    assert "from+2015" in url or "from%202015" in url


def test_api_key_is_appended_but_never_in_redacted_url(monkeypatch):
    q = MRTSQuery(category_code="44X72")
    monkeypatch.setenv(API_KEY_ENV, "not-a-real-key")
    key = api_key_from_env()
    assert key == "not-a-real-key"
    assert "key=not-a-real-key" in build_url(q, api_key=key)
    assert "not-a-real-key" not in redacted_url(q)
    monkeypatch.setenv(API_KEY_ENV, "   ")
    assert api_key_from_env() is None


def test_parse_response_returns_sorted_numeric_monthly_table():
    df = parse_response(ROWS)
    assert list(df.columns) == ["category_code", "data_type_code", "seasonally_adj", "time", "cell_value"]
    assert df["cell_value"].tolist() == [100.0, 120.0]
    assert str(df["time"].iloc[0])[:7] == "2020-01"


def test_parse_response_rejects_bad_header_and_non_numeric():
    with pytest.raises(SchemaError):
        parse_response([["cell_value", "time"], ["1", "2020-01"]])
    with pytest.raises(SchemaError):
        parse_response([HEADER, ["abc", "SM", "1", "X", "no", "2020-01", "1"]])
    with pytest.raises(SchemaError):
        parse_response([])


def test_fetch_caches_with_hash_date_and_vintage_and_never_stores_key(tmp_path):
    calls = []

    def transport(url):
        calls.append(url)
        return ROWS

    cache = CensusCache(tmp_path / "cache")
    q = MRTSQuery(category_code="SYN-RET-01")
    e1, hit1 = fetch_mrts(q, cache, transport=transport, api_key="secret-key")
    e2, hit2 = fetch_mrts(q, cache, transport=transport, api_key="secret-key")
    assert (hit1, hit2) == (False, True)
    assert len(calls) == 1 and "secret-key" in calls[0]
    assert e1.sha256 == e2.sha256 and len(e1.sha256) == 64
    assert e1.retrieved_at and "vintage" in e1.vintage_note.lower()
    on_disk = json.loads(cache.path_for(e1.key).read_text(encoding="utf-8"))
    assert "secret-key" not in json.dumps(on_disk)
    assert on_disk["sha256"] == e1.sha256


def test_fetch_force_refetches_and_rejects_empty_payload(tmp_path):
    calls = []

    def transport(url):
        calls.append(url)
        return ROWS

    cache = CensusCache(tmp_path)
    q = MRTSQuery(category_code="SYN-RET-01")
    fetch_mrts(q, cache, transport=transport)
    fetch_mrts(q, cache, transport=transport, force=True)
    assert len(calls) == 2
    with pytest.raises(SchemaError):
        fetch_mrts(MRTSQuery(category_code="OTHER"), cache, transport=lambda url: [])


def test_cli_fetch_dry_run_prints_terms_first_without_network(capsys):
    code = main(["fetch", "--category", "SYN-RET-01", "--dry-run"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert TERMS_URL in out[0]
    assert any("dry run" in line for line in out)
