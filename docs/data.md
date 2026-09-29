# Data

See `DATA_CARD.md` for the full card. This page covers how the code treats each source.

## Two sources, kept apart

| | Synthetic MRTS-shaped series | Synthetic SKU demand | Census MRTS (optional) |
|---|---|---|---|
| File | `examples/fixtures/v1/mrts_synthetic.csv` | `examples/fixtures/v1/sku_demand.csv` | `.census_cache/*.json` (local only) |
| Loader | `io.load_mrts_table`, `io.select_series` | `io.load_sku_table`, `io.select_sku_demand` | `census.fetch_mrts`, `census.parse_response` |
| Consumer | `forecast/` | `inventory/` | `forecast/` |
| Grain | category-month, national aggregate shape | SKU-store-week | category-month, national aggregate |

The simulator's `simulate()` takes a demand array and the scenario commands read that
array from `sku_demand.csv` only. There is no code path from a Census payload or from
`mrts_synthetic.csv` into `inventory/`. A national retail sales series is not a SKU's
demand and scaling it down would only manufacture a false resemblance.

## Schema validation

Loaders fail fast with a message naming the missing column, the missing file and the
command that regenerates it. `select_series` lists the available category codes when a
requested one is absent. Negative SKU demand is rejected. The Census parser rejects a
header without the required fields, non-numeric `cell_value` entries and malformed
`time` values.

## Fixture regeneration

```
python tools/make_fixture.py --seed 42
python tools/make_sku_fixture.py --seed 42
```

Both scripts are deterministic; `tests/test_scenarios_and_fixture.py` checks that the
committed SKU fixture equals a fresh build and that identifiers use the `SYN-` prefix.

## Census cache entry

```json
{
  "key": "<sha256 of the request URL without key, first 24 hex>",
  "url": "<request URL with no API key>",
  "retrieved_at": "<UTC ISO timestamp>",
  "sha256": "<sha256 of the JSON payload>",
  "vintage_note": "Census MRTS estimates are revised after first publication ...",
  "rows": [["cell_value", "data_type_code", "..."], ["...", "..."]]
}
```

The API key is appended to the request only in memory and never appears in the cache
file; a test asserts this.
