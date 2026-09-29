# Data card

## 1. Synthetic fixtures (shipped)

Location: `examples/fixtures/v1/`, described in `FIXTURE.md`. Authored synthetic
demonstration data, fictional identifiers, not derived from any person, company, account
or dataset. Evidence label for anything computed on them: `demo`.

### mrts_synthetic.csv

| Item | Value |
|---|---|
| Provider | this repository, `tools/make_fixture.py`, seed 42, version v1 |
| Terms | MIT (same as the code) |
| Record unit | one category-month |
| Rows | 360 (3 fictional categories x 120 months, 2016-01 to 2025-12) |
| Target | `cell_value` (abstract units, non-negative integer) |
| Columns | `category_code` (SYN-RET-01..03), `data_type_code` (SM), `seasonally_adj` (no), `time` (YYYY-MM), `cell_value` |
| Generation | level x (1 + trend)^t x seasonal factor (12-month sine plus a December bump) x (1 + Gaussian noise); rounded, floored at 0 |
| Splits | none stored; the rolling-origin backtest defines training windows `y[:o]` at run time |
| Missingness | none |

The shape mirrors a Census MRTS API response so that code written for the fixture also
runs on a fetched series. The values are not Census data.

### sku_demand.csv

| Item | Value |
|---|---|
| Provider | this repository, `tools/make_sku_fixture.py`, seed 42, version v1 |
| Record unit | one SKU-store-week |
| Rows | 2,496 (12 SKUs x 2 stores x 104 weeks from 2024-01-01) |
| Target | `demand` (non-negative integer units) |
| Columns | `sku_id` (SYN-SKU-0001..0012), `store_id` (SYN-STORE-01..02), `week_index`, `week_start`, `demand` |
| Generation | Poisson count with a gamma-mixed rate (negative binomial), mean following a 52-week sine cycle; per-SKU mean, dispersion and amplitude drawn once from the seeded generator |
| Splits | not applicable; the simulator consumes the whole series for one SKU-store pair |
| Missingness | none |

## 2. US Census Monthly Retail Trade Survey (optional, not shipped)

| Item | Value |
|---|---|
| Provider | US Census Bureau, Economic Indicators Time Series API, dataset `timeseries/eits/mrts` |
| Endpoint | https://api.census.gov/data/timeseries/eits/mrts |
| Terms | https://www.census.gov/data/developers/about/terms-of-service.html (printed by `make fetch` before any request) |
| Key | optional; read from environment variable `CENSUS_API_KEY`; never written to disk or logs |
| Record unit | one category-month at national level |
| Fields requested | `cell_value, data_type_code, time_slot_id, category_code, seasonally_adj`, plus `time` |
| Vintage | MRTS estimates are revised after first release. The cache stores the retrieval date and payload SHA-256 so a later fetch that differs is detectable. Comparisons across retrievals must state the vintage. |
| Storage | `.census_cache/<key>.json` (gitignored); the cache key is the SHA-256 of the request URL without the API key |
| Redistribution | none; no Census rows are committed |

How to obtain: run `python -m demand_inventory_lab fetch --category <MRTS category_code>`
(for example a NAICS-based code such as `44X72`). The command is not run in CI or tests.

## 3. Separation rule

Census data is national aggregate retail sales. The simulator models a single SKU at a
single location. These are different quantities at different scales, so **aggregate
Census data is never used as, scaled into, or blended with SKU demand**. The forecasting
layer accepts either source; the simulator accepts only the SKU fixture or a user's own
SKU-level series.
