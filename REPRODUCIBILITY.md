# Reproducibility

Release date: 2026-09-27. Python 3.10 locally; CI runs 3.10, 3.11 and 3.12.

## Exact commands

```
python -m pip install -e .
python -m pip install pytest

python tools/make_fixture.py --seed 42          # regenerates examples/fixtures/v1/mrts_synthetic.csv
python tools/make_sku_fixture.py --seed 42      # regenerates examples/fixtures/v1/sku_demand.csv
python -m demand_inventory_lab smoke            # examples/output/smoke/ (gitignored)
python -m demand_inventory_lab demo             # examples/output/ + docs/figures/
python -m demand_inventory_lab evaluate --records examples/output/backtest_records.csv
python -m pytest -q
```

## Seeds and determinism

| Component | Seed | Notes |
|---|---|---|
| MRTS-shaped fixture | 42 | `numpy.random.default_rng(42)`; regenerating gives byte-identical CSV |
| SKU fixture | 42 | same generator; a test compares the committed file with a fresh build |
| Lag GBM | `random_state: 42` in `configs/backtest.yaml` | HistGradientBoosting is deterministic for a seed and thread count; the fit is pinned to one thread |
| Holt-Winters | none needed | statsmodels' optimiser is deterministic for given data |
| Scenario grid | 42 | each grid cell derives its own seed from (seed, reorder point, lead time), so adding a cell never changes another cell |
| Simulator | argument `seed` | only consumed when the lead time is a distribution |

Seasonal naive has no fitted parameters.

## What is reproduced

Nothing. There is no source paper and no real dataset run. Every number in
`examples/output/` and in the docs carries the evidence label `demo` and the sentence
"computed by `make demo` on synthetic fixture v1, seed 42, not a benchmark result".

## What can drift between machines

- Holt-Winters point forecasts can differ in the last digits across BLAS builds. The
  tests check shape, finiteness and behaviour, not exact values.
- The GBM is pinned to a single OpenMP thread during fitting so that the histogram
  builder is deterministic; without the pin, sums over bins can differ by rounding.
- Timestamps and `run_id` in `run_manifest.yaml` change on every run by design.

## Run manifest

Every `smoke` and `demo` run writes `run_manifest.yaml` (schema in the shared contract):
fixture SHA-256 hashes, sample counts, configuration hash, seed, Python version and
`git_commit: null` (filled at push time, never fabricated).
