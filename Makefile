PY ?= python

.PHONY: setup smoke demo test evaluate backtest simulate figures fetch fixtures

setup:
	pip install -e . && pip install pytest

smoke:
	$(PY) -m demand_inventory_lab smoke

demo:
	$(PY) -m demand_inventory_lab demo

test:
	$(PY) -m pytest -q

evaluate:
	$(PY) -m demand_inventory_lab evaluate --records examples/output/backtest_records.csv

backtest:
	$(PY) -m demand_inventory_lab backtest

simulate:
	$(PY) -m demand_inventory_lab simulate

figures:
	$(PY) -m demand_inventory_lab figures

fixtures:
	$(PY) tools/make_fixture.py --seed 42
	$(PY) tools/make_sku_fixture.py --seed 42

# Public API; prints the Census terms of service URL first. Never run in CI or tests.
fetch:
	$(PY) -m demand_inventory_lab fetch
