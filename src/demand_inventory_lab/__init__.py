"""demand_inventory_lab: forecasting baselines, rolling-origin backtests and a
synthetic replenishment simulator.

Two data worlds live in this package and are never blended:

* ``census``: an optional client for aggregate US Census Monthly Retail Trade
  data (public API, cached on disk). Used only to demonstrate the forecasting
  layer on a real aggregate series when the user chooses to fetch one.
* ``inventory``: a fully synthetic SKU/store demand fixture and a replenishment
  simulator. Aggregate Census data is never used as SKU demand.
"""

from __future__ import annotations

__version__ = "0.1.0"
PROJECT_ID = "demand-inventory-lab"
RELEASE_DATE = "2026-09-27"
