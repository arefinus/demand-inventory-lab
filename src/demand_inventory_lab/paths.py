"""Repository-relative paths. Works for an editable install and for running from a
checkout; nothing here depends on the machine the code is run on."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURE_DIR = REPO_ROOT / "examples" / "fixtures" / "v1"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"
DOCS_FIGURE_DIR = REPO_ROOT / "docs" / "figures"
CENSUS_CACHE_DIR = REPO_ROOT / ".census_cache"

MRTS_FIXTURE = FIXTURE_DIR / "mrts_synthetic.csv"
SKU_FIXTURE = FIXTURE_DIR / "sku_demand.csv"
