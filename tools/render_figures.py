"""Render docs/figures/*.svg from examples/output/. Thin wrapper around the package
renderer so the figures can be rebuilt without the CLI.

Usage: python tools/render_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demand_inventory_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["figures"]))
