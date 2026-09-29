"""Generate the synthetic SKU/store weekly demand fixture for the simulator.

Authored synthetic demonstration data. Fictional SKUs (SYN-SKU-0001 ...) and stores
(SYN-STORE-01 ...). Seed 42, 104 weeks. Independent of the MRTS-shaped fixture.

Usage: python tools/make_sku_fixture.py --seed 42 [--out examples/fixtures/v1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demand_inventory_lab.inventory.demand import DemandParams, generate_demand  # noqa: E402

N_SKUS = 12
N_STORES = 2
N_WEEKS = 104
WEEK_START = "2024-01-01"


def build(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(WEEK_START, periods=N_WEEKS, freq="7D").strftime("%Y-%m-%d")
    frames = []
    for i in range(1, N_SKUS + 1):
        base_mean = float(rng.uniform(8.0, 80.0))
        disp = float(rng.uniform(4.0, 20.0))
        amp = float(rng.uniform(0.05, 0.35))
        for j in range(1, N_STORES + 1):
            params = DemandParams(mean=base_mean * float(rng.uniform(0.7, 1.3)), dispersion=disp, seasonal_amplitude=amp)
            d = generate_demand(rng, N_WEEKS, params)
            frames.append(
                pd.DataFrame(
                    {
                        "sku_id": f"SYN-SKU-{i:04d}",
                        "store_id": f"SYN-STORE-{j:02d}",
                        "week_index": np.arange(N_WEEKS),
                        "week_start": weeks,
                        "demand": d,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "examples" / "fixtures" / "v1")
    args = ap.parse_args(argv)
    df = build(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "sku_demand.csv", index=False)
    print(f"wrote {args.out / 'sku_demand.csv'} ({len(df)} rows, {N_SKUS} SKUs x {N_STORES} stores x {N_WEEKS} weeks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
