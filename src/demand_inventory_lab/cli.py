"""argparse entry point: python -m demand_inventory_lab <subcommand>.

Exit codes: 0 ok; 1 failure or schema mismatch; 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PROJECT_ID, paths
from .errors import DemandInventoryError, MissingDataError
from .manifest import utc_now

SMOKE_OVERRIDES = {"min_train": 96, "step": 8}
SMOKE_GRID_OVERRIDES = {"reorder_points": [60, 120, 180], "lead_times": [1, 3]}


def _p(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def cmd_smoke(args: argparse.Namespace) -> int:
    from . import commands

    out = Path(args.out)
    started = utc_now()
    bt = commands.run_backtest(out, overrides=SMOKE_OVERRIDES)
    sim = commands.run_simulation(out, overrides=SMOKE_GRID_OVERRIDES)
    problems = commands.check_outputs(out)
    status = "completed" if not problems else "failed"
    commands.write_run_manifest(out, "smoke", status, started, bt["config_hash"] + sim["config_hash"][:8], 42,
                                notes=("smoke uses a coarser fold step than demo",))
    for pr in problems:
        _p(f"SCHEMA PROBLEM: {pr}")
    if problems:
        return 1
    _p(f"smoke ok: {bt['summary']['n_folds']} folds x 3 models, {len(sim['grid']['grid'])} scenario cells -> {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from . import commands, figures
    from .io import load_mrts_table

    out = Path(args.out)
    started = utc_now()
    bt = commands.run_backtest(out)
    sim = commands.run_simulation(out)
    problems = commands.check_outputs(out)
    if problems:
        for pr in problems:
            _p(f"SCHEMA PROBLEM: {pr}")
        commands.write_run_manifest(out, "demo", "failed", started, bt["config_hash"], 42)
        return 1
    table = load_mrts_table(paths.MRTS_FIXTURE)
    hist = table[table["category_code"] == commands.DEFAULT_CATEGORY]
    figs = figures.render_all(out, Path(args.figures), hist)
    commands.write_run_manifest(out, "demo", "completed", started, bt["config_hash"] + sim["config_hash"][:8], 42)
    for m, v in bt["summary"]["metrics"].items():
        o = v["overall"]
        _p(f"{m:15s} MAE={o['mae']:.1f} WAPE={_fmt(o['wape'])} MASE={_fmt(o['mase'])} coverage80={_fmt(o['interval_coverage'])} (n={o['n']})")
    _p(f"scenario cells: {len(sim['grid']['grid'])}; figures: {len(figs)} -> {args.figures}")
    _p("All numbers above were computed on synthetic fixture v1 (seed 42); they are not benchmark results.")
    return 0


def _fmt(v) -> str:
    return "undefined" if v is None else f"{v:.3f}"


def cmd_backtest(args: argparse.Namespace) -> int:
    from . import commands

    overrides = {k: v for k, v in (("min_train", args.min_train), ("step", args.step)) if v is not None}
    bt = commands.run_backtest(Path(args.out), fixture=Path(args.fixture), category=args.category, overrides=overrides)
    _p(json.dumps({m: v["overall"] for m, v in bt["summary"]["metrics"].items()}, indent=2))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from . import commands

    sim = commands.run_simulation(Path(args.out), fixture=Path(args.fixture))
    _p(json.dumps({"cells": len(sim["grid"]["grid"]), "shock_comparison": sim["comparison"] is not None}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import commands

    _p(json.dumps(commands.evaluate_records(Path(args.records)), indent=2))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from . import figures
    from .io import load_mrts_table

    table = load_mrts_table(paths.MRTS_FIXTURE)
    hist = table[table["category_code"] == "SYN-RET-01"]
    for p in figures.render_all(Path(args.out), Path(args.figures), hist):
        _p(f"wrote {p}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .census import TERMS_URL, CensusCache, MRTSQuery, build_url, fetch_mrts, parse_response
    from .census.client import api_key_from_env

    _p(f"US Census Bureau API terms of service: {TERMS_URL}")
    _p("By continuing you accept those terms. Aggregate MRTS data is never combined with the SKU simulation.")
    q = MRTSQuery(category_code=args.category, data_type_code=args.data_type, seasonally_adj=args.seasonally_adj, time_from=args.time_from)
    if args.dry_run:
        _p(f"dry run; would request: {build_url(q)}")
        return 0
    key = api_key_from_env()
    entry, cached = fetch_mrts(q, CensusCache(Path(args.cache_dir)), api_key=key, force=args.force)
    df = parse_response(entry.rows)
    _p(f"{'cache hit' if cached else 'fetched'}: {len(df)} rows; retrieved_at={entry.retrieved_at}; sha256={entry.sha256[:16]}")
    _p(entry.vintage_note)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m {PROJECT_ID.replace('-', '_')}", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="fast offline check on fixtures; exit 1 on schema mismatch")
    s.add_argument("--out", default=str(paths.OUTPUT_DIR / "smoke"))
    s.set_defaults(func=cmd_smoke)

    d = sub.add_parser("demo", help="full demonstration on fixtures; writes examples/output/ and docs/figures/")
    d.add_argument("--out", default=str(paths.OUTPUT_DIR))
    d.add_argument("--figures", default=str(paths.DOCS_FIGURE_DIR))
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("backtest", help="rolling-origin backtest on one series")
    b.add_argument("--fixture", default=str(paths.MRTS_FIXTURE))
    b.add_argument("--category", default="SYN-RET-01")
    b.add_argument("--out", default=str(paths.OUTPUT_DIR))
    b.add_argument("--min-train", type=int, default=None)
    b.add_argument("--step", type=int, default=None)
    b.set_defaults(func=cmd_backtest)

    si = sub.add_parser("simulate", help="scenario grid and shock comparison on the SKU fixture")
    si.add_argument("--fixture", default=str(paths.SKU_FIXTURE))
    si.add_argument("--out", default=str(paths.OUTPUT_DIR))
    si.set_defaults(func=cmd_simulate)

    e = sub.add_parser("evaluate", help="recompute metrics from a saved backtest records table")
    e.add_argument("--records", default=str(paths.OUTPUT_DIR / "backtest_records.csv"))
    e.set_defaults(func=cmd_evaluate)

    f = sub.add_parser("figures", help="render docs/figures/*.svg from examples/output/")
    f.add_argument("--out", default=str(paths.OUTPUT_DIR))
    f.add_argument("--figures", default=str(paths.DOCS_FIGURE_DIR))
    f.set_defaults(func=cmd_figures)

    fe = sub.add_parser("fetch", help="fetch one Census MRTS series (network; prints terms URL first)")
    fe.add_argument("--category", required=True, help="MRTS category_code, for example 44X72")
    fe.add_argument("--data-type", default="SM")
    fe.add_argument("--seasonally-adj", default="no")
    fe.add_argument("--time-from", default="2010")
    fe.add_argument("--cache-dir", default=str(paths.CENSUS_CACHE_DIR))
    fe.add_argument("--force", action="store_true")
    fe.add_argument("--dry-run", action="store_true", help="print the request URL and exit without network access")
    fe.set_defaults(func=cmd_fetch)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MissingDataError as exc:
        sys.stderr.write(f"missing input: {exc}\n")
        return 2
    except DemandInventoryError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
