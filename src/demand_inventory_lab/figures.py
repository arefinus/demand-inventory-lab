"""Render docs/figures/*.svg from committed demo output. Every figure states that
it was computed on the synthetic fixture and is not a benchmark result."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .errors import MissingDataError
from .svg import PALETTE, Canvas, Frame

NOTE = "Computed by `make demo` on synthetic fixture v1, seed 42. Not a benchmark result."


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise MissingDataError(f"{path} not found. Run `python -m demand_inventory_lab demo` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def fig_mae_by_horizon(summary: dict, out: Path) -> Path:
    models = sorted(summary["metrics"].keys())
    horizons = sorted(int(h) for h in next(iter(summary["metrics"].values()))["by_horizon"].keys())
    vals = {m: [summary["metrics"][m]["by_horizon"][str(h)]["mae"] for h in horizons] for m in models}
    ymax = max(max(v) for v in vals.values()) * 1.15 or 1.0
    c = Canvas(Frame(), "Backtest MAE by horizon (synthetic MRTS-shaped series)")
    c.note(NOTE)
    c.axes("forecast horizon (months ahead)", "MAE (abstract units)")
    c.yticks(0, ymax, fmt="{:.0f}")
    gw = c.f.plot_w / len(horizons)
    bw = gw / (len(models) + 1)
    for i, h in enumerate(horizons):
        x0 = c.f.left + i * gw + bw / 2
        c.xtick(c.f.left + i * gw + gw / 2, str(h))
        for j, m in enumerate(models):
            top = c.y(vals[m][i], 0, ymax)
            c.rect(x0 + j * bw, top, bw * 0.9, c.f.top + c.f.plot_h - top, PALETTE[j % len(PALETTE)])
    c.legend([(m, PALETTE[j % len(PALETTE)]) for j, m in enumerate(models)], x=c.f.left + c.f.plot_w - 150)
    out.write_text(c.render(), encoding="utf-8")
    return out


def fig_forecast_fan(forecast: dict, history: pd.DataFrame, model: str, out: Path) -> Path:
    rows = forecast["models"][model]
    hist = history.tail(24)
    ys = hist["cell_value"].tolist()
    fut = [r["point"] for r in rows]
    los = [r["lo"] for r in rows if r["lo"] is not None]
    his = [r["hi"] for r in rows if r["hi"] is not None]
    lo_all, hi_all = min(ys + fut + los), max(ys + fut + his)
    pad = (hi_all - lo_all) * 0.1 or 1.0
    ylo, yhi = lo_all - pad, hi_all + pad
    n_hist, n_fut = len(ys), len(fut)
    xmax = n_hist + n_fut - 1
    c = Canvas(Frame(), f"Forecast with 80% empirical intervals: {model}")
    c.note(NOTE)
    c.axes("month (last 24 observed, then forecast steps)", "value (abstract units)")
    c.yticks(ylo, yhi, fmt="{:.0f}")
    labels = hist["time"].dt.strftime("%Y-%m").tolist() + [r["date"] or f"+{r['step']}" for r in rows]
    for i in range(0, len(labels), 6):
        c.xtick(c.x(i, 0, xmax), labels[i])
    c.polyline([(c.x(i, 0, xmax), c.y(v, ylo, yhi)) for i, v in enumerate(ys)], PALETTE[0])
    band = [r for r in rows if r["lo"] is not None]
    if band:
        upper = [(c.x(n_hist + r["step"] - 1, 0, xmax), c.y(r["hi"], ylo, yhi)) for r in band]
        lower = [(c.x(n_hist + r["step"] - 1, 0, xmax), c.y(r["lo"], ylo, yhi)) for r in reversed(band)]
        c.polygon(upper + lower, PALETTE[1], 0.25)
    c.polyline([(c.x(n_hist + i, 0, xmax), c.y(v, ylo, yhi)) for i, v in enumerate(fut)], PALETTE[1], dash="5,3")
    c.legend([("observed", PALETTE[0]), ("forecast (80% band)", PALETTE[1])])
    out.write_text(c.render(), encoding="utf-8")
    return out


def fig_fill_rate_grid(grid: dict, out: Path) -> Path:
    cells = grid["grid"]
    lts = sorted({g["lead_time"] for g in cells})
    sps = sorted({g["reorder_point"] for g in cells})
    c = Canvas(Frame(), "Fill rate by reorder point and lead time (synthetic SKU)")
    c.note(NOTE)
    c.axes("reorder point s (units); order-up-to S = s + Q", "fill rate")
    c.yticks(0.0, 1.0, fmt="{:.1f}")
    for sp in sps:
        c.xtick(c.x(sp, sps[0], sps[-1]), str(sp))
    for j, lt in enumerate(lts):
        pts = [(c.x(g["reorder_point"], sps[0], sps[-1]), c.y(g["fill_rate"] or 0.0, 0, 1)) for g in cells if g["lead_time"] == lt]
        pts.sort()
        c.polyline(pts, PALETTE[j % len(PALETTE)])
        for x, y in pts:
            c.circle(x, y, 3, PALETTE[j % len(PALETTE)])
    c.legend([(f"lead time {lt}", PALETTE[j % len(PALETTE)]) for j, lt in enumerate(lts)], x=c.f.left + c.f.plot_w - 130, y=c.f.top + c.f.plot_h - 70)
    out.write_text(c.render(), encoding="utf-8")
    return out


def fig_cost_frontier(grid: dict, out: Path) -> Path:
    cells = grid["grid"]
    lts = sorted({g["lead_time"] for g in cells})
    hmax = max(g["holding_cost"] for g in cells) * 1.1 or 1.0
    c = Canvas(Frame(), "Holding cost versus fill rate (each point is one scenario)")
    c.note(NOTE)
    c.axes("fill rate", "holding cost (abstract currency units)")
    c.yticks(0, hmax, fmt="{:.0f}")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        c.xtick(c.x(v, 0.5, 1.0), f"{v:.1f}")
    for j, lt in enumerate(lts):
        for g in cells:
            if g["lead_time"] == lt and g["fill_rate"] is not None:
                c.circle(c.x(max(0.5, g["fill_rate"]), 0.5, 1.0), c.y(g["holding_cost"], 0, hmax), 4, PALETTE[j % len(PALETTE)])
    c.legend([(f"lead time {lt}", PALETTE[j % len(PALETTE)]) for j, lt in enumerate(lts)])
    out.write_text(c.render(), encoding="utf-8")
    return out


def render_all(output_dir: Path, figure_dir: Path, history: pd.DataFrame) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_json(output_dir / "backtest_summary.json")
    forecast = _load_json(output_dir / "forecast.json")
    grid = _load_json(output_dir / "scenario_grid.json")
    return [
        fig_mae_by_horizon(summary, figure_dir / "backtest_mae_by_horizon.svg"),
        fig_forecast_fan(forecast, history, "lag_gbm", figure_dir / "forecast_lag_gbm.svg"),
        fig_fill_rate_grid(grid, figure_dir / "scenario_fill_rate.svg"),
        fig_cost_frontier(grid, figure_dir / "scenario_cost_frontier.svg"),
    ]
