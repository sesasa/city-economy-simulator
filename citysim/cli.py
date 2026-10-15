"""Command line: run, sweep, compare, experiments.

  python -m citysim run scenarios/stage1_base.yaml
  python -m citysim sweep scenarios/stage1_base.yaml --param a=1:10:0.5 [--param s=0,0.3]
  python -m citysim compare scenarios/stage1_shock_storage.yaml scenarios/stage1_shock_no_storage.yaml
  python -m citysim experiments stage1
  python -m citysim twin            # animated town for non-specialists -> reports/grainbridge.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go

from . import report as R
from .config import load_scenario, parse_range
from .core import runner
from .core.metrics import write_frame

REPORTS = Path("reports")


def _out(args, default: str) -> Path:
    return Path(args.out) if args.out else REPORTS / default


def cmd_run(args) -> int:
    sc = load_scenario(args.scenario)
    if args.ticks is not None:
        sc = sc.model_copy(update={"ticks": args.ticks})
    res = runner.run(sc)
    out = _out(args, f"{sc.name}.html")
    write_frame(res.frame, out.with_suffix(".csv"))
    R.write(out, f"Run · {sc.name}", sc.description or "", R.run_report([res], sc.name))
    print(f"{sc.name}: {sc.ticks} ticks, invariants OK -> {out} (+ {out.with_suffix('.csv').name})")
    return 0


def cmd_sweep(args) -> int:
    sc = load_scenario(args.scenario)
    grid = dict(parse_range(p) for p in args.param)
    table, _ = runner.sweep(sc, grid)
    out = _out(args, f"{sc.name}_sweep.html")
    write_frame(table, out.with_suffix(".csv"))
    keys = list(grid)
    x = keys[0]
    metrics = [c for c in table.columns if c.startswith("final_")]
    parts = []
    for m in metrics:
        fig = go.Figure()
        groups = table.groupby(keys[1:]) if len(keys) > 1 else [("", table)]
        for i, (g, d) in enumerate(groups):
            fig.add_trace(go.Scatter(x=d[x], y=d[m], mode="lines+markers", name=str(g),
                                     line=dict(width=2, color=R.SERIES[i % len(R.SERIES)]), marker=dict(size=8)))
        fig.update_layout(title=m, xaxis_title=x, height=300, showlegend=len(keys) > 1)
        parts.append(R.fig_html(fig))
    body = R.table_html(table) + '<div class="grid2">' + "".join(f"<div>{p}</div>" for p in parts) + "</div>"
    R.write(out, f"Sweep · {sc.name}", f"{len(table)} points over {', '.join(keys)}; invariants checked at every point.", body)
    print(f"{len(table)} points, invariants OK -> {out}")
    return 0


def cmd_compare(args) -> int:
    scs = [load_scenario(p) for p in args.scenarios]
    results = runner.compare(scs)
    out = _out(args, "compare_" + "_vs_".join(s.name for s in scs) + ".html")
    R.write(out, "Compare · " + " vs ".join(s.name for s in scs), "", R.run_report(results, ""))
    print(f"compared {len(scs)} scenarios, invariants OK -> {out}")
    return 0


def cmd_experiments(args) -> int:
    import importlib
    try:
        mod = importlib.import_module(f".experiments.{args.stage}", __package__)
    except ModuleNotFoundError:
        raise SystemExit(f"unknown experiment set {args.stage!r} (try stage1, stage2, ...)")
    s = mod.main()
    t = s["tally"]
    n = args.stage.replace("stage", "")
    print(f"{args.stage}: {t['runs']} runs, {t['ticks']} ticks, invariants OK -> reports/stage_{n}.html")
    return 0


def cmd_twin(args) -> int:
    from .twin import build_standalone
    out = build_standalone(Path(args.out) if args.out else REPORTS / "grainbridge.html")
    print(f"Grainbridge town twin -> {out} (open it in a browser)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="citysim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("scenario"); p.add_argument("--ticks", type=int); p.add_argument("--out")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("sweep"); p.add_argument("scenario")
    p.add_argument("--param", action="append", required=True, help="name=start:stop:step or name=v1,v2,...")
    p.add_argument("--out"); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("compare"); p.add_argument("scenarios", nargs="+"); p.add_argument("--out")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("experiments"); p.add_argument("stage"); p.set_defaults(fn=cmd_experiments)
    p = sub.add_parser("twin", help="build the animated Grainbridge town (reports/grainbridge.html)")
    p.add_argument("--out"); p.set_defaults(fn=cmd_twin)
    args = ap.parse_args(argv)
    return args.fn(args)
