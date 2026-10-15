"""HTML report helpers (plotly). One self-contained file per report."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Reference categorical palette (fixed order, never cycled) and sequential blue ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED = "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

pio.templates["citysim"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK2),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", showline=True),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks=""),
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, title=None),
    hovermode="x unified", margin=dict(l=60, r=24, t=48, b=40), height=420, autosize=True,
))
pio.templates.default = "citysim"


def line(df: pd.DataFrame, x: str, ys: dict[str, str], title: str, xlab: str = "", ylab: str = "",
         log_y: bool = False, colors: Iterable[str] | None = None, dash: dict[str, str] | None = None) -> go.Figure:
    """ys maps legend label -> column."""
    fig = go.Figure()
    cols = list(colors) if colors else SERIES
    for i, (label, col) in enumerate(ys.items()):
        fig.add_trace(go.Scatter(x=df[x], y=df[col], name=label, mode="lines",
                                 line=dict(width=2, color=cols[i % len(cols)],
                                           dash=(dash or {}).get(label, "solid"))))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def vline(fig: go.Figure, x: float, text: str) -> go.Figure:
    fig.add_vline(x=x, line=dict(color=MUTED, width=1), annotation_text=text,
                  annotation_position="top right", annotation_font=dict(color=INK2, size=12))
    return fig


def fig_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%",
                       config={"displaylogo": False, "responsive": True})


def table_html(df: pd.DataFrame, floatfmt: str = "{:,.4g}") -> str:
    def fmt(v):
        if isinstance(v, float):
            if v != v:
                return "–"
            return floatfmt.format(0.0 if abs(v) < 1e-9 else v)
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


CSS = f"""
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --grid:{GRID}; --accent:{SERIES[0]}; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font: 15px/1.55 Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 80px; }}
h1 {{ font-size: 28px; margin: 0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size: 21px; margin: 48px 0 8px; padding-top: 16px; border-top: 1px solid var(--grid); }}
h3 {{ font-size: 16px; margin: 24px 0 6px; }}
p, li {{ color: var(--ink2); max-width: 78ch; }}
.sub {{ color: var(--ink2); margin: 0 0 24px; }}
.grid2 > div {{ min-width: 0; }}
.grid2 {{ display:grid; grid-template-columns: minmax(0, 1fr); gap: 8px; }}
.tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0; }}
.tile {{ border:1px solid var(--grid); border-radius: 8px; padding: 12px 14px; }}
.tile .v {{ font-size: 24px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }}
.tile .k {{ font-size: 13px; color: var(--ink2); }}
.tbl {{ overflow-x:auto; margin: 12px 0; }}
table {{ border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 5px 12px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--ink); font-weight: 600; }}
.pass {{ color:#006300; font-weight:600; }} .fail {{ color:#b42318; font-weight:600; }}
nav a {{ color: var(--accent); margin-right: 14px; text-decoration: none; }}
code {{ font-size: 13px; background: #f0efec; padding: 1px 4px; border-radius: 4px; }}
"""


def tiles(items: list[tuple[str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{html.escape(v)}</div><div class="k">{html.escape(k)}</div></div>'
        for k, v in items) + "</div>"


def page(title: str, subtitle: str, body: str) -> str:
    from plotly.offline import get_plotlyjs
    plotly_js = f'<script type="text/javascript">{get_plotlyjs()}</script>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style>{plotly_js}</head>
<body><main><h1>{html.escape(title)}</h1><p class="sub">{subtitle}</p>{body}</main></body></html>"""


def write(path: str | Path, title: str, subtitle: str, body: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(title, subtitle, body), encoding="utf-8")
    return path


def run_report(results, title: str) -> str:
    """Generic report for `run` / `compare`: every scalar metric over time, one chart each."""
    frames = []
    for res in results:
        w = res.wide().reset_index()
        w["scenario"] = res.scenario.name
        frames.append(w)
    metrics = [c for c in frames[0].columns if c not in ("tick", "scenario")]
    parts = []
    for m in metrics:
        fig = go.Figure()
        for i, w in enumerate(frames):
            if m in w:
                fig.add_trace(go.Scatter(x=w["tick"], y=w[m], name=w["scenario"].iloc[0], mode="lines",
                                         line=dict(width=2, color=SERIES[i % len(SERIES)])))
        fig.update_layout(title=m, height=300, showlegend=len(frames) > 1, xaxis_title="tick")
        parts.append(fig_html(fig))
    return '<div class="grid2">' + "".join(f"<div>{p}</div>" for p in parts) + "</div>"
