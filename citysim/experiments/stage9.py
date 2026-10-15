"""Stage 9 experiments (technology and long-run growth) and the stage_9.html report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..config import load_scenario, with_params
from .stage1 import ROOT, SCEN, Tally
from .stage4 import SEEDS

PLATEAU = SCEN / "stage9_plateau.yaml"
MARKET = SCEN / "stage9_market.yaml"


def plateau_tick(pop: pd.Series, threshold: float = 0.001, window: int = 20) -> float:
    g = pop.pct_change().rolling(window).mean()
    late = g[(g < threshold) & (pop.index > 200)]
    return float(late.index[0]) if len(late) else np.nan


def planner_row(w: pd.DataFrame) -> dict:
    pop = w.population
    end = pop.iloc[-1]
    return dict(population_end=end, population_t1000=pop.loc[1000] if 1000 in pop.index else np.nan,
                plateau_tick=plateau_tick(pop), growth_last_200=float((pop.iloc[-1] / pop.iloc[-201]) ** (1 / 200) - 1),
                bread_per_person_last_200=w.consumption_per_person.iloc[-200:].mean(),
                tech_level_end=w.tech_level.iloc[-1] if "tech_level" in w else 1.0,
                frontier_end=w.frontier.iloc[-1] if "frontier" in w else 1.0,
                research_share_of_people=float((w.researchers / pop).iloc[-200:].mean()) if "researchers" in w else 0.0)


def exp1(T):
    """Break the Stage 1 plateau."""
    rows, series = [], {}
    cases = {"no technology (Stage 1)": {"technology.enabled": False},
             "learning by doing only": {"research_share": 0.0},
             "research 2% only": {"lbd_exponent": 0.0},
             "research 2% + learning by doing": {}}
    for label, kw in cases.items():
        w = T.run(with_params(load_scenario(PLATEAU), ticks=1500, **kw), "1").wide()
        rows.append(dict(case=label, **planner_row(w)))
        series[label] = w.reset_index()
    return dict(table=pd.DataFrame(rows), series=series)


def exp2(T):
    """Research share against long-run growth (and ideas getting harder to find)."""
    rows = []
    for difficulty in (0.0, 1.0):
        for share in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35):
            w = T.run(with_params(load_scenario(PLATEAU), ticks=1500, research_share=share, difficulty=difficulty,
                                  lbd_exponent=0.0), "2").wide()
            rows.append(dict(difficulty=difficulty, research_share=share, **planner_row(w)))
    return dict(table=pd.DataFrame(rows))


def market_row(w: pd.DataFrame, tail: int = 150) -> dict:
    t = w.iloc[-tail:]
    out = dict(bread=t.consumption_per_person.mean(), services=t.services_per_person.mean(),
               unemployment=t.unemployment_rate.mean(), real_wage=t.real_wage.mean(),
               tech_level=w.tech_level.iloc[-1] if "tech_level" in w else 1.0,
               tech_spread=t.tech_spread.mean() if "tech_spread" in t else 1.0,
               farm_concentration=t["concentration/farm"].mean() if "concentration/farm" in t else np.nan,
               bakery_concentration=t["concentration/factory"].mean() if "concentration/factory" in t else np.nan,
               profit_share=t.profit_share.mean() if "profit_share" in t else np.nan,
               gini_income=t.gini_income.mean())
    out["services_price_vs_bread"] = t.services_price_ratio.mean() if "services_price_ratio" in t else np.nan
    return out


def averaged(T, exp, **kw):
    rows, first = [], None
    for seed in SEEDS:
        w = T.run(with_params(load_scenario(MARKET), seed=seed, **kw), exp).wide()
        rows.append(market_row(w))
        first = w.reset_index() if first is None else first
    return pd.DataFrame(rows).mean().to_dict(), first


def exp3(T):
    """Slow spread of technology between firms."""
    rows, series = [], {}
    for d in (0.0, 0.005, 0.02, 0.1):
        s, w = averaged(T, "3", diffusion=d)
        rows.append(dict(diffusion_per_tick=d, **s))
        series[f"diffusion {d}"] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp4(T):
    """Research in the market economy, with and without learning by doing."""
    rows, series = [], {}
    for lbd in (0.0, 0.05):
        for share in (0.0, 0.01, 0.03, 0.06, 0.12):
            kw = dict(research_share=share, lbd_exponent=lbd)
            if share == 0.0 and lbd == 0.0:
                kw = {"technology.enabled": False}
            s, w = averaged(T, "4", **kw)
            rows.append(dict(learning_by_doing=lbd > 0, research_share=share, **s))
            if lbd > 0:
                series[f"research {share:.0%}"] = w
    return dict(table=pd.DataFrame(rows), series=series)


def build_report(res, T, path):
    summary = {}
    body = ['<nav><a href="#e1">1 · the plateau</a><a href="#e2">2 · research share</a><a href="#e3">3 · diffusion</a>'
            '<a href="#e4">4 · research in a market</a></nav>']
    body.append("<p>Technology multiplies farm output per worker and divides the labor needed per loaf in baking and "
                "delivery (fixed crews, maintenance and services do not improve). <b>Research</b>: a share of the "
                "population works on ideas, pushing the frontier out by 5% × research share per tick. <b>Diffusion</b>: "
                "users adopt a share of the gap to the frontier each tick. <b>Learning by doing</b>: productivity × "
                "(1 + cumulative output / 20,000)^0.05. Experiments 1–2 use the Stage 1 planner economy with "
                "population growth, land limits and investment (the economy that reached a plateau in Stage 1); "
                "experiments 3–4 use the Stage 5 market economy with 600 people, where each firm has its own "
                f"technology and research is a public institute hiring in the labor market ({len(SEEDS)} seeds, 600 ticks).</p>")
    sections = [("e1", "1 · Breaking the Stage 1 plateau", "1,500 ticks. The plateau tick is when population growth "
                 "falls below 0.1% per tick.", "1", "plateau"),
                ("e2", "2 · Research share against long-run growth",
                 "Research staff do not produce food. Difficulty 1: each idea is as hard to find as the frontier is high "
                 "(growth / A*), so constant research gives slowing growth.", "2", "research_share"),
                ("e3", "3 · Slow spread of technology between firms",
                 "Research 3% of people. Each firm's adoption speed is drawn around the average (lognormal, sd 0.5); "
                 "entrants start 10% below their sector's average.", "3", "diffusion"),
                ("e4", "4 · Research in the market economy",
                 "The research institute's wages are paid by the government (borrowing if needed).", "4", "market_research")]
    for anchor, title, text, key, sname in sections:
        t = res[key]["table"]
        summary[sname] = t.to_dict(orient="records")
        body.append(f'<h2 id="{anchor}">{title}</h2><p>{text}</p>')
        body.append(R.table_html(t))
        if key == "1":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w.population, name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
            f.update_yaxes(type="log")
            f.update_layout(title="Population (log scale)", xaxis_title="tick", yaxis_title="people")
            body.append(R.fig_html(f))
        if key == "3":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w.tech_spread, name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
            f.update_layout(title="Best / worst firm technology (one seed)", xaxis_title="tick", yaxis_title="ratio")
            body.append(R.fig_html(f))
        if key == "4":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w.real_wage, name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
            f.update_layout(title="Real wage (loaves per tick of work, one seed, with learning by doing)",
                            xaxis_title="tick", yaxis_title="loaves")
            body.append(R.fig_html(f))
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s. New in Stage 9: the frontier never falls and no firm uses more than the frontier; "
                "the planner's physics check uses the improved technology, and researchers are fed but produce "
                "nothing; research wages are explicit transfers from the government.</p>")
    R.write(path, "Stage 9 · Technology and long-run growth", "citysim — research, diffusion and learning by doing. "
            "Generated by <code>python -m citysim experiments stage9</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_9.html") -> dict:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T)}
    s = build_report(res, T, out)
    out.with_name("stage_9_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
