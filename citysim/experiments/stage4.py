"""Stage 4 experiments (many firms, competition and specialization) and the stage_4.html report.

Households are individual agents now and several rules draw on the seeded RNG, so
each setting is run with SEEDS seeds and averaged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..config import load_scenario, with_params
from ..core import runner
from .stage1 import ROOT, SCEN, Tally

SEEDS = (1, 2, 3)


def base():
    return load_scenario(SCEN / "stage4_competition.yaml")


def tail_stats(w: pd.DataFrame, tail: int = 100) -> dict:
    t = w.iloc[-tail:]
    out = dict(bread=t.consumption_per_person.mean(), services=t.services_per_person.mean(),
               unemployment=t.unemployment_rate.mean(), hungry=t.hungry_share.mean(),
               gini_income=t.gini_income.mean(), gini_wealth=t.gini_wealth.iloc[-1],
               top10_wealth=t.top10_wealth_share.iloc[-1], real_wage=t.real_wage.mean(),
               profit_share=t.profit_share.mean())
    for col in ("markups/distribution", "markups/farm", "markups/services", "concentration/services", "firms/services"):
        if col in w:
            out[col.replace("/", "_")] = t[col].mean() if not col.startswith("firms") else t[col].iloc[-1]
    return out


def averaged(T: Tally, exp: str, sc, **kw) -> tuple[dict, pd.DataFrame]:
    rows, first = [], None
    for seed in SEEDS:
        w = T.run(with_params(sc, seed=seed, **kw), exp).wide()
        rows.append(tail_stats(w))
        first = w if first is None else first
    return pd.DataFrame(rows).mean().to_dict(), first.reset_index()


def exp1(T: Tally) -> dict:
    """Monopoly vs competition in the bread chain (no entry)."""
    rows = []
    for n in (1, 2, 4, 8):
        for power in (True, False):
            kw = dict(ticks=400, farm_firms=n, distribution_firms=n, maintenance_firms=max(1, n // 2),
                      market_power=power, **{"entry_exit.enabled": False})
            s, _ = averaged(T, "1", base(), **kw)
            rows.append(dict(firms_per_sector=n, market_power=power, **s))
    return dict(table=pd.DataFrame(rows))


def exp2(T: Tally) -> dict:
    """How many services firms survive under different fixed costs (entry and exit on)?"""
    rows, series = [], {}
    for fc in (0.5, 2.0, 5.0, 10.0, 20.0):
        s, w = averaged(T, "2", base(), ticks=500, **{"services.fixed_crew": fc, "services.firms": 3})
        rows.append(dict(services_fixed_crew=fc, **s))
        series[fc] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp3(T: Tally) -> dict:
    """When does a services sector emerge? Start with none; entry on."""
    rows, series = [], {}
    for N0 in (150, 300, 600, 1200):
        firsts, ends = [], []
        for seed in SEEDS:
            k = max(1, N0 // 300)
            w = T.run(with_params(base(), seed=seed, ticks=400, N0=N0, initial_factories=max(1, N0 // 500 + 1),
                                  farm_firms=k, factory_firms=max(1, k // 2), distribution_firms=k, maintenance_firms=max(1, k // 2),
                                  **{"services.firms": 0}), "3").wide()
            appear = w.index[w["firms/services"] > 0]
            firsts.append(int(appear[0]) if len(appear) else np.nan)
            ends.append(tail_stats(w))
            if seed == SEEDS[0]:
                series[N0] = w.reset_index()
        e = pd.DataFrame(ends).mean().to_dict()
        rows.append(dict(N0=N0, first_services_firm_tick=np.nanmean(firsts) if not np.isnan(firsts).all() else None, **e))
    no_serv, wn = averaged(T, "3", base(), ticks=400, **{"services.enabled": False})
    with_serv, ws = averaged(T, "3", base(), ticks=400)
    compare = pd.DataFrame([dict(economy="bread only (Stage 2-3 goods)", price_level_end=wn.price_level.iloc[-1], **no_serv),
                            dict(economy="bread + services", price_level_end=ws.price_level.iloc[-1], **with_serv)])
    return dict(table=pd.DataFrame(rows), series=series, compare=compare)


def exp4(T: Tally) -> dict:
    """Inequality: how ownership is spread."""
    rows = []
    for label, kw in (("1 owner per firm", dict(owners_per_firm=1)), ("5 owners", dict(owners_per_firm=5)),
                      ("20 owners (base)", dict(owners_per_firm=20)), ("60 owners", dict(owners_per_firm=60)),
                      ("public (everyone)", dict(ownership="public"))):
        s, _ = averaged(T, "4", base(), ticks=400, **kw)
        rows.append(dict(ownership=label, **s))
    return dict(table=pd.DataFrame(rows))


def exp5(T: Tally) -> dict:
    """Structural unemployment: how long switching occupation takes."""
    rows = []
    for rt in (0, 3, 10, 30):
        for mob in (0.05, 0.2):
            s, _ = averaged(T, "5", base(), ticks=400, retrain_ticks=rt, mobility_unemployed=mob)
            rows.append(dict(retrain_ticks=rt, mobility_unemployed=mob, **s))
    return dict(table=pd.DataFrame(rows))


def scaling() -> dict:
    sc = with_params(base(), N0=100000, initial_factories=110, farm_firms=20, factory_firms=10, distribution_firms=20,
                     maintenance_firms=10, ticks=100, **{"services.firms": 30})
    sim = runner.build(sc)
    t = time.perf_counter()
    for _ in range(100):
        sim.step()
    dt = time.perf_counter() - t
    return dict(agents=100000, ticks=100, seconds=dt, projected_minutes_for_1000_ticks=dt * 10 / 60)


def build_report(res: dict, T: Tally, path: Path) -> dict:
    e1, e2, e3, e4, e5, sc = (res[k] for k in ("1", "2", "3", "4", "5", "scaling"))
    summary: dict = {}
    body = ['<nav><a href="#e1">1 · monopoly vs competition</a><a href="#e2">2 · fixed costs</a><a href="#e3">3 · new sectors</a>'
            '<a href="#e4">4 · inequality</a><a href="#e5">5 · structural unemployment</a><a href="#scale">scale</a></nav>']
    body.append("<p>This stage keeps Stage 3's bank switched on: without credit, any disruption (a factory closing, a "
                "wave of job switching) let firms run out of cash for inputs and the economy froze (Stage 2's "
                "liquidity trap).</p>")
    body.append("<p>Households are individuals now (600 people, one agent each): an occupation, an employer or none, "
                "savings, and three ticks of retraining after switching occupation. Firms have owners (20 households "
                "share each firm's dividends in the base case) and their normal markup rises with their sector's "
                "concentration (Herfindahl index): 10% when fully competitive, 60% for a monopoly. A services sector "
                "sells labor-only services (0.5 workers per unit, 2 to open) that people buy once their bread need is met. "
                "Firms enter where margins exceed 20% and leave after 12 loss-making ticks with an empty till. "
                f"Every number is the average of {len(SEEDS)} seeds over the last 100 ticks.</p>")

    t1 = e1["table"]
    summary["competition"] = t1.to_dict(orient="records")
    body.append('<h2 id="e1">1 · Monopoly versus competition</h2>')
    body.append("<p>Same town, same technology; the farm, distribution and repair sectors have 1 to 8 firms (no entry). "
                "With market power on, fewer firms mean higher normal markups.</p>")
    f = go.Figure()
    for i, power in enumerate((True, False)):
        d = t1[t1.market_power == power]
        f.add_trace(go.Scatter(x=d.firms_per_sector, y=d.markups_distribution, name=f"retail markup, market power {'on' if power else 'off'}",
                               mode="lines+markers", line=dict(width=2, color=R.SERIES[i]), marker=dict(size=8)))
        f.add_trace(go.Scatter(x=d.firms_per_sector, y=d.real_wage, name=f"real wage, market power {'on' if power else 'off'}",
                               mode="lines+markers", line=dict(width=2, color=R.SERIES[i + 2]), marker=dict(size=8)))
    f.update_xaxes(type="log", title="firms per sector")
    f.update_layout(title="More firms, lower markups, higher real wages", yaxis_title="markup / loaves per wage")
    body.append(R.fig_html(f))
    body.append(R.table_html(t1))

    t2 = e2["table"]
    summary["fixed_costs"] = t2.to_dict(orient="records")
    body.append('<h2 id="e2">2 · How many firms survive under different fixed costs?</h2>')
    body.append("<p>Services firms need a fixed crew to open. Entry and exit decide how many survive.</p>")
    body.append(R.table_html(t2))
    f = go.Figure()
    for i, (fc, w) in enumerate(e2["series"].items()):
        f.add_trace(go.Scatter(x=w.tick, y=w["firms/services"], name=f"fixed crew {fc}", mode="lines", line=dict(width=2, color=R.SERIES[i])))
    f.update_layout(title="Number of services firms over time (seed 1)", xaxis_title="tick", yaxis_title="firms")
    body.append(R.fig_html(f))

    t3 = e3["table"]
    summary["emergence"] = t3.to_dict(orient="records")
    summary["services_vs_none"] = e3["compare"].to_dict(orient="records")
    body.append('<h2 id="e3">3 · When do new sectors emerge?</h2>')
    body.append("<p>Start with no services firms at all. A founder opens one when the sector looks profitable (an empty "
                "sector with unspent household money counts as attractive). Bigger towns have more spare hands and "
                "more spare money.</p>")
    body.append(R.table_html(t3))
    body.append("<h3>Services cure the satiation trap of Stages 2–3</h3><p>With bread as the only good, people who have "
                "eaten their fill stop spending, idle hands pile up and prices deflate. With services, that money and "
                "labor have somewhere to go.</p>")
    body.append(R.table_html(e3["compare"]))

    t4 = e4["table"]
    summary["inequality"] = t4.to_dict(orient="records")
    body.append('<h2 id="e4">4 · Inequality: who owns the firms?</h2>')
    body.append("<p>Profits go to owners. The more concentrated ownership is, the more unequal incomes and wealth become, "
                "and the more workers go short of bread while owners buy services.</p>")
    f = go.Figure()
    f.add_trace(go.Bar(x=t4.ownership, y=t4.gini_wealth, name="wealth Gini", marker_color=R.SERIES[0]))
    f.add_trace(go.Bar(x=t4.ownership, y=t4.gini_income, name="income Gini", marker_color=R.SERIES[1]))
    f.add_trace(go.Bar(x=t4.ownership, y=t4.hungry, name="share of people short of bread", marker_color=R.SERIES[2]))
    f.update_layout(title="Ownership and inequality", barmode="group", yaxis_title="index / share", hovermode="closest")
    body.append(R.fig_html(f))
    body.append(R.table_html(t4))

    t5 = e5["table"]
    summary["structural_unemployment"] = t5.to_dict(orient="records")
    body.append('<h2 id="e5">5 · Structural unemployment</h2>')
    body.append("<p>People who switch occupation spend <code>retrain_ticks</code> unable to work; unemployed people "
                "reconsider their occupation with probability <code>mobility_unemployed</code> per tick.</p>")
    body.append(R.table_html(t5))

    summary["scaling"] = sc
    body.append('<h2 id="scale">Scale</h2>')
    body.append(f"<p>100,000 household agents, 90 firms: {sc['seconds']:.1f} s for 100 ticks with every invariant checked, "
                f"so about {sc['projected_minutes_for_1000_ticks']:.1f} minutes for 1,000 ticks (target: a few minutes).</p>")

    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s. New in Stage 4: nobody works for a firm that does not exist, no firm has more "
                "employees than it hired, occupations' pools cover hires and builders, and services sold never exceed "
                "what the staff could serve.</p>")
    R.write(path, "Stage 4 · Many firms, competition and specialization", "citysim — individual households, owners, "
            "entry and exit, and a services sector. Generated by <code>python -m citysim experiments stage4</code>.",
            "\n".join(body))
    return summary


def run_all():
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T), "scaling": scaling()}
    return res, T


def main(out: Path = ROOT / "reports" / "stage_4.html") -> dict:
    res, T = run_all()
    s = build_report(res, T, out)
    out.with_name("stage_4_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
