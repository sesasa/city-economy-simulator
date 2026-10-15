"""Stage 5 experiments (government and public goods) and the stage_5.html report."""
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

SHOCK = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=150, duration=5, multiplier=0.5)]}


def base():
    return load_scenario(SCEN / "stage5_government.yaml")


def stats(w: pd.DataFrame, tail: int = 100) -> dict:
    t = w.iloc[-tail:]
    out = dict(bread=t.consumption_per_person.mean(), services=t.services_per_person.mean(),
               unemployment=t.unemployment_rate.mean(), hungry=t.hungry_share.mean(),
               gini_income=t.gini_income.mean(), gini_wealth=t.gini_wealth.iloc[-1], population_end=w.population.iloc[-1])
    for c, k in (("tax_revenue", "taxes"), ("public_debt", "debt_end"), ("deficit", "deficit"), ("roads", "roads"),
                 ("loans", "private_loans"), ("unmaintained_share", "unmaintained")):
        if c in w:
            out[k] = t[c].mean() if k not in ("debt_end",) else w[c].iloc[-1]
    return out


def averaged(T, exp, sc, **kw):
    rows, first = [], None
    for seed in SEEDS:
        w = T.run(with_params(sc, seed=seed, **kw), exp).wide()
        rows.append(stats(w))
        first = w if first is None else first
    return pd.DataFrame(rows).mean().to_dict(), first.reset_index()


def exp1(T):
    """Does a safety net stop Stage 4's starvation spiral?"""
    cases = {"no government": {"government.enabled": False},
             "taxes only (no benefits)": {"unemployment_benefit": 0.0},
             "benefits 40% of wage (base)": {},
             "benefits 40% + basic income 20%": {"basic_income": 0.2},
             "benefits 80% + income tax 20%": {"unemployment_benefit": 0.8, "income_tax": 0.2}}
    rows, series = [], {}
    for label, kw in cases.items():
        s, w = averaged(T, "1", base(), ticks=300, **{"demography.enabled": True}, **kw)
        rows.append(dict(policy=label, **s))
        series[label] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp2(T):
    """Tax-funded versus private maintenance (calm times and with a harvest shock)."""
    rows = []
    for shock in (False, True):
        for public in (False, True):
            kw = dict(ticks=300, public_maintenance=public, **(SHOCK if shock else {}))
            s, _ = averaged(T, "2", base(), **kw)
            rows.append(dict(maintenance="public (tax-funded)" if public else "private (each owner pays)",
                             harvest_shock=shock, **s))
    return dict(table=pd.DataFrame(rows))


def exp3(T):
    """Benefits as automatic stabilizers in a harvest shock."""
    rows, series = [], {}
    for label, kw in (("no benefits", {"unemployment_benefit": 0.0}), ("benefits 40%", {}), ("benefits 80%", {"unemployment_benefit": 0.8})):
        ws = []
        for seed in SEEDS:
            w = T.run(with_params(base(), seed=seed, ticks=260, **SHOCK, **kw), "3").wide()
            cf = T.run(with_params(base(), seed=seed, ticks=260, **kw), "3").wide()
            win = w.loc[150:200]
            ws.append(dict(min_bread=win.consumption_per_person.min(),
                           bread_lost=float(((cf.consumption_per_person - w.consumption_per_person).clip(lower=0)
                                             * w.population).loc[150:200].sum()),
                           peak_unemployment=win.unemployment_rate.max(),
                           extra_benefits=float((w["public_spending/benefits"] - cf["public_spending/benefits"]).loc[150:200].sum()),
                           deficit_peak=win.deficit.max()))
            if seed == SEEDS[0]:
                series[label] = w.reset_index()
        rows.append(dict(policy=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def exp4(T):
    """Roads: a public good no single firm builds."""
    rows = []
    for target in (0, 5, 20, 60):
        s, w = averaged(T, "4", base(), ticks=300, road_target=target)
        dist = w["labor_share/distribution"].iloc[-100:].mean()
        rows.append(dict(road_target=target, distribution_labor_share=dist, **s))
    return dict(table=pd.DataFrame(rows))


def exp5(T):
    """Persistent deficits and crowding out: generous basic income, bonds sold to the bank."""
    rows, series = [], {}
    for label, bi, tax in (("modest deficits", 0.3, 0.1), ("large deficits", 0.6, 0.03)):
        for weight in (0.0, 1.0):
            for cap in (0.03, 0.1):
                kw = dict(ticks=300, basic_income=bi, income_tax=tax, profit_tax=2 * tax, bond_risk_weight=weight,
                          initial_capital=cap, investment_share=0.05)
                s, w = averaged(T, "5", base(), **kw)
                pl = w.price_level
                rows.append(dict(budget=label, bond_risk_weight=weight, bank_capital=cap,
                                 inflation_per_tick=(pl.iloc[-1] / pl.iloc[50]) ** (1 / (len(pl) - 50)) - 1, **s))
                if cap == 0.03:
                    series[f"{label}, bond weight {weight}"] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp6(T):
    """Misallocated public spending: a share of road workers produce nothing."""
    rows = []
    for waste in (0.0, 0.3, 0.6, 0.9):
        s, _ = averaged(T, "6", base(), ticks=300, road_target=20, waste=waste)
        rows.append(dict(waste=waste, **s))
    return dict(table=pd.DataFrame(rows))


def build_report(res, T, path):
    summary = {}
    body = ['<nav><a href="#e1">1 · safety net</a><a href="#e2">2 · maintenance</a><a href="#e3">3 · stabilizers</a>'
            '<a href="#e4">4 · roads</a><a href="#e5">5 · deficits</a><a href="#e6">6 · waste</a></nav>']
    body.append("<p>A government joins the Stage 4 economy (individual households, many firms, owners, services, a bank). "
                "Base policy: 10% income tax, 20% profit tax, unemployment benefits at 40% of the average wage, five units "
                "of roads (each cuts the labor needed to deliver bread). It borrows by selling bonds to the bank when "
                "spending exceeds revenue, and money beyond two ticks of spending repays debt and is then rebated per "
                f"head: a government that hoards money drains the economy. Averages of {len(SEEDS)} seeds.</p>")
    sections = [("e1", "1 · Does a safety net stop the starvation spiral?",
                 "Stage 4 found that with population dynamics the poor starve and the town dies. Same economy, 300 ticks, "
                 "families grow when fed and shrink when hungry.", "1", "safety_net"),
                ("e2", "2 · Tax-funded versus private maintenance",
                 "Private: each capital owner buys its own repairs out of its own cash. Public: the government pays the repair "
                 "shops for everyone's upkeep out of taxes.", "2", "maintenance"),
                ("e3", "3 · Benefits as automatic stabilizers",
                 "Harvest falls 50% for 5 ticks (150-154). Bread lost and benefits are measured against the same seed "
                 "without the shock.", "3", "stabilizers"),
                ("e4", "4 · Roads: a public good",
                 "No single distributor gains enough from a road to build it; the government builds roads with idle hands "
                 "and keeps them up.", "4", "roads"),
                ("e5", "5 · Persistent deficits and crowding out",
                 "A basic income of 30% (taxes 10%) or 60% of the wage (taxes 3%), financed by selling bonds to the bank. "
                 "Bonds either need no bank capital (weight 0) or as much as a loan (weight 1); the bank starts with thin "
                 "(3%) or normal (10%) capital.", "5", "deficits"),
                ("e6", "6 · Misallocated public spending",
                 "A share of the people the government hires for roads produce nothing.", "6", "waste")]
    for anchor, title, text, key, sname in sections:
        t = res[key]["table"]
        summary[sname] = t.to_dict(orient="records")
        body.append(f'<h2 id="{anchor}">{title}</h2><p>{text}</p>')
        body.append(R.table_html(t))
        if key in ("1", "3", "5") and res[key].get("series"):
            f = go.Figure()
            col = {"1": "population", "3": "consumption_per_person", "5": "public_debt"}[key]
            for i, (k, w) in enumerate(res[key]["series"].items()):
                if i >= 8:
                    break
                f.add_trace(go.Scatter(x=w.tick, y=w[col], name=str(k), mode="lines", line=dict(width=2, color=R.SERIES[i])))
            if key == "3":
                f.update_xaxes(range=[140, 220])
            f.update_layout(title={"1": "Population", "3": "Bread per person around the shock", "5": "Public debt"}[key],
                            xaxis_title="tick", yaxis_title=col.replace("_", " "))
            body.append(R.fig_html(f))
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s. New in Stage 5: the government never spends money it does not have and never holds "
                "its own bonds; roads built never exceed what the (useful) public labor could build; bond sales, interest and "
                "redemptions are part of the bank's money-explained check.</p>")
    R.write(path, "Stage 5 · Government and public goods", "citysim — taxes, a safety net, roads and public debt. "
            "Generated by <code>python -m citysim experiments stage5</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_5.html") -> dict:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T), "6": exp6(T)}
    s = build_report(res, T, out)
    out.with_name("stage_5_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
