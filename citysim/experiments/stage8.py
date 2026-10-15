"""Stage 8 experiments (trade between cities) and the stage_8.html report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..region import load_region, run_region, with_region
from .stage1 import ROOT, SCEN

SEEDS = (1, 2, 3)
TWO = SCEN / "stage8_two_cities.yaml"
THREE = SCEN / "stage8_three_cities.yaml"
SAME = {"grainbridge.farm_output_per_worker": 5.0, "grainbridge.factory_variable_labor": 0.1,
        "millford.farm_output_per_worker": 5.0, "millford.factory_variable_labor": 0.1}


class RegionTally:
    def __init__(self):
        self.runs = self.ticks = 0
        self.seconds = 0.0
        self.by_experiment: dict[str, int] = {}

    def run(self, sc, exp):
        t = time.perf_counter()
        res = run_region(sc)                 # every city's invariants plus the region's, every tick
        self.seconds += time.perf_counter() - t
        self.runs += 1
        self.ticks += sc.ticks * len(sc.cities)
        self.by_experiment[exp] = self.by_experiment.get(exp, 0) + 1
        return res.wide()


def city_stats(w: pd.DataFrame, c: str, tail: int = 100) -> dict:
    t = w.iloc[-tail:]
    ls = lambda s: t[f"{c}.labor_share/{s}"].mean() if f"{c}.labor_share/{s}" in t else np.nan
    return {f"{c}.bread": t[f"{c}.consumption_per_person"].mean(),
            f"{c}.services": t[f"{c}.services_per_person"].mean(),
            f"{c}.unemployment": t[f"{c}.unemployment_rate"].mean(),
            f"{c}.farm_labor": ls("farm"), f"{c}.bakery_labor": ls("factory")}


def averaged(T, exp, path, seeds=SEEDS, **kw):
    rows, first = [], None
    for seed in seeds:
        w = T.run(with_region(load_region(path), seed=seed, **kw), exp)
        names = [c.split(".")[0] for c in w.columns if c.endswith(".consumption_per_person")]
        r = {}
        for c in names:
            r.update(city_stats(w, c))
        ship = [x for x in w.columns if x.startswith("shipments")]
        r["shipped_per_tick"] = w[ship].iloc[-100:].sum(axis=1).mean() if ship else 0.0
        rows.append(r)
        first = w if first is None else first
    return pd.DataFrame(rows).mean().to_dict(), first


def exp1(T):
    """Gains from comparative advantage."""
    rows = []
    cases = [("autarky", {"trade.enabled": False}), ("trade, transport 2%", {"transport_cost": 0.02}),
             ("trade, transport 5% (base)", {}), ("trade, transport 10%", {"transport_cost": 0.1}),
             ("trade, transport 25%", {"transport_cost": 0.25})]
    for label, kw in cases:
        s, _ = averaged(T, "1", TWO, **kw)
        rows.append(dict(towns="different strengths", case=label, **s))
    for label, kw in (("autarky", {"trade.enabled": False}), ("trade, transport 5%", {})):
        s, _ = averaged(T, "1", TWO, **SAME, **kw)
        rows.append(dict(towns="identical (control)", case=label, **s))
    t = pd.DataFrame(rows)
    for c in ("grainbridge", "millford"):
        t[f"{c}.total"] = t[f"{c}.bread"] + t[f"{c}.services"]
    return dict(table=t)


def exp2(T):
    """Money between towns: floating rates, a currency union, a union with fiscal transfers."""
    rows, series = [], {}
    for label, kw in (("floating currencies", {}), ("currency union", {"currency": "union"}),
                      ("union + fiscal transfers 10%", {"currency": "union", "transfers": 0.1})):
        s, w = averaged(T, "2", TWO, **kw)
        r = dict(regime=label, **s)
        r["grainbridge.money_end"] = w["grainbridge.money_supply"].iloc[-1]
        r["millford.money_end"] = w["millford.money_supply"].iloc[-1]
        r["grainbridge.trade_balance"] = w["trade_balance/grainbridge"].iloc[-100:].mean()
        rows.append(r)
        series[label] = w.reset_index()
    t = pd.DataFrame(rows)
    for c in ("grainbridge", "millford"):
        t[f"{c}.total"] = t[f"{c}.bread"] + t[f"{c}.services"]
    return dict(table=t, series=series)


def exp3(T):
    """A regional shock: Grainbridge's harvest fails, under each money regime."""
    shock = {"grainbridge.harvest_shocks.enabled": True,
             "grainbridge.harvest_shocks.shocks": [dict(start=150, duration=10, multiplier=0.4)]}
    rows, series = [], {}
    for label, kw in (("autarky", {"trade.enabled": False}), ("floating currencies", {}),
                      ("currency union", {"currency": "union"}),
                      ("union + transfers 10%", {"currency": "union", "transfers": 0.1}),
                      ("union + capital flight", {"currency": "union", "capital_flight": 0.2}),
                      ("floating + capital flight", {"capital_flight": 0.2})):
        ws = []
        for seed in SEEDS:
            base = with_region(load_region(TWO), seed=seed, ticks=230, **kw)
            w = T.run(with_region(base, **shock), "3")
            cf = T.run(base, "3")
            win, cw = w.loc[150:230], cf.loc[150:230]
            r = {}
            for c in ("grainbridge", "millford"):
                pop = win[f"{c}.population"]
                r[f"{c}.bread_lost"] = float(((cw[f"{c}.consumption_per_person"] - win[f"{c}.consumption_per_person"]) * pop).sum())
                r[f"{c}.peak_unemployment"] = win[f"{c}.unemployment_rate"].max()
            if "exchange_rate/grainbridge" in win:
                r["grainbridge.currency_low"] = win["exchange_rate/grainbridge"].min() / w["exchange_rate/grainbridge"].loc[149]
            ws.append(r)
            if seed == SEEDS[0]:
                series[label] = w.reset_index()
        rows.append(dict(regime=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def exp4(T):
    """Dependency on a single partner: an embargo cuts Millford off from Grainbridge."""
    embargo = {"currency": "union", "embargoes": [dict(start=150, duration=30, between=["grainbridge", "millford"])]}
    rows, series = [], {}
    for label, path, kw in (("two towns, no embargo", TWO, {"currency": "union"}), ("two towns, embargo", TWO, embargo),
                            ("three towns, no embargo", THREE, {"currency": "union"}), ("three towns, embargo", THREE, embargo)):
        ws = []
        for seed in SEEDS:
            w = T.run(with_region(load_region(path), seed=seed, ticks=230, **kw), "4")
            win = w.loc[150:200]
            ws.append(dict(millford_bread_in_embargo=win["millford.consumption_per_person"].mean(),
                           millford_lowest_bread=win["millford.consumption_per_person"].min(),
                           millford_hungry=win["millford.hungry_share"].mean(),
                           millford_unemployment=win["millford.unemployment_rate"].mean()))
            if seed == SEEDS[0]:
                series[label] = w.reset_index()
        rows.append(dict(case=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def exp5(T):
    """Migration: people follow real incomes."""
    rows = []
    for regime, ckw in (("floating", {}), ("currency union", {"currency": "union"})):
        for mig in (0.0, 0.05):
            ws = []
            for seed in SEEDS:
                w = T.run(with_region(load_region(TWO), seed=seed, migration=mig, **ckw), "5")
                t = w.iloc[-100:]
                ws.append({"grainbridge.population": w["grainbridge.population"].iloc[-1],
                           "millford.population": w["millford.population"].iloc[-1],
                           "grainbridge.real_income": t["real_income/grainbridge"].mean(),
                           "millford.real_income": t["real_income/millford"].mean(),
                           "grainbridge.unemployment": t["grainbridge.unemployment_rate"].mean(),
                           "millford.unemployment": t["millford.unemployment_rate"].mean(),
                           "region_bread": (t["grainbridge.consumption_per_person"] * t["grainbridge.population"]
                                            + t["millford.consumption_per_person"] * t["millford.population"]).mean()
                           / (t["grainbridge.population"] + t["millford.population"]).mean()})
            rows.append(dict(regime=regime, migration=mig, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows))


def build_report(res, T, path):
    summary = {}
    body = ['<nav><a href="#e1">1 · comparative advantage</a><a href="#e2">2 · money regimes</a>'
            '<a href="#e3">3 · regional shock</a><a href="#e4">4 · single partner</a><a href="#e5">5 · migration</a></nav>']
    body.append("<p>Two Stage 5 towns (600 people each; households, firms, services, a bank, a government). "
                "<b>Grainbridge</b> has rich land (9 bushels per farmer) but clumsy bakers (0.35 workers per loaf); "
                "<b>Millford</b> has poor land (3.5) but skilled bakers (0.04). Each town gets a wheat importer and a bread "
                "importer that compete in its local markets; merchants buy from the other town's producers when the "
                "landed cost (after transport losses and conversion) plus a 2% markup beats the local price. Base: "
                f"floating currencies (a dealer moves the rate towards the money it is short of), 5% transport loss. "
                f"Averages of {len(SEEDS)} seeds, last 100 of 300 ticks.</p>")
    sections = [("e1", "1 · Gains from comparative advantage",
                 "Bread and services per person in each town (total = bread + services). Identical towns are a control: "
                 "with nothing to gain, almost nothing is shipped.", "1", "comparative_advantage"),
                ("e2", "2 · Floating currencies, a currency union, a fiscal union",
                 "In a union, payments move bank reserves between the towns; a town that buys more than it sells "
                 "loses money. A fiscal union moves 10% of the gap in money per head from the richer town's households "
                 "to the poorer's every tick.", "2", "money_regimes"),
                ("e3", "3 · A regional shock",
                 "Grainbridge's harvest falls 60% for 10 ticks (150–159). Losses against the same seed without the "
                 "shock, ticks 150–230. Capital flight: households in the town with higher unemployment send 20% × the "
                 "unemployment gap of their money to the other town per tick.", "3", "regional_shock"),
                ("e4", "4 · Dependency on a single partner",
                 "In a currency union Millford specialises in baking and buys most of its wheat. At tick 150 an embargo "
                 "cuts trade between Grainbridge and Millford for 30 ticks. With a third town (Cornhill, also good "
                 "farmland) Millford has another supplier.",
                 "4", "single_partner"),
                ("e5", "5 · Migration",
                 "5% of a town's people per unit of real-income gap (beyond a 10% moving cost) move per tick, the "
                 "unemployed first, with their money.", "5", "migration")]
    for anchor, title, text, key, sname in sections:
        t = res[key]["table"]
        summary[sname] = t.to_dict(orient="records")
        body.append(f'<h2 id="{anchor}">{title}</h2><p>{text}</p>')
        body.append(R.table_html(t))
        if key == "2":
            f = go.Figure()
            i = 0
            for k, w in res[key]["series"].items():
                for c, dash in (("grainbridge", "solid"), ("millford", "dot")):
                    f.add_trace(go.Scatter(x=w.tick, y=w[f"{c}.money_supply"], name=f"{c}, {k}", mode="lines",
                                           line=dict(width=2, dash=dash, color=R.SERIES[i])))
                i += 1
            f.update_layout(title="Money in each town (one seed)", xaxis_title="tick", yaxis_title="money supply")
            body.append(R.fig_html(f))
        if key == "3":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w["grainbridge.consumption_per_person"], name=k, mode="lines",
                                       line=dict(width=2, color=R.SERIES[i % len(R.SERIES)])))
            f.update_xaxes(range=[140, 230])
            f.update_layout(title="Bread per person in Grainbridge around its harvest failure (one seed)",
                            xaxis_title="tick", yaxis_title="loaves per person")
            body.append(R.fig_html(f))
        if key == "4":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w["millford.consumption_per_person"], name=k, mode="lines",
                                       line=dict(width=2, color=R.SERIES[i])))
            f.update_xaxes(range=[140, 230])
            f.update_layout(title="Bread per person in Millford (one seed)", xaxis_title="tick", yaxis_title="loaves per person")
            body.append(R.fig_html(f))
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} region runs, "
                f"{T.ticks:,} city-ticks, {T.seconds:.1f} s. New in Stage 8: every shipment leaving one town (flow "
                "'export') arrives in another (flow 'import'), minus transport losses booked as waste; under floating "
                "rates no money crosses a border; under a union, every unit of money leaving one town's bank arrives at "
                "another's; goods and money conservation in each town count imports and exports.</p>")
    R.write(path, "Stage 8 · Trade between cities", "citysim — comparative advantage, currencies, shocks and "
            "migration. Generated by <code>python -m citysim experiments stage8</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_8.html") -> dict:
    T = RegionTally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T)}
    s = build_report(res, T, out)
    out.with_name("stage_8_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
