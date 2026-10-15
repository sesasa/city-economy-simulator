"""Stage 6 experiments (central bank and monetary policy) and the stage_6.html report."""
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

MORE_SEEDS = (1, 2, 3, 4, 5)
DEMAND = {"demand_shocks.enabled": True, "demand_shocks.shocks": [dict(start=150, duration=20, multiplier=0.6)]}
SUPPLY = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=150, duration=5, multiplier=0.5)]}
SLUMP = {"demand_shocks.enabled": True, "demand_shocks.shocks": [dict(start=150, duration=30, multiplier=0.5)]}
RULES = {"Taylor rule": {},
         "fixed money stock": {"central_bank.rule": "money_growth", "central_bank.money_growth": 0.0},
         "fixed interest rate": {"central_bank.rule": "fixed_rate"}}


def base():
    return load_scenario(SCEN / "stage6_central_bank.yaml")


def shock_cost(T, exp, shock, kw, ticks=260, window=(150, 220), seeds=SEEDS):
    """Losses against the same seed without the shock."""
    rows, first = [], None
    a, b = window
    for seed in seeds:
        w = T.run(with_params(base(), seed=seed, ticks=ticks, **shock, **kw), exp).wide()
        cf = T.run(with_params(base(), seed=seed, ticks=ticks, **kw), exp).wide()
        win, cw = w.loc[a:b], cf.loc[a:b]
        pop = w.population.loc[a:b]
        rows.append(dict(bread_lost=float(((cw.consumption_per_person - win.consumption_per_person) * pop).sum()),
                         services_lost=float(((cw.services_per_person - win.services_per_person) * pop).sum()),
                         peak_unemployment=win.unemployment_rate.max(),
                         extra_unemployment=float((win.unemployment_rate - cw.unemployment_rate).sum()),
                         price_swing=float((win.cpi / cw.cpi - 1).abs().mean()),
                         lowest_rate=win.policy_rate.min(), highest_rate=win.policy_rate.max()))
        first = w.reset_index() if first is None else first
    return pd.DataFrame(rows).mean().to_dict(), first


def episodes(flag: pd.Series) -> int:
    """Number of separate spells in a 0/1 series."""
    f = (flag > 0).astype(int)
    return int(((f.diff().fillna(f.iloc[0])) > 0).sum())


def exp1(T):
    """A Taylor rule versus a fixed money stock under demand and supply shocks."""
    rows, series = [], {}
    for sname, shock in (("demand: spending -40% for 20 ticks", DEMAND), ("supply: harvest -50% for 5 ticks", SUPPLY)):
        for rule, kw in RULES.items():
            s, w = shock_cost(T, "1", shock, kw)
            rows.append(dict(shock=sname, rule=rule, **s))
            series[(sname.split(":")[0], rule)] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp2(T):
    """Inflation: the central bank prints money faster than the economy grows."""
    rows, series = [], {}
    for g in (0.0, 0.0025, 0.005, 0.01):
        ws = []
        for seed in SEEDS:
            w = T.run(with_params(base(), seed=seed, ticks=300, **{"central_bank.rule": "money_growth",
                                                                     "central_bank.money_growth": g}), "2").wide()
            t = w.iloc[100:]
            ws.append(dict(money_growth_per_tick=float(np.log(w.money_supply.iloc[-1] / w.money_supply.iloc[100]) / 200),
                           inflation_per_tick=float(np.log(w.cpi.iloc[-1] / w.cpi.iloc[100]) / 200),
                           price_volatility=float(t.cpi.pct_change().std()),
                           bread=t.consumption_per_person.mean(), services=t.services_per_person.mean(),
                           unemployment=t.unemployment_rate.mean(), hungry=t.hungry_share.mean(),
                           real_wage=t.real_wage.mean(), bank_failures=int(w.bank_failure.sum())))
            if seed == SEEDS[0]:
                series[f"money +{g:.2%}/tick"] = w.reset_index()
        rows.append(dict(printing=f"{g:.2%} per tick", **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def exp3(T):
    """The zero lower bound: a deep slump when the rate cannot go below zero."""
    cases = {"fixed rate": {"central_bank.rule": "fixed_rate"},
             "Taylor, floor at 0 (base)": {},
             "Taylor, no floor (rates down to -5%)": {"central_bank.lower_bound": -0.05},
             "Taylor, floor at 0, higher neutral rate (1%)": {"central_bank.neutral_rate": 0.01}}
    rows, series = [], {}
    for label, kw in cases.items():
        s, w = shock_cost(T, "3", SLUMP, kw, window=(150, 230))
        rows.append(dict(policy=label, **s))
        series[label] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp4(T):
    """Lender of last resort: a rumour starts a run on a solvent bank."""
    rows, series = [], {}
    for label, kw in (("no lender of last resort", {"central_bank.lender_of_last_resort": False}),
                      ("central bank lends in the run", {})):
        ws = []
        for seed in SEEDS:
            w = T.run(with_params(base(), seed=seed, ticks=260, panic_start=150, panic_duration=6, **kw), "4").wide()
            win = w.loc[150:220]
            ws.append(dict(failure_episodes=episodes(win.bank_failure), emergency_lending=win.lolr_lent.sum(),
                           deposit_haircut=win.haircut.sum(), withdrawals_blocked_ticks=int(win.bank_failure.sum()),
                           credit_frozen_ticks=int(win.credit_frozen.sum()),
                           lending_in_window=win.new_lending.sum(),
                           lowest_reserves=win.reserves.min(), cash_hoarded_peak=win.cash_hoarded.max(),
                           bread=win.consumption_per_person.mean(), unemployment=win.unemployment_rate.mean()))
            if seed == SEEDS[0]:
                series[label] = w.reset_index()
        rows.append(dict(policy=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def exp5(T):
    """Bailouts and moral hazard in a credit boom."""
    boom = dict(ticks=400, procyclicality=0.05, capital_requirement=0.04, initial_capital=0.05)
    cases = {"bail-in (depositors pay)": {},
             "surprise bailouts": {"central_bank.bailouts": True},
             "expected bailouts (moral hazard)": {"central_bank.bailouts": True, "central_bank.moral_hazard": 0.9}}
    rows, series = [], {}
    for label, kw in cases.items():
        ws = []
        for seed in MORE_SEEDS:
            w = T.run(with_params(base(), seed=seed, **boom, **kw), "5").wide()
            ws.append(dict(loans_peak=w.loans.max(), loans_average=w.loans.mean(), lending_loosening=w.optimism.mean(),
                           defaults=w.defaults.sum(), failure_episodes=episodes(w.bank_failure),
                           bank_runs=int(w.bank_run.sum()), deposit_haircuts=w.haircut.sum(),
                           bailout_money=w.bailout_cost.iloc[-1],
                           bread=w.consumption_per_person.iloc[100:].mean(),
                           unemployment=w.unemployment_rate.iloc[100:].mean()))
            if seed == MORE_SEEDS[0]:
                series[label] = w.reset_index()
        rows.append(dict(policy=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def build_report(res, T, path):
    summary = {}
    body = ['<nav><a href="#e1">1 · Taylor vs money</a><a href="#e2">2 · inflation</a><a href="#e3">3 · zero lower bound</a>'
            '<a href="#e4">4 · lender of last resort</a><a href="#e5">5 · moral hazard</a></nav>']
    body.append("<p>A central bank joins the Stage 5 economy (households, many firms, services, a bank, a government). "
                "It sets the policy rate; the bank lends at that rate plus 0.7%, government bonds pay it, and households "
                "spend less when it is above its neutral level (0.3% per tick) and more when below. The base rule is a "
                "Taylor rule: the rate rises 1.5 points per point of inflation and falls when unemployment is above 4%, "
                "never below zero; during the first 50 ticks every rule keeps the rate neutral while prices settle, and the "
                "rate's effect on spending is bounded to ±50%. The bank keeps 20% of its deposits as cash reserves; the rest sits in central bank "
                f"bills it cannot use in a run. Averages of {len(SEEDS)} seeds unless noted.</p>")
    sections = [("e1", "1 · A Taylor rule versus a fixed money stock",
                 "Two shocks hit at tick 150: households suddenly want to save (spending −40% for 20 ticks), or the harvest "
                 "fails (−50% for 5 ticks). Losses are counted against the same seed without the shock, ticks 150–220. "
                 "Fixed money stock (monetary targeting): when money is above its starting level the rate rises 0.1 "
                 "points per 1% excess; when below, new money is paid per head. Fixed rate: the rate never moves.",
                 "1", "taylor_vs_money"),
                ("e2", "2 · Inflation: printing money",
                 "The money rule with the money supply's path growing 0–1% per tick (money short of the path is paid per "
                 "head). The economy's real output barely grows, so the extra money mostly raises prices — and disrupts "
                 "the economy on the way.",
                 "2", "inflation"),
                ("e3", "3 · The zero lower bound",
                 "A deep slump: spending −50% for 30 ticks. The Taylor rule wants a negative rate; with a floor at zero it "
                 "cannot deliver it.", "3", "zero_lower_bound"),
                ("e4", "4 · Lender of last resort",
                 "At tick 150 a rumour makes depositors withdraw 40% of their deposits per tick for 6 ticks, though the bank "
                 "is solvent. Its 20% cash reserves run out. Without help it suspends withdrawals (a failure with no "
                 "haircut, since it is solvent) and stops lending for 10 ticks.", "4", "lender_of_last_resort"),
                ("e5", "5 · Bailouts and moral hazard",
                 "A credit boom: the bank loosens its standards while loans perform (procyclicality 0.05) with thin capital "
                 "(4% required). When the bank becomes insolvent, either depositors take a haircut (bail-in) or the central "
                 "bank injects new money (bailout). If bailouts are expected, depositors never run and the bank barely "
                 f"tightens after defaults. {len(MORE_SEEDS)} seeds, 400 ticks.", "5", "moral_hazard")]
    plots = {
        "1": ("unemployment_rate", "Unemployment through the demand shock (one seed)", (140, 230)),
        "2": ("cpi", "Price index (one seed)", None),
        "3": ("policy_rate", "Policy rate in the slump (one seed); dotted: what the Taylor rule asks for", (140, 240)),
        "4": ("new_lending", "New bank lending around the run (one seed)", (140, 200)),
        "5": ("loans", "Loans outstanding (one seed)", None),
    }
    for anchor, title, text, key, sname in sections:
        t = res[key]["table"]
        summary[sname] = t.to_dict(orient="records")
        body.append(f'<h2 id="{anchor}">{title}</h2><p>{text}</p>')
        body.append(R.table_html(t))
        col, ptitle, xr = plots[key]
        f = go.Figure()
        for i, (k, w) in enumerate(res[key]["series"].items()):
            if key == "1" and k[0] != "demand":
                continue
            name = k[1] if isinstance(k, tuple) else str(k)
            f.add_trace(go.Scatter(x=w.tick, y=w[col], name=name, mode="lines", line=dict(width=2, color=R.SERIES[i % len(R.SERIES)])))
            if key == "3" and "Taylor" in name:
                f.add_trace(go.Scatter(x=w.tick, y=w.shadow_rate, name=f"{name}: rule", mode="lines", showlegend=False,
                                       line=dict(width=1, dash="dot", color=R.SERIES[i % len(R.SERIES)])))
        if xr:
            f.update_xaxes(range=list(xr))
        f.update_layout(title=ptitle, xaxis_title="tick", yaxis_title=col.replace("_", " "))
        body.append(R.fig_html(f))
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s. New in Stage 6: the policy rate never goes below its floor, the central bank's claims "
                "on the bank are never negative, and every new unit of money from the central bank (emergency loans, bailouts, "
                "per-head transfers) is an explicit creation that the bank's money-explained check accounts for.</p>")
    R.write(path, "Stage 6 · Central bank and monetary policy", "citysim — interest rates, money, runs and bailouts. "
            "Generated by <code>python -m citysim experiments stage6</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_6.html") -> dict:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T)}
    s = build_report(res, T, out)
    out.with_name("stage_6_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
