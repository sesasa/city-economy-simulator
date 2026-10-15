"""Stage 10 experiments (the modern economy and comparing allocation systems) and the stage_10.html report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..config import load_scenario, with_params
from .stage1 import ROOT, SCEN, Tally
SEEDS = (1, 2, 3, 4, 5)

MODERN = SCEN / "stage10_modern.yaml"
PLANNED = SCEN / "stage10_planned.yaml"
HARVEST = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=300, duration=5, multiplier=0.5)]}
DEMAND = {"demand_shocks.enabled": True, "demand_shocks.shocks": [dict(start=300, duration=20, multiplier=0.6)]}
SYSTEMS = {
    "central planning": (PLANNED, {}),
    "free market": (MODERN, {"government.enabled": False, "central_bank.enabled": False, "risk_sharing.enabled": False}),
    "mixed economy": (MODERN, {}),
}


def col(w, name, default=0.0):
    return w[name] if name in w else pd.Series(default, index=w.index)


def total(w):
    return col(w, "consumption_per_person") + col(w, "services_per_person")


def outcomes(w: pd.DataFrame) -> dict:
    t = w.loc[100:]
    tot = total(w)
    return dict(output=float(tot.iloc[-200:].mean()), bread=float(col(w, "consumption_per_person").iloc[-200:].mean()),
                services=float(col(w, "services_per_person").iloc[-200:].mean()),
                growth=float(tot.iloc[-100:].mean() / tot.loc[200:300].mean() - 1),
                technology=float(col(w, "tech_level", 1.0).iloc[-1]),
                instability=float(tot.loc[100:].std()), unemployment=float(col(t, "unemployment_rate").mean()),
                unemployment_sd=float(col(t, "unemployment_rate").std()),
                inequality=float(col(t, "gini_income").mean()))


def loss(w, cf, a=300, b=380):
    return float((total(cf).loc[a:b] - total(w).loc[a:b]).sum())


def exp1(T):
    """Does the modern scenario look like a rich economy today?"""
    rows = []
    for seed in SEEDS:
        w = T.run(with_params(load_scenario(MODERN), seed=seed), "1").wide().iloc[-200:]
        ls = lambda s: float(col(w, f"labor_share/{s}").mean())
        services = ls("services") + ls("distribution") + ls("public") + ls("research")
        rows.append(dict(agriculture=ls("farm"), manufacturing=ls("factory") + ls("maintenance") + ls("building"),
                         services=services, unemployed=float(col(w, "unemployment_rate").mean()),
                         deposits_share_of_money=float(1 - (col(w, "cash_hoarded") / col(w, "money_supply")).mean())))
    t = pd.DataFrame(rows).mean().to_dict()
    target = {"agriculture": "1–3%", "manufacturing": "10–20%", "services": "the large majority",
              "unemployed": "low (3–8%)", "deposits_share_of_money": "most"}
    return dict(table=pd.DataFrame([dict(measure=k, model=v, rich_economies_today=target[k]) for k, v in t.items()]))


def exp2(T):
    """Central planning, free markets and the mixed economy."""
    rows, series = [], {}
    for label, (path, kw) in SYSTEMS.items():
        ws = []
        for seed in SEEDS:
            sc = with_params(load_scenario(path), seed=seed, **kw)
            w = T.run(sc, "2").wide()
            r = outcomes(w)
            r["harvest_shock_loss"] = loss(T.run(with_params(sc, **HARVEST), "2").wide(), w)
            r["demand_shock_loss"] = loss(T.run(with_params(sc, **DEMAND), "2").wide(), w) if path == MODERN else np.nan
            ws.append(r)
            if seed == SEEDS[0]:
                series[label] = w.reset_index()
        rows.append(dict(system=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


ABLATION = {
    "services": {"services.enabled": False},
    "banking (with central bank and stock market)": {"banking.enabled": False, "central_bank.enabled": False,
                                                      "stock_market.enabled": False},
    "entry and exit": {"entry_exit.enabled": False},
    "government": {"government.enabled": False},
    "central bank": {"central_bank.enabled": False},
    "stock market": {"stock_market.enabled": False},
    "insurance and futures": {"risk_sharing.enabled": False},
    "technology": {"technology.enabled": False},
}


def exp3(T):
    """Which layer contributes most to each outcome? Switch them off one at a time."""
    def run_case(kw):
        ws = []
        for seed in SEEDS:
            sc = with_params(load_scenario(MODERN), seed=seed, **kw)
            w = T.run(sc, "3").wide()
            r = outcomes(w)
            r["harvest_shock_loss"] = loss(T.run(with_params(sc, **HARVEST), "3").wide(), w)
            ws.append(r)
        return pd.DataFrame(ws).mean()
    base = run_case({})
    rows = [dict(without="(nothing: full mixed economy)", **base.to_dict())]
    for label, kw in ABLATION.items():
        r = run_case(kw)
        rows.append(dict(without=label, **r.to_dict()))
    t = pd.DataFrame(rows)
    delta = t.copy()
    for c in base.index:
        delta[c] = t[c] - base[c]
    return dict(table=t, delta=delta.iloc[1:])


def build_report(res, T, path):
    summary = {}
    body = ['<nav><a href="#e1">1 · calibration</a><a href="#e2">2 · systems</a><a href="#e3">3 · layers</a></nav>']
    body.append("<p>The modern economy: 1,000 people, individual households and many firms, services, a bank, a "
                "central bank (Taylor rule), a government (taxes, benefits, roads), a stock market with margin "
                "lending, crop insurance and a research institute (3% of people) with slow diffusion of technology. "
                "Farms are productive (40 bushels per farmer), bakeries need few hands. The planned economy has the "
                "same people, technology and research: the planner feeds everyone, invests 30% of free labor and "
                f"puts 80% of the rest into services. {len(SEEDS)} seeds, 600 ticks. Output = bread + services per "
                "person (last 200 ticks); growth = output in the last 100 ticks vs ticks 200–300; instability = sd of "
                "output after tick 100; shock losses = output lost per person over 80 ticks after a harvest failure "
                "(−50%, 5 ticks) or a fall in spending (−40%, 20 ticks), against the same seed without the shock.</p>")
    sections = [("e1", "1 · Does the modern economy look like a rich economy today?",
                 "Employment shares in the last 200 ticks (services include retail delivery, public works and "
                 "research; manufacturing includes repair and construction).", "1", "calibration"),
                ("e2", "2 · Central planning, free markets and the mixed economy",
                 "Free market: no government, no central bank (the bank lends at a fixed rate), no insurance; research "
                 "is paid by a levy on households. Planned economies have no unemployment and no income inequality by "
                 "construction, and nobody's spending can fall, so a demand shock does not apply.", "2", "systems"),
                ("e3", "3 · Which layer contributes most?",
                 "The mixed economy with one layer switched off at a time (banking also takes the central bank and "
                 "stock market with it). Second table: change against the full economy.", "3", "ablation")]
    for anchor, title, text, key, sname in sections:
        t = res[key]["table"]
        summary[sname] = t.to_dict(orient="records")
        body.append(f'<h2 id="{anchor}">{title}</h2><p>{text}</p>')
        body.append(R.table_html(t))
        if key == "2":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=total(w), name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
            f.update_layout(title="Bread + services per person (one seed)", xaxis_title="tick", yaxis_title="units per person")
            body.append(R.fig_html(f))
        if key == "3":
            summary["ablation_delta"] = res[key]["delta"].to_dict(orient="records")
            body.append("<h3>Change when the layer is switched off</h3>")
            body.append(R.table_html(res[key]["delta"]))
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s — every layer's checks at once: goods and money conservation, balance sheets, the "
                "bank's money accounting, the central bank's claims, shares and margin debt, the insurer, the frontier, "
                "the planner's physics and the planned services' labor.</p>")
    R.write(path, "Stage 10 · The modern economy and comparing systems", "citysim — planning, markets and the mixed "
            "economy. Generated by <code>python -m citysim experiments stage10</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_10.html") -> dict:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T)}
    s = build_report(res, T, out)
    out.with_name("stage_10_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
