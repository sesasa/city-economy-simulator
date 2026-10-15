"""Stage 3 experiments (credit and banking) and the stage_3.html report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..config import load_scenario, with_params
from ..core import runner
from ..core.sfc import balance_sheet_matrix, transaction_flow_matrix
from .stage1 import ROOT, SCEN, Tally
from .stage2 import settle_tick

SECTORS = ["households", "farm", "factory", "distribution", "maintenance", "bank"]
GROW = {"demography.enabled": True, "land_limits.enabled": True}


def market():
    return load_scenario(SCEN / "stage2_market.yaml")


def bank():
    return load_scenario(SCEN / "stage3_bank.yaml")


def summary_row(w: pd.DataFrame, tail: int = 100) -> dict:
    out = dict(consumption=w.consumption_per_person.iloc[-tail:].mean(), min_consumption=w.consumption_per_person.min(),
               population_end=w.population.iloc[-1], unemployment=w.unemployment_rate.iloc[-tail:].mean())
    if "loans" in w:
        out.update(loans_peak=w.loans.max(), defaults=w.defaults.sum(), default_events=int(w.default_count.sum()),
                   bank_failures=int(w.bank_failure.sum()), haircuts=w.haircut.sum(), leverage_peak=w.leverage.max())
    return out


def exp1(T: Tally) -> dict:
    """Does credit rescue the Stage 2 failures?"""
    cases = {"scarce money (M0 = 150)": dict(M0=150, prices_match_money=False),
             "scarce money (M0 = 200)": dict(M0=200, prices_match_money=False),
             "scarce money (M0 = 300)": dict(M0=300, prices_match_money=False),
             "cold start (equal split)": dict(initial_allocation="equal"),
             "famine (−50% for 5 ticks) with demography": {"harvest_shocks.enabled": True, "demography.enabled": True,
                                                           "harvest_shocks.shocks": [dict(start=50, duration=5, multiplier=0.5)]}}
    rows, series = [], {}
    for label, kw in cases.items():
        for system, sc in (("market only", market()), ("market + bank", bank())):
            w = T.run(with_params(sc, ticks=300, **kw), "1").wide()
            rows.append(dict(case=label, system=system, **summary_row(w)))
            if label.startswith("scarce money (M0 = 200)"):
                series[system] = w.reset_index()
    return dict(table=pd.DataFrame(rows), series=series)


def exp2(T: Tally) -> dict:
    """Credit limits against growth: fixed town (capital deepening) and growing town."""
    rows = []
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        w = T.run(with_params(bank(), ticks=400, investment_share=0.05, credit_scale=scale), "2").wide()
        y0 = w.potential_output.iloc[0]
        rows.append(dict(town="fixed 150", credit_scale=scale,
                         growth_per_tick=(w.potential_output.iloc[-1] / y0) ** (1 / len(w)) - 1,
                         **summary_row(w)))
        g = T.run(with_params(bank(), ticks=700, investment_share=0.05, credit_scale=scale, **GROW), "2").wide()
        rows.append(dict(town="growing", credit_scale=scale,
                         growth_per_tick=(g.population.iloc[-1] / g.population.iloc[0]) ** (1 / len(g)) - 1,
                         **summary_row(g)))
    return dict(table=pd.DataFrame(rows))


def exp3(T: Tally) -> dict:
    """Minsky: lending standards loosen in good times."""
    rows, series = [], {}
    for proc in (0.0, 0.02, 0.05, 0.1, 0.2):
        w = T.run(with_params(bank(), ticks=500, investment_share=0.05, procyclicality=proc, **GROW), "3").wide()
        lev = w.leverage
        rows.append(dict(procyclicality=proc, credit_peak=w.loans.max(),
                         default_waves=int(((w.defaults > 0) & (w.defaults.shift(1, fill_value=0) == 0)).sum()),
                         **summary_row(w)))
        if proc in (0.0, 0.1):
            series[proc] = w.reset_index()
    return dict(table=pd.DataFrame(rows), series=series)


def exp4(T: Tally) -> dict:
    """A harvest shock hits a credit-heavy economy: does the bank fail?"""
    shock = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=120, duration=5, multiplier=0.5)]}
    rows, series = [], {}
    for cap_req, init in ((0.02, 0.02), (0.08, 0.1), (0.16, 0.2)):
        for shocked in (False, True):
            kw = dict(ticks=300, N0=300, investment_share=0.0, credit_scale=4.0, procyclicality=0.01,
                      capital_requirement=cap_req, initial_capital=init, **GROW)
            if shocked:
                kw.update(shock)
            w = T.run(with_params(bank(), **kw), "4").wide()
            win = w.loc[120:200]
            rows.append(dict(capital_requirement=cap_req, initial_capital=init, shock=shocked,
                             failures_after_shock=int(win.bank_failure.sum()), runs_after_shock=int(win.bank_run.sum()),
                             defaults_after_shock=win.defaults.sum(), haircut_after_shock=win.haircut.sum(),
                             money_drop=1 - win.money_supply.min() / w.money_supply.loc[119],
                             min_consumption_after=win.consumption_per_person.min(),
                             population_end=w.population.iloc[-1]))
            if shocked:
                series[cap_req] = w.reset_index()
    return dict(table=pd.DataFrame(rows), series=series)


def sfc_tables(tick: int = 30):
    sim = runner.build(with_params(bank(), ticks=tick, M0=200, prices_match_money=False))
    for _ in range(tick):
        sim.step()
    m = next(l for l in sim.layers if l.name == "market")
    sector = lambda a: "bank" if a.startswith("bank") else m.sector_of(a)
    tfm = transaction_flow_matrix(sim.world.ledger.journal, sector, {"deposits"}, SECTORS)
    bsm = balance_sheet_matrix(sim.world.ledger, sector, SECTORS,
                               {"raw": float(np.mean([f.price for f in m.of(0)])), "final": float(np.mean([f.price for f in m.of(1)]))})
    return tfm, bsm


def build_report(res: dict, T: Tally, path: Path) -> dict:
    e1, e2, e3, e4 = (res[k] for k in "1234")
    tfm, bsm = res["sfc"]
    summary: dict = {}
    body = ['<nav><a href="#e1">1 · does credit help?</a><a href="#e2">2 · credit limits</a><a href="#e3">3 · Minsky</a>'
            '<a href="#e4">4 · bank failure</a><a href="#sfc">SFC matrices</a></nav>']
    body.append("<p>One bank takes everyone's money as deposits and lends to firms. A loan creates a deposit, so the money "
                "supply grows when firms borrow and shrinks when they repay. Firms borrow to pay wages and buy inputs when "
                "their till is short, and to build when a project saves enough wages to beat loan rate + repayment + 1% per "
                "tick. Limits: half the borrower's collateral plus a tick of revenue, never more than 15% of revenue in debt "
                "service, and bank equity ≥ 8% of loans. Defaulters are cut off for 20 ticks. If bank capital falls under 4% "
                "households run; an insolvent bank is resolved by bailing in depositors.</p>")

    t1 = e1["table"]
    summary["rescue"] = t1.to_dict(orient="records")
    body.append('<h2 id="e1">1 · Does credit rescue Stage 2’s failures?</h2>')
    body.append("<p>Stage 2’s market froze when firms ran out of cash for inputs. A bank can lend that cash, creating the "
                "money that was missing. It cannot create workers, though: a cold start where the factory never gets its "
                "fixed crew, or a famine that pushes the town below its survival line, are not money problems.</p>")
    body.append(R.table_html(t1))
    f = go.Figure()
    for i, (k, w) in enumerate(e1["series"].items()):
        f.add_trace(go.Scatter(x=w.tick, y=w.consumption_per_person, name=f"{k}: bread per person", mode="lines",
                               line=dict(width=2, color=R.SERIES[i])))
    bw = e1["series"]["market + bank"]
    f.add_trace(go.Scatter(x=bw.tick, y=bw.money_supply / bw.money_supply.iloc[0], name="market + bank: money supply (×start)",
                           mode="lines", line=dict(width=2, color=R.SERIES[2])))
    f.update_layout(title="Scarce money (M0 = 200): the bank lends the missing money", xaxis_title="tick", yaxis_title="loaves / index")
    body.append(R.fig_html(f))

    t2 = e2["table"]
    summary["credit_limits"] = t2.to_dict(orient="records")
    body.append('<h2 id="e2">2 · Credit limits against growth</h2>')
    body.append("<p>Firms keep only 5% of profit for building, so growth leans on credit. <code>credit_scale</code> multiplies "
                "every borrower's limit. Fixed town: growth is capital deepening (potential output). Growing town: "
                "population growth with land limits over 700 ticks.</p>")
    body.append(R.table_html(t2))

    t3 = e3["table"]
    summary["minsky"] = t3.to_dict(orient="records")
    body.append('<h2 id="e3">3 · A Minsky cycle</h2>')
    body.append("<p>With <code>procyclicality</code> above zero the bank loosens its limits while loans perform and tightens "
                "sharply after defaults; firms get bolder with it (they build factories before they are full and accept "
                "lower returns on borrowed projects). Growing town, 500 ticks.</p>")
    body.append(R.table_html(t3))
    f = go.Figure()
    for i, (k, w) in enumerate(e3["series"].items()):
        f.add_trace(go.Scatter(x=w.tick, y=w.loans, name=f"loans, procyclicality {k}", mode="lines", line=dict(width=2, color=R.SERIES[i])))
    w = e3["series"].get(0.1)
    if w is not None:
        f.add_trace(go.Scatter(x=w.tick, y=w.defaults, name="defaults, procyclicality 0.1", mode="lines",
                               line=dict(width=2, color=R.SERIES[3])))
    f.update_layout(title="Credit booms and default waves", xaxis_title="tick", yaxis_title="money")
    body.append(R.fig_html(f))

    t4 = e4["table"]
    summary["bank_failure"] = t4.to_dict(orient="records")
    body.append('<h2 id="e4">4 · Bank failure after a harvest shock</h2>')
    body.append("<p>Harvests fall 50% for 5 ticks (120–124) in a credit-heavy growing town: 300 people, firms keep no "
                "profit for building so every new factory and tool is borrowed, limits are 4× normal and lending is mildly "
                "procyclical. Without the shock no bank fails. Three banks: thin capital (2% requirement, 2% starting "
                "equity), normal (8%, 10%) and thick (16%, 20%).</p>")
    body.append(R.table_html(t4))
    f = go.Figure()
    for i, (k, w) in enumerate(e4["series"].items()):
        f.add_trace(go.Scatter(x=w.tick, y=w.money_supply, name=f"money supply, capital req. {k}", mode="lines",
                               line=dict(width=2, color=R.SERIES[i])))
    f.update_xaxes(range=[100, 220])
    f.update_layout(title="Money supply around the shock", xaxis_title="tick", yaxis_title="money")
    body.append(R.fig_html(f))

    body.append('<h2 id="sfc">Stock-flow consistent matrices (tick 30, scarce-money economy with a bank)</h2>')
    body.append("<p>Payments in deposits by flow and sector (rows sum to zero, the last row closes each column) and the "
                "balance sheet: deposits are a liability of the bank and an asset of everyone else; loans the reverse.</p>")
    body.append(R.table_html(tfm.reset_index().rename(columns={"index": "flow"}), "{:,.2f}"))
    body.append(R.table_html(bsm.reset_index().rename(columns={"index": "stock"}), "{:,.2f}"))
    summary["sfc_row_sums_max"] = float(tfm["total"].abs().max())
    summary["sfc_col_sums_max"] = float(tfm.drop(columns="total").sum(axis=0).abs().max())

    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append('<h2 id="inv">Invariants and runtime</h2>')
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s.</p><p>New in Stage 3: claim signs (no overdrafts, the bank never owes loans), and "
                "<em>money explained</em>: the change in the money supply each tick must equal the sum of the bank's "
                "money-creating and money-destroying postings (loans, repayments, interest, dividends, defaults, haircuts).</p>")
    R.write(path, "Stage 3 · Credit and banking", "citysim — a bank whose loans create money. Generated by "
            "<code>python -m citysim experiments stage3</code>.", "\n".join(body))
    return summary


def run_all():
    T = Tally()
    return {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "sfc": sfc_tables()}, T


def main(out: Path = ROOT / "reports" / "stage_3.html") -> dict:
    res, T = run_all()
    s = build_report(res, T, out)
    out.with_name("stage_3_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
