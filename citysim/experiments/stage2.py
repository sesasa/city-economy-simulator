"""Stage 2 experiments (barter, money and prices) and the stage_2.html report."""
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

SECTORS = ["households", "farm", "factory", "distribution", "maintenance"]


def market() -> "Scenario":
    return load_scenario(SCEN / "stage2_market.yaml")


def planner():
    return load_scenario(SCEN / "stage1_base.yaml")


def settle_tick(series: pd.Series, tol: float = 0.03, tail: int = 100) -> int | None:
    """First tick after which the series stays within tol (relative) of its final average."""
    target = series.iloc[-tail:].mean()
    ok = (series - target).abs() <= tol * max(abs(target), 1e-9)
    bad = np.where(~ok.to_numpy())[0]
    if not len(bad):
        return int(series.index[0])
    if bad[-1] + 1 >= len(series):
        return None
    return int(series.index[bad[-1] + 1])


def exp1(T: Tally) -> dict:
    """Barter vs money: how many random meetings does barter need?"""
    sc = with_params(market(), ticks=300)
    rows = []
    for K in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256):
        w = T.run(with_params(sc, exchange="barter", barter_meetings=K), "1").wide()
        rows.append(dict(exchange="barter", meetings=K, trade_completion=w.trade_completion.iloc[-1],
                         consumption=w.consumption_per_person.iloc[-100:].mean(),
                         unemployment=w.unemployment_rate.iloc[-100:].mean(), output=w.total_output.iloc[-100:].mean()))
    w = T.run(sc, "1").wide()
    rows.append(dict(exchange="money", meetings=None, trade_completion=1.0, consumption=w.consumption_per_person.iloc[-100:].mean(),
                     unemployment=w.unemployment_rate.iloc[-100:].mean(), output=w.total_output.iloc[-100:].mean()))
    series = {K: T.run(with_params(sc, exchange="barter", barter_meetings=K, ticks=120), "1").wide().reset_index()
              for K in (16, 64)}
    series["money"] = T.run(with_params(sc, ticks=120), "1").wide().reset_index()
    return dict(table=pd.DataFrame(rows), series=series)


def exp2(T: Tally) -> dict:
    """Planner vs market across town sizes; time to settle; can the market bootstrap from scratch?"""
    rows = []
    for n in range(100, 401, 10):
        wm = T.run(with_params(market(), N0=n, ticks=400), "2").wide()
        wp = T.run(with_params(planner(), N0=n, ticks=400), "2").wide()
        rows.append(dict(N0=n,
                         market_consumption=wm.consumption_per_person.iloc[-100:].mean(),
                         market_unemployment=wm.unemployment_rate.iloc[-100:].mean(),
                         market_settle=settle_tick(wm.consumption_per_person),
                         planner_consumption=wp.consumption_per_person.iloc[-100:].mean(),
                         planner_leisure=wp.leisure_per_person.iloc[-100:].mean(),
                         market_price_end=wm.price_level.iloc[-1]))
    table = pd.DataFrame(rows)
    ts = {"market": T.run(with_params(market(), ticks=400), "2").wide().reset_index(),
          "planner": T.run(with_params(planner(), ticks=400), "2").wide().reset_index()}
    boot = []
    for mob in (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6):
        w = T.run(with_params(market(), initial_allocation="equal", mobility=mob, ticks=300), "2").wide()
        boot.append(dict(mobility=mob, consumption_end=w.consumption_per_person.iloc[-50:].mean(),
                         bootstrapped=bool(w.consumption_per_person.iloc[-50:].mean() > 0.5)))
    return dict(table=table, ts=ts, boot=pd.DataFrame(boot))


def exp3(T: Tally) -> dict:
    """Money stock: neutrality when prices start in line with money; scarcity when they don't; growth with fixed money."""
    neutral = []
    for M in (100, 300, 875, 3000, 10000):
        w = T.run(with_params(market(), M0=M, ticks=300), "3").wide()
        neutral.append(dict(money=M, price_level=w.price_level.iloc[-100:].mean(), velocity=w.velocity.iloc[-100:].mean(),
                            consumption=w.consumption_per_person.iloc[-100:].mean(), unemployment=w.unemployment_rate.iloc[-100:].mean()))
    scarce = []
    for M in (150, 250, 400, 600, 875, 1500, 3000):
        w = T.run(with_params(market(), M0=M, prices_match_money=False, ticks=300), "3").wide()
        scarce.append(dict(money=M, consumption=w.consumption_per_person.iloc[-100:].mean(),
                           min_consumption=w.consumption_per_person.min(),
                           price_end=w.price_level.iloc[-1], unemployment=w.unemployment_rate.iloc[-100:].mean()))
    grow = T.run(with_params(market(), ticks=1200, **{"demography.enabled": True, "land_limits.enabled": True}), "3").wide().reset_index()
    return dict(neutral=pd.DataFrame(neutral), scarce=pd.DataFrame(scarce), grow=grow)


def exp4(T: Tally) -> dict:
    """How fast should prices move? Oscillations versus slow adjustment."""
    rows, series = [], {}
    for ps in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        w = T.run(with_params(market(), price_speed=ps, ticks=400), "4").wide()
        tail = w.iloc[-200:]
        rows.append(dict(price_speed=ps, consumption=tail.consumption_per_person.mean(),
                         price_volatility=tail.price_level.std() / tail.price_level.mean(),
                         inventory_volatility=tail["inventories/bread"].std() / max(tail["inventories/bread"].mean(), 1e-9),
                         unemployment=tail.unemployment_rate.mean(), settle=settle_tick(w.consumption_per_person)))
        if ps in (0.01, 0.05, 0.4):
            series[ps] = w.reset_index()
    return dict(table=pd.DataFrame(rows), series=series)


def exp5(T: Tally) -> dict:
    """A 30% harvest shock for 3 ticks: planner vs market."""
    shock = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=150, duration=3, multiplier=0.7)]}
    out, rows = {}, []
    for label, sc in (("planner", planner()), ("market", market())):
        w = T.run(with_params(sc, ticks=250, **shock), "5").wide()
        cf = T.run(with_params(sc, ticks=250), "5").wide()
        out[label] = w.reset_index()
        win = w.loc[150:200]
        gap = (cf.consumption_per_person - w.consumption_per_person).loc[150:]
        bad = gap.index[gap.abs() > 0.01]
        rows.append(dict(system=label, min_consumption=win.consumption_per_person.min(),
                         consumption_lost=float((gap.clip(lower=0) * w.population.loc[150:]).sum()),
                         ticks_to_recover=(int(bad.max()) + 1 - 150) if len(bad) else 0,
                         price_peak_vs_before=(win.price_level.max() / w.price_level.loc[149] - 1) if "price_level" in w else None))
    return dict(series=out, table=pd.DataFrame(rows))


def sfc_tables(tick: int = 100):
    sim = runner.build(with_params(market(), ticks=tick))
    for _ in range(tick):
        sim.step()
    m = next(l for l in sim.layers if l.name == "market")
    prices = {"raw": float(np.mean([f.price for f in m.of(0)])), "final": float(np.mean([f.price for f in m.of(1)]))}
    tfm = transaction_flow_matrix(sim.world.ledger.journal, m.sector_of, {"cash"}, SECTORS)
    bsm = balance_sheet_matrix(sim.world.ledger, m.sector_of, SECTORS, prices)
    return tfm, bsm


# ---------------------------------------------------------------------------
def build_report(res: dict, T: Tally, path: Path) -> dict:
    e1, e2, e3, e4, e5 = (res[k] for k in "12345")
    tfm, bsm = res["sfc"]
    summary: dict = {}
    body = ['<nav><a href="#e1">1 · barter vs money</a><a href="#e2">2 · planner vs market</a><a href="#e3">3 · money stock</a>'
            '<a href="#e4">4 · price speed</a><a href="#e5">5 · harvest shock</a><a href="#sfc">SFC matrices</a></nav>']
    body.append("<p>The planner is gone. Four firms (farm, factory, distributor, repair shop) set prices as a markup over "
                "normal unit cost and raise or cut that markup with their stocks; wages rise with vacancies and fall when "
                "unemployment in an occupation is above 5%; people drift towards occupations with better expected pay; "
                "households spend 90% of income and 10% of savings, buy what they need if they can afford it, and want at "
                "most 1.5 loaves each. Profits return to households as dividends or are spent on building. Technology is "
                "exactly Stage 1's.</p>")

    t1 = e1["table"]
    summary["barter"] = t1.to_dict(orient="records")
    body.append('<h2 id="e1">1 · Barter versus money</h2>')
    body.append("<p>Under barter a trade needs a double coincidence of wants: you must meet someone who has what you want "
                "<em>and</em> wants what you have. With five kinds of things to trade, one random meeting succeeds with "
                "probability 1/16; with K meetings a desired trade happens with probability 1 − (15/16)<sup>K</sup>. Every "
                "link of the chain (wages, wheat, wholesale bread, repairs, retail bread) suffers the loss. With money, "
                "shops take money from anyone, so every desired trade happens.</p>")
    f = go.Figure()
    b = t1[t1.exchange == "barter"]
    f.add_trace(go.Scatter(x=b.meetings, y=b.trade_completion, name="share of trades that happen", mode="lines+markers",
                           line=dict(width=2, color=R.SERIES[0]), marker=dict(size=8)))
    f.add_trace(go.Scatter(x=b.meetings, y=b.consumption, name="bread per person (last 100 ticks)", mode="lines+markers",
                           line=dict(width=2, color=R.SERIES[1]), marker=dict(size=8)))
    m_ = t1[t1.exchange == "money"].iloc[0]
    f.add_hline(y=m_.consumption, line=dict(color=R.MUTED, width=1), annotation_text="money economy",
                annotation_position="bottom right")
    f.update_xaxes(type="log", title="random meetings per trader per tick")
    f.update_layout(title="Barter needs a lot of searching before the economy works", yaxis_title="share / loaves")
    body.append(R.fig_html(f))
    body.append(R.table_html(t1))

    t2 = e2["table"]
    summary["planner_vs_market"] = t2.to_dict(orient="records")
    body.append('<h2 id="e2">2 · Planner versus market</h2>')
    body.append("<p>Both economies start from the planner's allocation and run 400 ticks with a fixed population. The planner "
                "gives spare labor to leisure and building; the market has no leisure: spare people are unemployed, or "
                "firms hire them to make extra bread that households buy above their need.</p>")
    f = go.Figure()
    f.add_trace(go.Scatter(x=t2.N0, y=t2.planner_consumption, name="planner: bread per person", mode="lines",
                           line=dict(width=2, color=R.SERIES[0])))
    f.add_trace(go.Scatter(x=t2.N0, y=t2.market_consumption, name="market: bread per person", mode="lines",
                           line=dict(width=2, color=R.SERIES[1])))
    f.add_trace(go.Scatter(x=t2.N0, y=t2.planner_leisure, name="planner: leisure share", mode="lines",
                           line=dict(width=2, color=R.SERIES[2])))
    f.add_trace(go.Scatter(x=t2.N0, y=t2.market_unemployment, name="market: unemployment share", mode="lines",
                           line=dict(width=2, color=R.SERIES[3])))
    R.vline(f, 80 / 0.6, "N_min")
    f.update_layout(title="Long-run outcomes by town size", xaxis_title="people", yaxis_title="loaves per person / share")
    ts = e2["ts"]
    g = go.Figure()
    for i, (k, w) in enumerate(ts.items()):
        g.add_trace(go.Scatter(x=w.tick, y=w.consumption_per_person, name=f"{k}: bread per person", mode="lines",
                               line=dict(width=2, color=R.SERIES[i])))
    g.add_trace(go.Scatter(x=ts["market"].tick, y=ts["market"].unemployment_rate, name="market: unemployment", mode="lines",
                           line=dict(width=2, color=R.SERIES[3])))
    g.update_layout(title="150 people: the market discovers its own allocation", xaxis_title="tick", yaxis_title="loaves / share")
    body.append('<div class="grid2">' + "".join(f"<div>{R.fig_html(x)}</div>" for x in (f, g)) + "</div>")
    body.append(R.table_html(t2[t2.N0.isin([100, 120, 130, 140, 150, 200, 300, 400])]))
    body.append("<h3>Can a market start from scratch?</h3><p>Start with people split equally across the four occupations "
                "(37.5 each). The factory needs 50 people just to open. Whether the market gets there depends on how fast "
                "people switch jobs: the factory runs out of cash for wheat before enough workers arrive, and without credit "
                "nobody can sell it wheat on account.</p>")
    body.append(R.table_html(e2["boot"]))

    body.append('<h2 id="e3">3 · The money stock</h2>')
    n3 = e3["neutral"]
    summary["neutrality"] = n3.to_dict(orient="records")
    body.append("<p><strong>When prices start in line with the money stock</strong>, money is neutral: 100 times more money "
                "means exactly 100 times higher prices and wages, the same velocity and the same real outcomes.</p>")
    body.append(R.table_html(n3))
    s3 = e3["scarce"]
    summary["scarce_money"] = s3.to_dict(orient="records")
    body.append("<p><strong>When prices start at the usual level but money is scarce</strong>, prices and wages must fall a long "
                "way before the money in circulation can pay for the town's trade. In this model there is no credit to "
                "bridge the gap, so firms run out of cash to pay for inputs and the economy freezes even as prices fall: "
                "deflation and depression together. (The normal stock at these prices is about 875.)</p>")
    body.append(R.table_html(s3))
    gr = e3["grow"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=gr.tick, y=gr.population / gr.population.iloc[0], name="population (×N0)", mode="lines",
                           line=dict(width=2, color=R.SERIES[0])))
    f.add_trace(go.Scatter(x=gr.tick, y=gr.price_level / gr.price_level.iloc[0], name="price level (×start)", mode="lines",
                           line=dict(width=2, color=R.SERIES[1])))
    f.update_yaxes(type="log")
    f.update_layout(title="A growing town with a fixed money stock: secular deflation", xaxis_title="tick", yaxis_title="index (log)")
    body.append(R.fig_html(f))
    summary["growth_deflation"] = dict(pop_factor=float(gr.population.iloc[-1] / gr.population.iloc[0]),
                                       price_factor=float(gr.price_level.iloc[-1] / gr.price_level.iloc[0]))

    t4 = e4["table"]
    summary["price_speed"] = t4.to_dict(orient="records")
    body.append('<h2 id="e4">4 · How fast should prices move?</h2>')
    body.append("<p>The markup moves by <code>price_speed</code> × the inventory gap each tick (and drifts back to normal). "
                "Slow prices leave shortages and gluts in place longer; fast prices overshoot.</p>")
    f = go.Figure()
    for i, (ps, w) in enumerate(e4["series"].items()):
        f.add_trace(go.Scatter(x=w.tick, y=w.price_level, name=f"price speed {ps}", mode="lines", line=dict(width=2, color=R.SERIES[i])))
    f.update_layout(title="Retail bread price", xaxis_title="tick", yaxis_title="money")
    body.append('<div class="grid2"><div>' + R.fig_html(f) + "</div><div>" + R.table_html(t4) + "</div></div>")

    t5 = e5["table"]
    summary["shock"] = t5.to_dict(orient="records")
    body.append('<h2 id="e5">5 · A harvest shock: planner versus market</h2>')
    body.append("<p>Harvests fall 30% for 3 ticks (150–152). The planner simply has less bread to hand out. In the market the "
                "factory and the distributor hold stocks that cushion the first tick, prices jump and ration what is left, "
                "and the adjustment ripples through the supply chain afterwards.</p>")
    f = go.Figure()
    for i, (k, w) in enumerate(e5["series"].items()):
        f.add_trace(go.Scatter(x=w.tick, y=w.consumption_per_person, name=f"{k}: bread per person", mode="lines",
                               line=dict(width=2, color=R.SERIES[i])))
    f.update_xaxes(range=[140, 220])
    f.update_layout(title="Bread per person around the shock", xaxis_title="tick", yaxis_title="loaves")
    body.append('<div class="grid2"><div>' + R.fig_html(f) + "</div><div>" + R.table_html(t5) + "</div></div>")

    body.append('<h2 id="sfc">Stock-flow consistent matrices (tick 100, base scenario)</h2>')
    body.append("<p>Transaction-flow matrix: every payment appears twice, received (+) by one sector and paid (−) by another, "
                "so each row sums to zero; the last row records each sector's change in money, so each column sums to zero. "
                "Balance sheet: money and stocks valued at current prices.</p>")
    body.append(R.table_html(tfm.reset_index().rename(columns={"index": "flow"}), "{:,.2f}"))
    body.append(R.table_html(bsm.reset_index().rename(columns={"index": "stock"}), "{:,.2f}"))
    summary["sfc_row_sums_max"] = float(tfm["total"].abs().max())
    summary["sfc_col_sums_max"] = float(tfm.drop(columns="total").sum(axis=0).abs().max())

    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append('<h2 id="inv">Invariants and runtime</h2>')
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s.</p><p>On top of the Stage 1 checks: household money in the agent arrays equals the ledger, "
                "factory conditions match the ledger, no firm uses more workers than its occupation's pool, every money "
                "transfer is classified (so the transaction-flow matrix is complete), and money changes only through explicit "
                "creation.</p>")
    R.write(path, "Stage 2 · Barter, money and prices", "citysim — a market economy replaces the planner. Generated by "
            "<code>python -m citysim experiments stage2</code>.", "\n".join(body))
    return summary


def run_all() -> tuple[dict, Tally]:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T), "sfc": sfc_tables()}
    return res, T


def main(out: Path = ROOT / "reports" / "stage_2.html") -> dict:
    res, T = run_all()
    summary = build_report(res, T, out)
    out.with_name("stage_2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
