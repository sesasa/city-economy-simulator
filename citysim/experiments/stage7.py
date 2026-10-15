"""Stage 7 experiments (financial markets and risk sharing) and the stage_7.html report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..config import load_scenario, with_params
from .stage1 import ROOT, SCEN, Tally

SEEDS5 = (1, 2, 3, 4, 5)
SEEDS8 = tuple(range(1, 9))
WEATHER = {"harvest_shocks.enabled": True, "harvest_shocks.volatility": 0.2}
MOOD = {"demand_shocks.enabled": True, "demand_shocks.volatility": 0.05}


def base():
    return load_scenario(SCEN / "stage7_markets.yaml")


def market_stats(w: pd.DataFrame, skip: int = 50) -> dict:
    t = w.iloc[skip:]
    out = dict(bread=t.consumption_per_person.mean(), unemployment=t.unemployment_rate.mean(),
               unemployment_sd=t.unemployment_rate.std())
    if "share_price" in t:
        pv = t.price_to_value.clip(lower=1e-9)
        out.update(bubble_p95=float(pv.quantile(0.95)), typical_mispricing=float(np.log(pv).abs().median()),
                   worst_crash=float((1 - t.share_price / t.share_price.cummax()).max()),
                   peak_margin_debt=t.margin_debt.max(), fire_sales=t.forced_sales.sum(),
                   margin_defaults=t.margin_defaults.sum(), bank_failures=int(t.bank_failure.sum()))
    return out


def averaged(T, exp, seeds, **kw):
    rows, first = [], None
    for seed in seeds:
        w = T.run(with_params(base(), seed=seed, **kw), exp).wide()
        rows.append(market_stats(w))
        first = w.reset_index() if first is None else first
    return pd.DataFrame(rows).median().to_dict(), first          # medians: slumps make value (V) collapse


def exp1(T):
    """Margin requirements against bubble size."""
    rows, series = [], {}
    for mr in (1.0, 0.5, 0.2, 0.1):
        s, w = averaged(T, "1", SEEDS5, margin_requirement=mr)
        rows.append(dict(margin_requirement=mr, max_leverage=1 / mr, **s))
        series[f"margin {mr:.0%}"] = w
    return dict(table=pd.DataFrame(rows), series=series)


def exp2(T):
    """Who trades: the share of trend followers."""
    rows = []
    for tf in (0.0, 0.15, 0.3, 0.5):
        s, _ = averaged(T, "2", SEEDS5, trend_followers=tf, fundamentalists=min(0.3, 1 - tf))
        rows.append(dict(trend_followers=tf, **s))
    return dict(table=pd.DataFrame(rows))


def farm_stats(w: pd.DataFrame, skip: int = 50) -> dict:
    t = w.iloc[skip:]
    fi = t.farm_income
    out = dict(farm_income=fi.mean(), farm_income_sd=fi.std(), worst_tick=fi.min(), loss_share=float((fi < 0).mean()),
               farm_firms=t["firms/farm"].mean(), hedge_gains=t.hedge_pnl.sum(),
               bread=t.consumption_per_person.mean(), unemployment=t.unemployment_rate.mean())
    if "insurance_payouts" in t:
        out.update(payouts=t.insurance_payouts.sum(), premiums=t.insurance_premiums.sum(),
                   unpaid_claims=t.unpaid_claims.sum(), insurer_low=t.insurer_capital.min())
    return out


def exp3(T):
    """Do harvest futures (or insurance) smooth farmers' income?"""
    cases = {"no hedging": {}, "futures 50%": {"futures_hedge": 0.5}, "futures 100%": {"futures_hedge": 1.0},
             "insurance 80%": {"insurance_cover": 0.8}, "futures 100% + insurance 80%": {"futures_hedge": 1.0, "insurance_cover": 0.8}}
    rows = []
    for risk, shock in (("random harvests (sd 20%)", WEATHER), ("random demand (AR(1), sd 5%)", MOOD)):
        for label, kw in cases.items():
            ws = pd.DataFrame([farm_stats(T.run(with_params(base(), seed=s, **shock, **kw), "3").wide()) for s in SEEDS8])
            row = ws.median().to_dict()                        # robust to regime switches ...
            row.update(worst_tick_mean=ws.worst_tick.mean(), loss_share_mean=ws.loss_share.mean())   # ... tails too
            rows.append(dict(risk=risk, hedging=label, **row))
    return dict(table=pd.DataFrame(rows))


def exp4(T):
    """A correlated disaster: every farm claims at once."""
    disaster = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=150, duration=5, multiplier=0.4)]}
    rows, series = [], {}
    for label, kw in (("no insurance", {}), ("insurer with 30 ticks of premiums", {"insurance_cover": 1.0}),
                      ("insurer with 300 ticks of premiums", {"insurance_cover": 1.0, "insurer_capital": 300.0,
                                                               "capital_target": 600.0})):
        ws = []
        for seed in SEEDS5:
            w = T.run(with_params(base(), seed=seed, ticks=260, **disaster, **kw), "4").wide()
            win = w.loc[150:200]
            r = dict(farm_income_in_disaster=win.farm_income.loc[150:160].mean(), worst_tick=win.farm_income.min(),
                     farm_firms_after=w["firms/farm"].loc[200:].mean(), bread=win.consumption_per_person.mean())
            if "insurance_payouts" in w:
                r.update(claims_paid=win.insurance_payouts.sum(), claims_unpaid=win.unpaid_claims.sum(),
                         insurer_capital_low=win.insurer_capital.min())
            ws.append(r)
            if seed == SEEDS5[0]:
                series[label] = w.reset_index()
        rows.append(dict(case=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows), series=series)


def exp5(T):
    """Who owns the firms: founders, everyone, or shareholders?"""
    cases = {"founders (Stage 4-6), no stock market": {"stock_market.enabled": False},
             "everyone per head, no stock market": {"stock_market.enabled": False, "ownership": "public"},
             "listed shares, no trading": {"stock_market.start_tick": 100000},
             "listed shares, traded (margin 50%)": {}}
    rows = []
    for label, kw in cases.items():
        ws = []
        for seed in SEEDS5:
            w = T.run(with_params(base(), seed=seed, **kw), "5").wide()
            t = w.iloc[50:]
            ws.append(dict(bread=t.consumption_per_person.mean(), services=t.services_per_person.mean(),
                           unemployment=t.unemployment_rate.mean(), unemployment_sd=t.unemployment_rate.std(),
                           gini_income=t.gini_income.mean(), gini_wealth=w.gini_wealth.iloc[-1]))
        rows.append(dict(ownership=label, **pd.DataFrame(ws).mean().to_dict()))
    return dict(table=pd.DataFrame(rows))


def build_report(res, T, path):
    summary = {}
    body = ['<nav><a href="#e1">1 · margin and bubbles</a><a href="#e2">2 · trader mix</a><a href="#e3">3 · futures</a>'
            '<a href="#e4">4 · disaster</a><a href="#e5">5 · ownership</a></nav>']
    body.append("<p>Stage 6's economy gets a stock market. All firms' equity is one index of shares (split at the listing so "
                "that at fair value shares are half of households' wealth); holders receive "
                "every dividend. 30% of households trade on value (they buy below the fundamental value — average "
                "dividends divided by the policy rate plus a 5% premium — and sell above it), 30% follow trends (and trust "
                "a trend less the further the price is from value) and may borrow from the bank on margin; the rest hold, "
                "with a little noise trading. Trades need money, and the price moves with the order imbalance. Farms can "
                "also sell wheat forward (futures, settled in cash after 4 ticks) or insure their harvest with an insurer "
                "owned by households.</p>")
    sections = [("e1", "1 · Margin requirements and bubbles",
                 "Margin requirement = the share of a position the buyer must own (100%: no borrowing). Bubble size = the "
                 "95th percentile of price / fundamental value; medians of 5 seeds, 400 ticks (in a slump dividends and "
                 "so the fundamental value collapse, which would dominate a maximum or a mean).",
                 "1", "margin"),
                ("e2", "2 · Who trades", "Share of households following trends (value traders 30%, or what is left).",
                 "2", "trader_mix"),
                ("e3", "3 · Do harvest futures smooth farmers' income?",
                 "Farm income = the farms' profit including hedging gains, premiums and payouts. Two kinds of risk: random "
                 "weather (each harvest × lognormal noise, sd 20%) or random swings in household spending. Medians of 8 seeds: "
                 "the economy is path-dependent, and a small change can tip a run into a long slump that would dominate a mean.",
                 "3", "futures"),
                ("e4", "4 · A correlated disaster",
                 "The harvest falls 60% for 5 ticks (150–154): every insured farm claims at once.", "4", "disaster"),
                ("e5", "5 · Who owns the firms",
                 "Founder ownership (20 owner households per firm, Stages 4–6) versus dividends per head versus listed shares.",
                 "5", "ownership")]
    for anchor, title, text, key, sname in sections:
        t = res[key]["table"]
        summary[sname] = t.to_dict(orient="records")
        body.append(f'<h2 id="{anchor}">{title}</h2><p>{text}</p>')
        body.append(R.table_html(t))
        if key == "1":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                if k not in ("margin 100%", "margin 20%"):
                    continue
                f.add_trace(go.Scatter(x=w.tick, y=w.share_price, name=f"price, {k}", mode="lines", line=dict(width=2, color=R.SERIES[i])))
                f.add_trace(go.Scatter(x=w.tick, y=w.fundamental_value, name=f"value, {k}", mode="lines",
                                       line=dict(width=1, dash="dot", color=R.SERIES[i])))
            f.update_layout(title="Share price and fundamental value (one seed)", xaxis_title="tick", yaxis_title="money per share")
            body.append(R.fig_html(f))
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w.margin_debt, name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
            f.update_layout(title="Margin debt (one seed)", xaxis_title="tick", yaxis_title="money")
            body.append(R.fig_html(f))
        if key == "4":
            f = go.Figure()
            for i, (k, w) in enumerate(res[key]["series"].items()):
                f.add_trace(go.Scatter(x=w.tick, y=w.farm_income, name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
            f.update_xaxes(range=[130, 220])
            f.update_layout(title="Farm income around the disaster (one seed)", xaxis_title="tick", yaxis_title="money per tick")
            body.append(R.fig_html(f))
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s. New in Stage 7: shares held by households always add up to the shares listed; nobody "
                "holds negative shares or negative debt (no short selling); every unit of margin debt is a bank loan in the "
                "ledger; the insurer never pays out money it does not have; margin loans, interest, repayments and defaults "
                "are part of the bank's money-explained check.</p>")
    R.write(path, "Stage 7 · Financial markets and risk sharing", "citysim — shares, margin loans, bubbles, futures and "
            "insurance. Generated by <code>python -m citysim experiments stage7</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_7.html") -> dict:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T)}
    s = build_report(res, T, out)
    out.with_name("stage_7_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
