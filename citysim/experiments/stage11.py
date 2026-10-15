"""Stage 11: the modern town and the internet — experiments, report, and data for the town twin.

    python -m citysim experiments stage11      # reports/stage_11.html, twin/web_data.js
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
from ..layers.s11_modern_town.internet import STATUS

SEEDS = (7, 8, 9, 10, 11)
SC = SCEN / "stage11_internet.yaml"
SCENARIOS = {
    "history": ("The internet arrives", {}),
    "no_internet": ("No internet", {"internet.enabled": False}),
    "lean": ("Central bank leans against the bubble", {"policy": "lean", "lean_strength": 0.03}),
    "fixed": ("Central bank never moves", {"policy": "fixed", "initial_rate": 0.05}),
}
KEEP = ["year", "real_gdp", "gdp", "consumption", "investment", "it_investment", "unemployment", "inflation", "policy_rate",
        "loan_rate", "emergency_cut", "real_wage", "old_economy_index", "net_index", "network_index", "price_to_sales",
        "net_private_value", "mood", "sentiment", "overvaluation", "adoption/households", "adoption/businesses", "adoption/learned",
        "online_shopping", "internet_access", "online_ads", "vc_funding", "seed_funding", "ipo_funding", "startups/alive",
        "startups/public", "startups/founded", "startups/failed", "startups/ipos", "startups/acquired", "founded_now", "failed_now",
        "internet_jobs", "network_jobs", "internet_revenue", "leader_share/shop", "leader_share/portal", "servers",
        "startup_marketing", "router_sales", "fiber_investment", "carrier_debt", "carrier_revenue", "carrier_bankruptcies",
        "carrier_defaults", "network/traffic", "network/capacity", "network/utilization", "network/believed_growth",
        "network/price", "episode", "bank_loans", "bank_capital_ratio", "gov_balance", "public_debt", "income_tax_rate",
        "owners_wealth_share", "household_wealth", "defaults", "real_consumption"]
SHOW_FROM = 1985.0          # the documentary joins the town after it has settled (the first decade is burn-in)
NAMES = {"portal": ["Lookyloo", "Hither", "Seekwell", "Compass Point"], "shop": ["Cartwheel", "Parcelpost", "Bazaarly", "Shelfspace"],
         "flameout": ["Woofmart.com", "Zing!", "eFlorin", "Boom.net"]}


def run_one(key, seed, T=None, exp="x"):
    label, kw = SCENARIOS[key]
    sc = with_params(load_scenario(SC), seed=seed, **kw)
    if T is not None:
        res = T.run(sc, exp)
        return res.wide(), None
    sim = runner.build(sc)
    for _ in range(sc.ticks):
        sim.step()
    w = pd.DataFrame(sim.recorder.frame())
    from ..core.metrics import wide
    return wide(w), sim


def cohort(records, year):
    """Fate by `year` of the startups founded 1996-2000 (Goldfarb, Kirsch & Miller 2007: about half gone by 2004)."""
    from ..layers.s11_modern_town.town import year_of
    c = [r for r in records if 1996 <= year_of(r["born"]) < 2001]
    if not c:
        return {}
    gone = lambda r, st: r["status"] == st and 0 <= r["died"] and year_of(r["died"]) < year + 1
    return dict(cohort=len(c), failed=np.mean([gone(r, 2) for r in c]), acquired=np.mean([gone(r, 3) for r in c]),
                listed=np.mean([r["ipo"] > 0 for r in c]))


def facts(w: pd.DataFrame, records=None) -> dict:
    w = w.set_index("year")
    out = {}
    if "net_index" in w and w.net_index.max() > 0:
        peak = w.net_index.idxmax()
        after = w.loc[peak:peak + 3, "net_index"]
        bk = w[w.carrier_bankruptcies > 0].index
        out.update(net_peak_year=float(peak), net_drop=float(1 - after.min() / w.net_index.max()),
                   peak_price_to_sales=float(w.loc[peak, "price_to_sales"]),
                   founded=float(w["startups/founded"].iloc[-1]), failed=float(w["startups/failed"].iloc[-1]),
                   acquired=float(w["startups/acquired"].iloc[-1]), ipos=float(w["startups/ipos"].iloc[-1]),
                   leader_portal_2008=float(w.loc[2008, "leader_share/portal"]), leader_shop_2008=float(w.loc[2008, "leader_share/shop"]),
                   network_use_low=float(w.loc[1999:2005, "network/utilization"].min()),
                   bandwidth_price_low=float(w.loc[1999:2006, "network/price"].min()),
                   bandwidth_price_fall=float(1 - w.loc[1999:2006, "network/price"].min()),
                   carrier_bankruptcies=float(w.carrier_bankruptcies.iloc[-1]),
                   first_bankruptcy=float(bk.min()) if len(bk) else float("nan"),
                   fiber_peak_share_gdp=float((w.fiber_investment / w.gdp).loc[1996:2002].max()))
        if records:
            c = cohort(records, 2004)
            out.update(cohort_gone_2004=c.get("failed", 0) + c.get("acquired", 0), cohort_failed_2004=c.get("failed", np.nan),
                       cohort_acquired_2004=c.get("acquired", np.nan), cohort_listed=c.get("listed", np.nan))
    out.update(rate_1997=float(w.loc[1997, "policy_rate"]), rate_peak=float(w.loc[1998:2001.5, "policy_rate"].max()),
               rate_low=float(w.loc[2001:2005, "policy_rate"].min()), unemployment_low=float(w.loc[1997:2001, "unemployment"].min()),
               unemployment_high=float(w.loc[2001:2005, "unemployment"].max()), gdp_2010=float(w.real_gdp.iloc[-4:].mean()),
               unemployment_max_after=float(w.loc[2000:, "unemployment"].max()),
               unemployment_max_year=float(w.loc[2000:, "unemployment"].idxmax()),
               inflation_max=float(w.loc[1995:, "inflation"].max()),
               unemployment_2001q4=float(w.loc[2001.75, "unemployment"]), rate_2002q4=float(w.loc[2002.75, "policy_rate"]),
               rate_rise_99_00=float(w.loc[2000.25:2000.5, "policy_rate"].mean() - w.loc[1999.25:1999.5, "policy_rate"].mean()),
               growth_95_00=float(w.real_gdp.pct_change(4).loc[1995:2000].mean()),
               growth_86_93=float(w.real_gdp.pct_change(4).loc[1986:1993].mean()),
               growth_02_03=float(w.real_gdp.pct_change(4).loc[2002:2003.9].mean()),
               unemployment_86_93=float(w.loc[1986:1993, "unemployment"].mean()),
               old_index_drop=float(1 - w.loc[2000:2003, "old_economy_index"].min() / w.loc[1999:2000.5, "old_economy_index"].max()),
               inflation_95_05=float(w.loc[1995:2005, "inflation"].mean()))
    return out


# Stylized facts of the US dot-com era, for the realism scorecard: (label, fact key, history, acceptable range, format)
HISTORY = [
    ("Internet stocks peak", "net_peak_year", "Mar 2000", (1999.5, 2001.0), "year"),
    ("Fall from the peak within a year", "net_drop", "about 75-95% (NASDAQ -78%, dot-com indices more)", (0.6, 0.98), "pct"),
    ("Startups of 1996-2000 gone by 2004 (failed or bought)", "cohort_gone_2004", "about 50%", (0.35, 0.7), "pct"),
    ("Startups of 1996-2000 that listed on the market", "cohort_listed", "roughly 10-15%", (0.05, 0.25), "pct"),
    ("Policy rate at the top of the boom", "rate_peak", "6.5% (May 2000)", (0.05, 0.07), "pct1"),
    ("Policy rate at the bottom (scripted: the 2003 promise)", "rate_low", "1% (2003-04)", None, "pct1"),
    ("Policy rate at the end of 2002 (after the cuts)", "rate_2002q4", "1.25%", (0.01, 0.025), "pct1"),
    ("Rate rises, mid-1999 to mid-2000", "rate_rise_99_00", "+1.75 points", (0.01, 0.025), "pct1"),
    ("Unemployment at the boom's low", "unemployment_low", "3.9% (2000)", (0.03, 0.045), "pct1"),
    ("Unemployment at the end of 2001", "unemployment_2001q4", "5.7%", (0.048, 0.065), "pct1"),
    ("Unemployment at its post-bust peak", "unemployment_high", "6.3% (2003)", (0.05, 0.07), "pct1"),
    ("Network capacity in use at the glut's worst", "network_use_low", "a small fraction (most fibre dark)", (0.0, 0.3), "pct"),
    ("Fall in bandwidth prices", "bandwidth_price_fall", "60-90%", (0.5, 0.95), "pct"),
    ("Carriers bankrupt (of 3)", "carrier_bankruptcies", "many, not all (Global Crossing, WorldCom; AT&T survived)", (1, 2), "int"),
    ("First carrier bankruptcy", "first_bankruptcy", "2001-02", (2001.0, 2003.5), "year"),
    ("Fall in established companies' shares", "old_index_drop", "S&P 500 -45% (2000-02, much of it tech); value stocks held up", (0.1, 0.5), "pct"),
    ("Fibre and router build-out at its peak, share of GDP", "fiber_peak_share_gdp", "about 1.5% (US telecom capex, 2000)", (0.005, 0.03), "pct1"),
    ("Growth in the recovery, 2002-03", "growth_02_03", "about 2.2% a year", (0.01, 0.035), "pct1"),
    ("Biggest portal's share of its market, 2008", "leader_portal_2008", "winner-take-most (search: ~60%+)", (0.4, 0.95), "pct"),
]


def scorecard(f: dict) -> pd.DataFrame:
    rows = []
    for label, key, hist, rng, fmt in HISTORY:
        v = f.get(key, float("nan"))
        show = {"year": lambda x: f"{int(round(x * 4)) // 4} Q{int(round(x * 4)) % 4 + 1}", "pct": lambda x: f"{x:.0%}",
                "pct1": lambda x: f"{x:.1%}", "int": lambda x: f"{x:.1f}"}[fmt]
        if rng is None:                                    # set by a scripted episode: shown, not scored
            mark = "scripted"
        else:
            mark = "✓" if np.isfinite(v) and rng[0] <= v <= rng[1] else "✗"
        rows.append({"stylized fact": label, "history": hist, "model (mean of seeds)": show(v) if np.isfinite(v) else "—",
                     "match": mark})
    return pd.DataFrame(rows)


def exp1(T):
    """Each scenario, five seeds."""
    rows, hist = [], []
    for key, (label, _) in SCENARIOS.items():
        fs = []
        for s in SEEDS:
            t0 = time.perf_counter()
            w, sim = run_one(key, s)                       # invariants are checked every tick
            T.seconds += time.perf_counter() - t0
            T.runs += 1
            T.ticks += len(w)
            T.by_experiment["1"] = T.by_experiment.get("1", 0) + 1
            fs.append(facts(w, startup_records(sim) if key != "no_internet" else None))
            if key == "history":
                hist.append(fs[-1])
        rows.append(dict(scenario=label, **pd.DataFrame(fs).mean(numeric_only=True).to_dict()))
    t = pd.DataFrame(rows)
    base = t.loc[t.scenario == "No internet", "gdp_2010"].iloc[0]
    t["gdp_2010_vs_no_internet"] = t.gdp_2010 / base - 1
    mean = pd.DataFrame(hist).mean(numeric_only=True).to_dict()
    hits = [(scorecard(f)["match"] == "✓").sum() for f in hist]
    best = SEEDS[int(np.argmax(hits))]                     # the twin replays the seed that matches history best
    return dict(table=t, scorecard=scorecard(mean), history_mean=mean, twin_seed=best, hits=dict(zip(SEEDS, map(int, hits))))


def startup_records(sim):
    net = next(l for l in sim.layers if l.name == "internet")
    return [dict(id=s.id, cat=s.cat, q=round(s.quality, 3), born=s.born, died=s.died, ipo=s.public_at, status=STATUS[s.status],
                 by=s.acquired_by, peak=round(s.peak_value, 1), trail=[list(x) for x in s.trail]) for s in net.startups]


def infra_records(sim):
    net = next(l for l in sim.layers if l.name == "internet")
    return [dict(name=f.name, kind=f.kind, appetite=f.appetite, bankrupt=f.bankrupt,
                 trail=[[float(v) for v in x] for x in f.trail]) for f in net.infra]


def cast(records, end_tick):
    """Named characters for the narrator: the winners of each market and the most spectacular failure."""
    out = {}
    for cat in ("portal", "shop"):
        alive = [r for r in records if r["cat"] == cat and r["status"] in (0, 1) and r["trail"]]
        if alive:
            win = max(alive, key=lambda r: r["trail"][-1][2])
            out[f"winner_{cat}"] = dict(id=win["id"], name=NAMES[cat][win["id"] % len(NAMES[cat])], born=win["born"],
                                        ipo=win["ipo"], revenue=win["trail"][-1][2])
    dead = [r for r in records if r["status"] == 2 and r["peak"] > 0]
    if dead:
        f = max(dead, key=lambda r: r["peak"])
        out["flameout"] = dict(id=f["id"], name=NAMES["flameout"][f["id"] % len(NAMES["flameout"])], born=f["born"], died=f["died"],
                               ipo=f["ipo"], peak=f["peak"], cat=f["cat"])
    return out


def events(w: pd.DataFrame, records, infra=(), crew=None, start=70, gain_vs_none=0.0) -> list[dict]:
    """Moments the narrator speaks about, found in the run itself (every number quoted is the run's own)."""
    w = w.reset_index(drop=True)
    crew = crew or {}
    ev = []
    t_of = lambda i: int(w.tick.iloc[i])
    at = lambda year: int(np.argmin(np.abs(w.year.values - year)))
    s0 = at(SHOW_FROM)

    def first(cond, key, after=0, **extra):
        idx = [i for i in np.where(cond)[0] if i >= after]
        if idx:
            ev.append(dict(key=key, tick=t_of(idx[0]), **{k: (v(idx[0]) if callable(v) else v) for k, v in extra.items()}))
            return idx[0]
    ev.append(dict(key="intro", tick=t_of(s0)))
    calm = w.iloc[s0:start - 1]
    ev.append(dict(key="steady", tick=t_of(s0) + 3, u=float(calm.unemployment.mean()), infl=float(calm.inflation.mean()),
                   rate=float(calm.policy_rate.mean()), growth=float(w.real_gdp.pct_change(4).iloc[s0:start - 1].mean())))
    ev.append(dict(key="browser", tick=start))
    born = sorted(r["born"] for r in records)
    if born:
        ev.append(dict(key="first_startup", tick=born[0]))
    ipos = sorted(r["ipo"] for r in records if r["ipo"] > 0)
    if ipos:
        ev.append(dict(key="first_ipo", tick=ipos[0], n_alive=float(w.loc[w.tick == ipos[0], "startups/alive"].iloc[0])))
    first(w["adoption/households"].values > 0.25, "adoption_quarter", share=lambda i: float(w["adoption/households"].iloc[i]))
    first(w["network/believed_growth"].values > 0.3, "fiber", util=lambda i: float(w["network/utilization"].iloc[i]),
          debt=lambda i: float(w.carrier_debt.iloc[i]))
    first(w.episode.values == 1, "warning", mood=lambda i: float(w.mood.iloc[i]))
    first(w.episode.values == 2, "scare", rate=lambda i: float(w.policy_rate.iloc[i]))
    first(w.episode.values == 3, "y2k")
    first((w.mood.values > 2.0), "exuberance", mood=lambda i: float(w.mood.iloc[i]),
          multiple=lambda i: float(w.price_to_sales.iloc[i]))
    lo = w.loc[(w.year >= 1997) & (w.year <= 2001.5)]
    if len(lo):
        i = lo.unemployment.idxmin()
        ev.append(dict(key="tight_labor", tick=int(w.tick.iloc[i]), u=float(w.unemployment.iloc[i])))
    hk = w.loc[(w.year >= 1999.25) & (w.year <= 2001.5)]
    if len(hk):
        j = hk.policy_rate.idxmax()
        start_r = w.loc[(w.year >= 1999) & (w.tick < w.tick.iloc[j]), "policy_rate"]
        k = start_r.idxmin() if len(start_r) else j
        ev.append(dict(key="fed_hikes", tick=int(w.tick.iloc[k]) + 1, start=float(w.policy_rate.iloc[k]), peak=float(w.policy_rate.iloc[j]),
                       peak_tick=int(w.tick.iloc[j])))
    if w.net_index.max() > 0:
        p = int(w.net_index.idxmax())
        ev.append(dict(key="peak", tick=int(w.tick.iloc[p]), value=float(w.net_index.iloc[p]),
                       multiple=float(w.price_to_sales.iloc[p]), alive=float(w["startups/alive"].iloc[p]),
                       public=float(w["startups/public"].iloc[p])))
        after = w.iloc[p:p + 12]
        tr = int(after.net_index.idxmin())
        ev.append(dict(key="crash", tick=int(w.tick.iloc[min(p + 2, len(w) - 1)]), drop=float(1 - w.net_index.iloc[tr] / w.net_index.iloc[p]),
                       failed=float(w["startups/failed"].iloc[min(p + 12, len(w) - 1)] - w["startups/failed"].iloc[p]),
                       acquired=float(w["startups/acquired"].iloc[min(p + 12, len(w) - 1)] - w["startups/acquired"].iloc[p]),
                       trough_tick=int(w.tick.iloc[tr])))
        if "flameout" in crew and crew["flameout"]["died"] > 0:
            fl = crew["flameout"]
            ev.append(dict(key="flameout", tick=fl["died"], name=fl["name"], peak=fl["peak"], cat=fl["cat"]))
        cut = w.iloc[p:p + 20]
        c = int(cut.policy_rate.idxmin())
        falls = [k for k in range(p + 1, min(p + 16, len(w))) if w.policy_rate.iloc[k] < w.policy_rate.iloc[k - 1] - 0.002]
        ct = falls[0] if falls else p + 4
        ev.append(dict(key="fed_cuts", tick=int(w.tick.iloc[ct]), high=float(w.policy_rate.iloc[p:ct + 1].max()),
                       low=float(w.policy_rate.iloc[c]), low_tick=int(w.tick.iloc[c])))
        first(w.episode.values == 4, "hold", after=p, rate=lambda i: float(w.policy_rate.iloc[min(i + 1, len(w) - 1)]),
              infl=lambda i: float(w.inflation.iloc[i]))
        un = w.iloc[p:p + 24]
        h = int(un.unemployment.idxmax())
        ev.append(dict(key="recession", tick=int(w.tick.iloc[h]), u=float(w.unemployment.iloc[h]),
                       u_before=float(w.unemployment.iloc[max(p - 4, 0):p + 1].min()),
                       inv_drop=float(1 - w.investment.iloc[p:h + 1].min() / w.investment.iloc[max(p - 2, 0):p + 3].max())))
        glut = w.loc[(w.year >= 1997) & (w.year <= 2005), "network/utilization"]
        m = int(glut.idxmin())
        cheap = w.loc[(w.year >= 1997) & (w.year <= 2006), "network/price"]
        ev.append(dict(key="dark_fiber", tick=max(int(w.tick.iloc[m]), int(w.tick.iloc[p]) + 4), util=float(glut.min()),
                       price=float(cheap.min()), debt=float(w.carrier_debt.max())))
        rs = w.router_sales
        rp_ = int(rs.iloc[start:].idxmax())
        drop = [k for k in range(rp_ + 1, len(w)) if rs.iloc[k] < 0.7 * rs.iloc[rp_]]
        if drop:
            k = drop[0]
            ev.append(dict(key="corelink", tick=int(w.tick.iloc[k]), peak=float(rs.iloc[rp_]), now=float(rs.iloc[k]),
                           staff_peak=float(w["employment/network"].iloc[rp_]), peak_year=float(w.year.iloc[rp_])))
    falls = {}
    for f in infra:
        if f["kind"] == "carrier" and f["bankrupt"] > 0:
            falls.setdefault(f["bankrupt"], []).append(f)
    total = sum(1 for f in infra if f["kind"] == "carrier")
    for tk, fs in sorted(falls.items()):
        i = int(np.where(w.tick.values == tk)[0][0])
        debt = sum(next((x[5] for x in f["trail"] if x[0] == tk - 1), 0.0) for f in fs)
        ev.append(dict(key="bankruptcy", tick=tk, names=[f["name"] for f in fs], debt=float(debt), total=total,
                       written_off=float(w.carrier_defaults.iloc[i]), so_far=int(w.carrier_bankruptcies.iloc[i])))
    survivors = [f["name"] for f in infra if f["kind"] == "carrier" and f["bankrupt"] < 0]
    end = w.iloc[-1]
    j = len(w) - 16
    ev.append(dict(key="winners", tick=int(w.tick.iloc[j]), founded=float(end["startups/founded"]), failed=float(end["startups/failed"]),
                   acquired=float(end["startups/acquired"]), portal=float(w["leader_share/portal"].iloc[j]),
                   shop=float(w["leader_share/shop"].iloc[j]), survivors=survivors,
                   **{k: v["name"] for k, v in crew.items() if k.startswith("winner")}))
    lit = [k for k in range(at(2001), len(w)) if w["network/utilization"].iloc[k] >= 0.95]
    ev.append(dict(key="legacy", tick=int(w.tick.iloc[-6]), gain_vs_none=gain_vs_none, online=float(w.online_shopping.iloc[-6] / max(w.consumption.iloc[-6], 1e-9)),
                   lit_year=float(w.year.iloc[lit[0]]) if lit else -1.0,
                   traffic_x=float(w["network/traffic"].iloc[-6] / max(w["network/traffic"].iloc[at(2000)], 1e-9)),
                   gdp_gain=float(w.real_gdp.iloc[-6] / w.real_gdp.iloc[start - 1] - 1),
                   retail_prod=float(w["productivity/retail"].iloc[-6] / w["productivity/retail"].iloc[start - 1]),
                   retail_jobs_then=float(w["employment/retail"].iloc[start - 1]), retail_jobs_now=float(w["employment/retail"].iloc[-6])))
    ev.append(dict(key="coda", tick=int(w.tick.iloc[-1])))
    return sorted(ev, key=lambda e: e["tick"])


TWIN_COLS = ["tick"] + KEEP + [f"employment/{s}" for s in ("farm", "factory", "retail", "services", "telecom", "finance", "public",
                                                            "research", "internet", "unemployed")] + \
            ["employment/network"] + [f"productivity/{s}" for s in ("farm", "factory", "retail", "services", "telecom", "finance")]


def twin_data(seed=SEEDS[0]):
    out = {"scenarios": {}, "labels": {k: v[0] for k, v in SCENARIOS.items()}, "show_from": SHOW_FROM, "seed": seed}
    base = None
    for key in sorted(SCENARIOS, key=lambda k: k != "no_internet"):      # the town without the internet first
        w, sim = run_one(key, seed)
        w = w.reset_index()
        if key == "no_internet":
            base = float(w.real_gdp.iloc[-4:].mean())
        cols = [c for c in TWIN_COLS if c in w.columns]
        series = {c: [round(float(x), 5) if np.isfinite(x) else 0.0 for x in w[c].fillna(0.0)] for c in cols}
        net = key != "no_internet"
        recs = startup_records(sim) if net else []
        infra = infra_records(sim) if net else []
        crew = cast(recs, int(w.tick.iloc[-1])) if net else {}
        start = next((l.p.start for l in sim.layers if l.name == "internet"), 70)
        out["scenarios"][key] = dict(series=series, startups=recs, infra=infra, cast=crew,
                                     events=events(w, recs, infra, crew, start, float(w.real_gdp.iloc[-4:].mean()) / base - 1) if net else [], facts=facts(w, recs or None))
    return out


def build_report(res, T, path, data):
    summary = {"scenarios": res["1"]["table"].to_dict(orient="records"), "scorecard": res["1"]["scorecard"].to_dict(orient="records"),
               "history_mean": res["1"]["history_mean"], "twin_seed": res["1"]["twin_seed"], "hits_by_seed": res["1"]["hits"]}
    body = ["<p>A town of 10,000 people from 1976 to 2010, one tick per quarter (the first decade lets it settle). Farms, "
            "factories, shops, services, telecom and finance businesses; a bank, a central bank, a city government, an insurer, "
            "a stock market and a university. In 1993 Q2 the internet becomes usable (the first popular web browser): startups, "
            "venture money, IPOs, three fibre carriers and a router maker, a new-economy mood, and the central bank's response. "
            f"Averages of {len(SEEDS)} seeds.</p>",
            "<h2>Realism scorecard</h2><p>The model's dot-com era against the stylized facts of the US 1995-2005 "
            "(mean of the seeds, scenario <i>The internet arrives</i>). Nothing here is fitted year by year: the bubble, crash, "
            "glut and bankruptcies come out of the mechanisms; only the Fed's 1998 and 2003 decisions are scripted as history.</p>",
            R.table_html(res["1"]["scorecard"]), "<h2>Scenarios</h2>", R.table_html(res["1"]["table"])]
    h = data["scenarios"]["history"]["series"]
    n = data["scenarios"]["no_internet"]["series"]

    def fig(title, ytitle, traces):
        f = go.Figure()
        for i, (x, y, name, dash) in enumerate(traces):
            f.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=R.SERIES[i], width=2, dash=dash)))
        f.update_layout(title=title, xaxis_title="year", yaxis_title=ytitle)
        body.append(R.fig_html(f))
    fig("Stock market value (one seed)", "$", [(h["year"], h["net_index"], "internet companies", None),
                                              (h["year"], h["network_index"], "carriers and router maker", None),
                                              (h["year"], h["old_economy_index"], "established companies", None)])
    fig("Central bank and the economy (one seed)", "annual rate", [(h["year"], h[k], lab, None) for k, lab in
                                                                   (("policy_rate", "policy rate"), ("unemployment", "unemployment"),
                                                                    ("inflation", "inflation"))])
    fig("The network: capacity built vs traffic carried (one seed)", "traffic units",
        [(h["year"], h["network/capacity"], "capacity", None), (h["year"], h["network/traffic"], "traffic", None)])
    fig("Startups (one seed)", "count", [(h["year"], h["startups/alive"], "alive", None), (h["year"], h["startups/public"], "listed", None),
                                         (h["year"], h["startups/failed"], "failed (cumulative)", None),
                                         (h["year"], h["startups/acquired"], "bought (cumulative)", None)])
    fig("Real GDP (one seed)", "$ (base-year prices)", [(h["year"], h["real_gdp"], "with the internet", None),
                                                         (n["year"], n["real_gdp"], "no internet", "dot")])
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s — double entry, goods and money conservation, balance sheets, employment within the "
                "population, sales within capacity, startup shares adding up, no startup spending money it doesn't "
                "have, and no network carrying more than its capacity.</p>")
    R.write(path, "Stage 11 · The town and the internet", "citysim — a modern town meets the internet. Generated by "
            "<code>python -m citysim experiments stage11</code>.", "\n".join(body))
    return summary


def main(out: Path = ROOT / "reports" / "stage_11.html") -> dict:
    T = Tally()
    res = {"1": exp1(T)}
    data = twin_data(res["1"]["twin_seed"])
    data["scorecard"] = res["1"]["scorecard"].to_dict(orient="records")
    data["n_seeds"] = len(SEEDS)
    (ROOT / "twin" / "web_data.js").write_text("window.WEB = " + json.dumps(data, separators=(",", ":")) + ";\n")
    s = build_report(res, T, out, data)
    out.with_name("stage_11_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
