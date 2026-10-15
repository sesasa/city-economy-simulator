"""Stage 1 experiments (SPEC 'Stage 1 > Experiments') and the stage_1.html report.

Every simulation here goes through `runner.run`, so invariants are checked at
every tick of every sweep point; a violation raises and aborts the report.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import report as R
from ..config import Scenario, load_scenario, with_params
from ..core import runner
from ..layers.s1_real_economy.rules import analytical_a_min, analytical_n_min

ROOT = Path(__file__).resolve().parents[2]
SCEN = ROOT / "scenarios"


@dataclass
class Tally:
    runs: int = 0
    ticks: int = 0
    seconds: float = 0.0
    by_experiment: dict[str, int] = field(default_factory=dict)

    def run(self, sc: Scenario, exp: str) -> runner.RunResult:
        t = time.perf_counter()
        res = runner.run(sc)               # invariants checked every tick
        self.seconds += time.perf_counter() - t
        self.runs += 1
        self.ticks += sc.ticks
        self.by_experiment[exp] = self.by_experiment.get(exp, 0) + 1
        return res


def base() -> Scenario:
    return load_scenario(SCEN / "stage1_base.yaml")


def params_of(sc: Scenario) -> dict:
    p = sc.layers["real_economy"]
    return dict(c=p["consumption_per_person"], e=p["conversion_rate"], F_p=p["factory_fixed_crew"],
                F_m=p["maintenance_fixed_crew"], v_p=p["factory_variable_labor"], v_d=p["distribution_labor"])


def viable(res: runner.RunResult) -> bool:
    """No labor shortfall at any tick and everyone fully fed."""
    w = res.wide()
    return bool((w["labor_shortfall"] <= 1e-9).all() and (w["consumption_per_person"] >= 1 - 1e-9).all())


def bisect_param(T: Tally, exp: str, sc: Scenario, key: str, lo: float, hi: float, tol: float = 1e-4) -> float:
    """Smallest value of `key` in [lo, hi] for which the city is viable (viability is monotone)."""
    for _ in range(60):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        if viable(T.run(with_params(sc, **{key: mid}), exp)):
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
def exp1(T: Tally) -> dict:
    """Sweep N0 50..400 with fixed population."""
    sc = with_params(base(), ticks=600)
    an = analytical_n_min(a=5, **params_of(sc))
    rows = []
    for n0 in list(np.arange(50, 401, 2.0)) + [an]:
        w = T.run(with_params(sc, N0=float(n0)), "1").wide()
        rows.append(dict(N0=float(n0), viable=bool((w.labor_shortfall <= 1e-9).all()),
                         surplus_share_t1=w.surplus_share.iloc[0], surplus_share_end=w.surplus_share.iloc[-1],
                         cons_end=w.consumption_per_person.iloc[-1], factory_end=w["capital/factory"].iloc[-1],
                         potential_t1=w.potential_output.iloc[0], potential_end=w.potential_output.iloc[-1]))
    sweep = pd.DataFrame(rows).sort_values("N0").reset_index(drop=True)
    sim = bisect_param(T, "1", with_params(sc, ticks=20), "N0", 50, 400, tol=1e-6)
    # Outcome-based threshold: smallest N0 that keeps everyone fully fed for T ticks.
    # Unlike the labor-shortfall test above (an accounting identity), this lets the
    # factory's spare capacity buy time, so it depends on the horizon.
    fed = {}
    for horizon in (20, 100, 300, 1000):
        lo, hi = 50.0, 400.0
        while hi - lo > 1e-3:
            mid = 0.5 * (lo + hi)
            w = T.run(with_params(sc, N0=mid, ticks=horizon), "1").wide()
            lo, hi = (lo, mid) if (w.consumption_per_person >= 1 - 1e-9).all() else (mid, hi)
        fed[horizon] = hi
    series = {}
    for n0 in (100, 120, 130, an, 150, 300):
        series[n0] = T.run(with_params(sc, N0=n0), "1").wide().reset_index()
    demo, survival = {}, {}
    dsc = with_params(sc, ticks=400, **{"demography.enabled": True})
    for rule, n0s in (("priority", (70, 90, 125)), ("max_feasible", (125, 133, 134))):
        for n0 in n0s:
            demo[(rule, n0)] = T.run(with_params(dsc, N0=n0, shortage_rule=rule), "1").wide().reset_index()
        lo, hi = 50.0, 400.0                       # smallest N0 whose population survives 400 ticks
        while hi - lo > 1e-3:
            mid = 0.5 * (lo + hi)
            w = T.run(with_params(dsc, N0=mid, shortage_rule=rule), "1").wide()
            lo, hi = (lo, mid) if w.population.iloc[-1] > mid else (mid, hi)
        survival[rule] = hi
    return dict(analytical=an, simulated=sim, sweep=sweep, series=series, demo=demo, survival=survival, fed=fed)


def exp2(T: Tally) -> dict:
    """Sweep a from 1 to 10 at several population sizes; find minimum farm productivity."""
    sc = with_params(base(), ticks=30)
    pops = [150, 200, 300, 400, 1000]
    p = params_of(sc)
    rows = []
    for n in pops:
        sim = bisect_param(T, "2", with_params(sc, N0=n), "a", 1.0, 10.0, tol=1e-6)
        rows.append(dict(N0=n, a_min_analytical=analytical_a_min(n, **p), a_min_simulated=sim))
    table = pd.DataFrame(rows)
    grid = []
    for n in range(50, 401, 10):
        for a in np.arange(1.0, 10.01, 0.25):
            res = T.run(with_params(sc, N0=float(n), a=float(a), ticks=10), "2")
            grid.append(dict(N0=n, a=float(a), viable=viable(res)))
    grid = pd.DataFrame(grid)
    a_sweep = []
    for n in pops:
        for a in np.arange(1.0, 10.01, 0.5):
            w = T.run(with_params(sc, N0=n, a=float(a)), "2").wide()
            a_sweep.append(dict(N0=n, a=float(a), surplus_share=w.surplus_share.iloc[0],
                                shortfall=w.labor_shortfall.iloc[0]))
    return dict(table=table, grid=grid, a_sweep=pd.DataFrame(a_sweep))


def exp3(T: Tally) -> dict:
    """Sweep s from 0 to 0.8: growth of potential output (fixed population) and of population."""
    sc = with_params(base(), ticks=300)
    rows, series = [], {}
    for s in np.round(np.arange(0.0, 0.801, 0.05), 2):
        for n0 in (150, 300):
            w = T.run(with_params(sc, s=float(s), N0=n0), "3").wide()
            y0, yT = w.potential_output.iloc[0], w.potential_output.iloc[-1]
            # growth measured from the tick-0 baseline: potential(t1) = potential(t0) * (1 + g1)
            y_start = y0 / (1 + w.growth_rate.iloc[0])
            rows.append(dict(s=float(s), N0=n0, growth_per_tick=(yT / y_start) ** (1 / len(w)) - 1,
                             growth_first50=(w.potential_output.iloc[49] / y_start) ** (1 / 50) - 1,
                             first_idle_tick=int(w.index[(w.investment_labor <= 0) & (w.surplus_share > 0)].min())
                             if s > 0 and ((w.investment_labor <= 0) & (w.surplus_share > 0)).any() else None,
                             potential_start=y_start, potential_end=yT,
                             max_abs_growth=float(w.growth_rate.abs().max()),
                             surplus_start=w.surplus_share.iloc[0], surplus_end=w.surplus_share.iloc[-1]))
            if n0 == 150 and s in (0.0, 0.1, 0.3, 0.8):
                series[float(s)] = w.reset_index()
    demo_rows, demo_series = [], {}
    for s in np.round(np.arange(0.0, 0.801, 0.1), 2):
        w = T.run(with_params(sc, s=float(s), ticks=400, **{"demography.enabled": True}), "3").wide()
        tail = w.population.iloc[-100:]
        demo_rows.append(dict(s=float(s), pop_end=w.population.iloc[-1],
                              growth_last100=(tail.iloc[-1] / tail.iloc[0]) ** (1 / 99) - 1,
                              factories_end=w.factories.iloc[-1]))
        if s in (0.0, 0.3):
            demo_series[float(s)] = w.reset_index()
    return dict(table=pd.DataFrame(rows), series=series, demo=pd.DataFrame(demo_rows), demo_series=demo_series)


def exp4(T: Tally) -> dict:
    """Long run with demography and land limits on."""
    sc = load_scenario(SCEN / "stage1_long_run.yaml")
    variants = {"default (land limits + land improvements)": {},
                "land limits, no land improvements": {"land_improvement_elasticity": 0.0},
                "no land limits": {"land_limits.enabled": False, "ticks": 1200}}
    out, rows = {}, []
    for label, kw in variants.items():
        w = T.run(with_params(sc, **kw), "4").wide().reset_index()
        out[label] = w
        g = w.population_growth.rolling(25, min_periods=1).mean()
        below = w.tick[g < 0.001]
        tail = w.population.iloc[-200:]
        rows.append(dict(variant=label, ticks=int(w.tick.iloc[-1]), pop_end=w.population.iloc[-1],
                         pop_peak=w.population.max(), tick_growth_below_0_1pct=int(below.iloc[0]) if len(below) else None,
                         drift_last200=tail.iloc[-1] / tail.iloc[0] - 1,
                         farm_share_end=w["labor_share/farming"].iloc[-1],
                         eff_a_end=w.effective_farm_productivity.iloc[-1], eff_a_peak=w.effective_farm_productivity.max(),
                         leisure_peak=w.leisure_per_person.max(), leisure_end=w.leisure_per_person.iloc[-1]))
    return dict(series=out, table=pd.DataFrame(rows))


def _shock_metrics(w: pd.DataFrame, cf: pd.DataFrame, start: int, dur: int, storage: bool,
                   demography: bool) -> dict:
    pre_n = w.population.loc[start - 1]
    after = w.loc[start:]
    # bread missing from people actually alive (not people who were never born)
    lost = ((1 - w.consumption_per_person).clip(lower=0) * w.population).loc[start:].sum()
    cons_ok = after.consumption_per_person >= 1 - 1e-9
    ok_idx = after.index[cons_ok & (after.index >= start)]
    # consumption recovered: first tick after which consumption stays full
    bad = after.index[~cons_ok]
    cons_rec = (int(bad.max()) + 1 - start) if len(bad) else 0
    full = cons_ok.copy()
    if storage:
        target = w.population  # storage_target_ticks = 1 tick of consumption
        full &= after.storage_stock >= 0.99 * target.loc[start:]
    bad_full = after.index[~full]
    full_rec = (int(bad_full.max()) + 1 - start) if len(bad_full) else 0
    pop_back = after.index[after.population >= pre_n * (1 - 1e-9)]
    pop_back = [t for t in pop_back if t >= start + dur]
    if after.population.min() >= pre_n * (1 - 1e-9):
        pop_back = [start]  # never fell below the pre-shock level
    return dict(min_consumption_per_person=after.consumption_per_person.min(),
                consumption_lost=lost,
                ticks_to_full_consumption=cons_rec,
                ticks_to_full_recovery=full_rec,
                investment_lost=(cf.investment_labor - w.investment_labor).loc[start:start + 19].sum(),
                steady_investment=cf.investment_labor.loc[start - 1],
                pop_loss=1 - after.population.min() / pre_n,
                ticks_to_regain_population=((int(pop_back[0]) - start) if pop_back else None) if demography else None,
                pop_gap_vs_no_shock_end=w.population.iloc[-1] / cf.population.iloc[-1] - 1)


def exp5(T: Tally) -> dict:
    """30% harvest cut for 3 ticks, with and without storage (and with/without demography)."""
    sc = load_scenario(SCEN / "stage1_shock_storage.yaml")
    shock = sc.layers["harvest_shocks"]["shocks"][0]
    start, dur = shock["start"], shock["duration"]
    rows, series = [], {}
    for demo in (False, True):
        for storage in (False, True):
            kw = {"storage.enabled": storage, "demography.enabled": demo}
            w = T.run(with_params(sc, **kw), "5").wide()
            cf = T.run(with_params(sc, **kw, **{"harvest_shocks.enabled": False}), "5").wide()
            label = f"{'storage' if storage else 'no storage'}, {'demography' if demo else 'fixed population'}"
            rows.append(dict(case=label, **_shock_metrics(w, cf, start, dur, storage, demo)))
            series[label] = w.reset_index()
    dur_rows = []
    for d in range(1, 9):
        for storage in (False, True):
            s2 = with_params(sc, **{"storage.enabled": storage, "harvest_shocks.shocks":
                                    [dict(start=start, duration=d, multiplier=0.7)]})
            w = T.run(s2, "5").wide()
            short = (1 - w.consumption_per_person).clip(lower=0) * w.population
            dur_rows.append(dict(duration=d, storage=storage, consumption_lost=short.sum(),
                                 min_cons=w.consumption_per_person.min()))
    return dict(table=pd.DataFrame(rows), series=series, durations=pd.DataFrame(dur_rows), start=start, dur=dur)


# ---------------------------------------------------------------------------
def run_all() -> tuple[dict, Tally]:
    T = Tally()
    res = {"1": exp1(T), "2": exp2(T), "3": exp3(T), "4": exp4(T), "5": exp5(T)}
    return res, T


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}"


def build_report(res: dict, T: Tally, path: Path) -> dict:
    e1, e2, e3, e4, e5 = (res[k] for k in "12345")
    body = ['<nav><a href="#e1">1 · N_min</a><a href="#e2">2 · farm productivity</a>'
            '<a href="#e3">3 · investment share</a><a href="#e4">4 · long run</a><a href="#e5">5 · harvest shock</a>'
            '<a href="#inv">Invariants</a></nav>']
    summary: dict = {}

    # ---- Experiment 1 ------------------------------------------------------------
    an, sim = e1["analytical"], e1["simulated"]
    summary["n_min"] = dict(analytical=an, simulated=sim, error=sim - an, fed_for_horizon=e1["fed"])
    sw = e1["sweep"]
    body.append('<h2 id="e1">1 · Minimum viable population</h2>')
    body.append("<p>Fixed population, investment share s = 0.3, 600 ticks per point. "
                "A city is <em>viable</em> when its labor covers farming, processing, distribution and "
                "maintenance in full. Below N<sub>min</sub>, the planner's priority order starves maintenance "
                "first; the factory decays until its capacity is low enough that the freed labor can maintain "
                "it, which is when consumption per person settles at (N − 80)/(0.4 N).</p>")
    body.append("<p><strong>What the match does and doesn't show.</strong> The simulated N<sub>min</sub> below is the "
                "smallest town with no labor shortfall in the planner's books. That is the formula rearranged, so the "
                "near-perfect match only confirms the planner implements the accounting correctly. The more telling "
                "number is the outcome-based threshold: the smallest town that keeps everyone fully fed for T ticks. "
                "Because the factory starts with spare capacity, a town below N<sub>min</sub> can eat well for decades "
                "while its capital decays, so this threshold depends on the horizon and only approaches N<sub>min</sub> "
                "in the long run.</p>")
    fedtab = pd.DataFrame([dict(horizon_ticks=k, smallest_town_fully_fed=v) for k, v in e1["fed"].items()])
    body.append(R.table_html(fedtab))
    body.append(R.tiles([("analytical N_min = 80 / 0.6", f"{an:.4f}"), ("no-shortfall threshold (bisection)", f"{sim:.4f}"),
                         ("difference (people)", f"{sim - an:+.2e}"),
                         ("surplus share at N0 = 150 / 300", f"{sw.loc[sw.N0 == 150, 'surplus_share_t1'].iloc[0]:.3f} / "
                                                             f"{sw.loc[sw.N0 == 300, 'surplus_share_t1'].iloc[0]:.3f}")]))
    f = go.Figure()
    f.add_trace(go.Scatter(x=sw.N0, y=sw.cons_end, name="consumption per person, tick 600", mode="lines",
                           line=dict(width=2, color=R.SERIES[0])))
    f.add_trace(go.Scatter(x=sw.N0, y=sw.surplus_share_t1, name="surplus share of labor, tick 1", mode="lines",
                           line=dict(width=2, color=R.SERIES[1])))
    f.add_trace(go.Scatter(x=sw.N0, y=sw.factory_end, name="factory condition, tick 600", mode="lines",
                           line=dict(width=2, color=R.SERIES[2])))
    ns = np.linspace(80, 400, 321)
    f.add_trace(go.Scatter(x=ns, y=np.minimum(1, np.maximum(0, (ns - 80) / (0.4 * ns))), name="theory: (N − 80)/(0.4 N), capped at 1",
                           mode="lines", line=dict(width=1, color=R.MUTED)))
    R.vline(f, an, f"N_min = {an:.1f}")
    f.update_layout(title="Long-run outcome by initial population (fixed population)", xaxis_title="N0 (people)",
                    yaxis_title="share / units per person")
    f2 = go.Figure()
    for i, (n0, w) in enumerate(e1["series"].items()):
        f2.add_trace(go.Scatter(x=w.tick, y=w.potential_output / n0, name=f"N0 = {n0:.1f}", mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    f2.update_layout(title="Sustainable consumption per person over time", xaxis_title="tick",
                     yaxis_title="potential output / N")
    f2.add_hline(y=1, line=dict(color=R.MUTED, width=1))
    f3 = go.Figure()
    for i, ((rule, n0), w) in enumerate(e1["demo"].items()):
        f3.add_trace(go.Scatter(x=w.tick, y=w.population, name=f"{rule}, N0 = {n0}", mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    f3.update_layout(title="With demography on: which planner escapes the trap?", xaxis_title="tick",
                     yaxis_title="population (log)")
    f3.update_yaxes(type="log", range=[0, 4])
    surv = e1["survival"]
    summary["survival_threshold"] = surv
    body.append(R.fig_html(f))
    body.append(R.fig_html(f2))
    body.append("<h3>With demography on</h3><p>The analytical N<sub>min</sub> is a statement about a steady state. "
                "With population dynamics it only binds for a planner that protects its capital: the "
                "<code>max_feasible</code> planner keeps maintenance whole and cuts consumption, so people starve "
                f"and the city collapses for every N0 below {surv['max_feasible']:.2f}. The default <code>priority</code> "
                "planner feeds everyone and lets maintenance slip; while the factory's spare capacity lasts, "
                "everyone eats, population grows at 1% and passes N<sub>min</sub> before the factory has decayed "
                f"too far. It survives from N0 = {surv['priority']:.2f} upward — it borrows from its capital stock.</p>")
    body.append(R.tiles([("collapse threshold, max_feasible planner", f"{surv['max_feasible']:.2f}"),
                         ("collapse threshold, priority planner (default)", f"{surv['priority']:.2f}")]))
    body.append(R.fig_html(f3))
    pick = sw[sw.N0.isin([50, 80, 100, 120, 130, 134, 150, 200, 300, 400]) | (abs(sw.N0 - an) < 1e-9)].copy()
    pick["viable"] = pick["viable"].map({True: "yes", False: "no"})
    body.append("<p>Note on N0 = 50: the factory cannot even be crewed (F<sub>p</sub> = 50 plus farmers), decays to "
                "nothing and is scrapped; with nothing left to operate all labor is formally 'surplus', which is why "
                "its end surplus share reads 1.0 while consumption is zero.</p>")
    body.append(R.table_html(pick[["N0", "viable", "surplus_share_t1", "surplus_share_end", "cons_end",
                                   "factory_end", "potential_t1", "potential_end"]]))

    # ---- Experiment 2 ------------------------------------------------------------
    tb = e2["table"]
    summary["a_min"] = tb.to_dict(orient="records")
    body.append('<h2 id="e2">2 · Minimum farm productivity</h2>')
    body.append("<p>Viability requires the denominator 1 − c/(e·a) − v<sub>p</sub>c − v<sub>d</sub>c to cover the fixed "
                "overhead per person, so a<sub>min</sub>(N) = c / (e·(1 − v<sub>p</sub>c − v<sub>d</sub>c − (F<sub>p</sub>+F<sub>m</sub>)/N)). "
                "As N → ∞ it falls to 1/0.8 = 1.25; at N ≤ 100 no productivity is enough, because the fixed crews "
                "(80) plus the 20% variable labor exhaust the population.</p>")
    body.append(R.table_html(tb.assign(difference=tb.a_min_simulated - tb.a_min_analytical)))
    g = e2["grid"]
    piv = g.pivot(index="a", columns="N0", values="viable").astype(int)
    hm = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=[[0, "#f0efec"], [1, R.SEQ_BLUE[3]]],
                              showscale=False, xgap=1, ygap=1,
                              hovertemplate="N0 %{x}<br>a %{y}<br>viable %{z}<extra></extra>"))
    ns = np.linspace(101, 400, 300)
    hm.add_trace(go.Scatter(x=ns, y=[analytical_a_min(n, **params_of(base())) for n in ns], name="analytical a_min(N)",
                            mode="lines", line=dict(width=2, color=R.SERIES[1])))
    hm.update_layout(title="Viable (blue) vs not viable (gray), analytical boundary in orange",
                     xaxis_title="N0 (people)", yaxis_title="a (raw units per farmer)", hovermode="closest")
    hm.update_yaxes(range=[1, 10])
    sa = e2["a_sweep"]
    fa = go.Figure()
    for i, n in enumerate(sorted(sa.N0.unique())):
        d = sa[sa.N0 == n]
        fa.add_trace(go.Scatter(x=d.a, y=d.surplus_share - d.shortfall / n, name=f"N0 = {n}", mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    fa.add_hline(y=0, line=dict(color=R.MUTED, width=1))
    fa.update_layout(title="Labor surplus (+) or shortfall (−) as share of population", xaxis_title="a",
                     yaxis_title="share of labor")
    body.append('<div class="grid2">' + "".join(f"<div>{R.fig_html(x)}</div>" for x in (hm, fa)) + "</div>")

    # ---- Experiment 3 ------------------------------------------------------------
    t3 = e3["table"]
    s0 = t3[t3.s == 0]
    summary["s_zero_max_abs_growth"] = float(s0.max_abs_growth.max())
    summary["growth_vs_s"] = t3[t3.N0 == 150][["s", "growth_per_tick"]].to_dict(orient="records")
    body.append('<h2 id="e3">3 · Investment share and growth</h2>')
    body.append("<p>Fixed population, 300 ticks. Growth is measured as the growth of <em>potential output</em> — the "
                "largest consumption the city could sustain with all its labor and its current capital. With a fixed "
                "population, the only way it can rise is through capital: land improvements raise a, tools lower "
                "v<sub>p</sub> and v<sub>d</sub>. With s = 0 the surplus becomes leisure and nothing accumulates.</p>")
    body.append("<p>Higher s builds capital faster, but every unit of capital needs upkeep (1 worker per unit per "
                "tick), so with a fixed population capital stops paying for itself at a finite stock. Beyond s ≈ 0.35 "
                "the city reaches that optimum inside 300 ticks and the extra investment labor finds no worthwhile "
                "project and becomes leisure; s then only changes how <em>fast</em> the optimum is reached "
                "(first-50-tick growth keeps rising with s).</p>")
    body.append(R.tiles([("max |growth| at s = 0 (any tick, N0 150 & 300)", f"{summary['s_zero_max_abs_growth']:.1e}"),
                         ("growth/tick at s = 0.3, N0 = 150", f"{t3[(t3.s == 0.3) & (t3.N0 == 150)].growth_per_tick.iloc[0] * 100:.3f}%"),
                         ("growth/tick at s = 0.8, N0 = 150", f"{t3[(t3.s == 0.8) & (t3.N0 == 150)].growth_per_tick.iloc[0] * 100:.3f}%")]))
    fg = go.Figure()
    for i, (n0, col, lab) in enumerate([(150, "growth_first50", "N0 = 150, first 50 ticks"),
                                        (150, "growth_per_tick", "N0 = 150, 300 ticks"),
                                        (300, "growth_first50", "N0 = 300, first 50 ticks"),
                                        (300, "growth_per_tick", "N0 = 300, 300 ticks")]):
        d = t3[t3.N0 == n0]
        fg.add_trace(go.Scatter(x=d.s, y=d[col] * 100, name=lab, mode="lines+markers",
                                line=dict(width=2, color=R.SERIES[i]), marker=dict(size=8)))
    fg.update_layout(title="Average growth of potential output vs investment share", xaxis_title="s",
                     yaxis_title="% per tick")
    fs = go.Figure()
    for i, (s, w) in enumerate(e3["series"].items()):
        fs.add_trace(go.Scatter(x=w.tick, y=w.potential_output, name=f"s = {s}", mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    fs.update_layout(title="Potential output over time, N0 = 150", xaxis_title="tick", yaxis_title="final units / tick")
    body.append('<div class="grid2">' + "".join(f"<div>{R.fig_html(x)}</div>" for x in (fg, fs)) + "</div>")
    body.append("<h3>With demography on</h3><p>Population grows at n = 1% while everyone is fed. Without investment "
                "the single initial factory (capacity 1,000) becomes the ceiling: the city fills it and stops.</p>")
    dd = e3["demo"]
    fd = go.Figure()
    for i, (s, w) in enumerate(e3["demo_series"].items()):
        fd.add_trace(go.Scatter(x=w.tick, y=w.population, name=f"s = {s}", mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    fd.update_layout(title="Population with demography on", xaxis_title="tick", yaxis_title="population (log)")
    fd.update_yaxes(type="log")
    body.append('<div class="grid2"><div>' + R.fig_html(fd) + "</div><div>" + R.table_html(dd) + "</div></div>")

    # ---- Experiment 4 ------------------------------------------------------------
    t4 = e4["table"]
    summary["long_run"] = t4.to_dict(orient="records")
    body.append('<h2 id="e4">4 · Long run: demography and land limits</h2>')
    body.append("<p>Population grows at 1% per tick whenever it is fed (the spec's rule is all-or-nothing), so the "
                "slowdown shows up first in the <em>surplus</em>: as farmers exceed the land limit, each extra person "
                "needs more farm labor, leisure and investment shrink, and when the surplus reaches zero population "
                "growth stops. Land improvements postpone the wall but cannot remove it because their returns "
                "(elasticity 0.25) are weaker than the land limit's diminishing returns (β = 0.5).</p>")
    body.append(R.table_html(t4.rename(columns={"tick_growth_below_0_1pct": "tick growth < 0.1%",
                                                "drift_last200": "pop change, last 200 ticks",
                                                "farm_share_end": "farm labor share, end",
                                                "eff_a_end": "effective a, end", "eff_a_peak": "effective a, peak",
                                                "leisure_peak": "leisure/person, peak",
                                                "leisure_end": "leisure/person, end"})))
    ser = e4["series"]
    labels = list(ser)
    fp = go.Figure()
    for i, k in enumerate(labels):
        fp.add_trace(go.Scatter(x=ser[k].tick, y=ser[k].population, name=k, mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    fp.update_layout(title="Population", xaxis_title="tick", yaxis_title="people (log)")
    fp.update_yaxes(type="log")
    w = ser[labels[0]]
    fl = go.Figure()
    for i, (lab, col) in enumerate([("farming", "labor_share/farming"), ("processing", "labor_share/processing"),
                                    ("distribution", "labor_share/distribution"), ("maintenance", "labor_share/maintenance"),
                                    ("investment", "labor_share/investment"), ("leisure", "labor_share/leisure")]):
        fl.add_trace(go.Scatter(x=w.tick, y=w[col].rolling(10, min_periods=1).mean(), name=lab, mode="lines", stackgroup="one",
                                line=dict(width=0.5, color="#fcfcfb"), fillcolor=R.SERIES[i]))
    fl.update_layout(title="Labor allocation, default variant (10-tick average)", xaxis_title="tick", yaxis_title="share of population")
    fe = go.Figure()
    for i, k in enumerate(labels[:2]):
        fe.add_trace(go.Scatter(x=ser[k].tick, y=ser[k].effective_farm_productivity, name=k, mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    fe.update_layout(title="Effective farm productivity (raw per farmer)", xaxis_title="tick", yaxis_title="units / farmer")
    fr = go.Figure()
    for i, k in enumerate(labels):
        fr.add_trace(go.Scatter(x=ser[k].tick, y=ser[k].population_growth.rolling(25, min_periods=1).mean() * 100,
                                name=k, mode="lines", line=dict(width=2, color=R.SERIES[i])))
    fr.update_layout(title="Population growth (25-tick average)", xaxis_title="tick", yaxis_title="% per tick")
    body.append('<div class="grid2">' + "".join(f"<div>{R.fig_html(x)}</div>" for x in (fp, fr, fl, fe)) + "</div>")

    # ---- Experiment 5 ------------------------------------------------------------
    t5 = e5["table"]
    summary["shock"] = t5.to_dict(orient="records")
    body.append('<h2 id="e5">5 · Harvest shock: −30% for 3 ticks</h2>')
    body.append(f"<p>Shock at ticks {e5['start']}–{e5['start'] + e5['dur'] - 1}; the planner does not anticipate it. "
                "Storage keeps one tick of consumption in store; the store spoils 10% per tick and is refilled from "
                "surplus labor before investment. Consumption lost counts bread missing from the people actually alive; the population gap is reported "
                "separately. Investment lost is measured against the same scenario without the shock.</p>")
    body.append(R.table_html(t5))
    fc = go.Figure()
    for i, (k, w) in enumerate(e5["series"].items()):
        fc.add_trace(go.Scatter(x=w.tick, y=w.consumption_per_person, name=k, mode="lines",
                                line=dict(width=2, color=R.SERIES[i])))
    fc.update_layout(title="Consumption per person", xaxis_title="tick", yaxis_title="units / person")
    fc.update_xaxes(range=[e5["start"] - 5, e5["start"] + 25])
    fpop = go.Figure()
    for i, (k, w) in enumerate(e5["series"].items()):
        if "demography" in k and "fixed" not in k:
            fpop.add_trace(go.Scatter(x=w.tick, y=w.population, name=k, mode="lines",
                                      line=dict(width=2, color=R.SERIES[i])))
    fpop.update_layout(title="Population (demography on)", xaxis_title="tick", yaxis_title="people")
    fpop.update_xaxes(range=[e5["start"] - 5, e5["start"] + 25])
    fst = go.Figure()
    for i, (k, w) in enumerate(e5["series"].items()):
        if k.startswith("storage"):
            fst.add_trace(go.Scatter(x=w.tick, y=w.storage_stock / w.population, name=k, mode="lines",
                                     line=dict(width=2, color=R.SERIES[i])))
    fst.update_layout(title="Store, in ticks of consumption", xaxis_title="tick", yaxis_title="stock / (N·c)")
    fst.update_xaxes(range=[e5["start"] - 5, e5["start"] + 25])
    du = e5["durations"]
    fdu = go.Figure()
    for i, st in enumerate((False, True)):
        d = du[du.storage == st]
        fdu.add_trace(go.Scatter(x=d.duration, y=d.consumption_lost, name="storage" if st else "no storage",
                                 mode="lines+markers", line=dict(width=2, color=R.SERIES[i]), marker=dict(size=8)))
    fdu.update_layout(title="Consumption lost vs shock length (fixed population)", xaxis_title="ticks of −30% harvest",
                      yaxis_title="final units")
    body.append('<div class="grid2">' + "".join(f"<div>{R.fig_html(x)}</div>" for x in (fc, fpop, fst, fdu)) + "</div>")

    # ---- Invariants ------------------------------------------------------------------
    summary["tally"] = dict(runs=T.runs, ticks=T.ticks, seconds=T.seconds, by_experiment=T.by_experiment)
    body.append('<h2 id="inv">Invariants and runtime</h2>')
    body.append(f"<p class='pass'>All invariants passed on every tick of every run: {T.runs:,} runs, {T.ticks:,} ticks, "
                f"{T.seconds:.1f} s of simulation.</p><p>Checked each tick: double-entry identity, goods conservation "
                "(stock = previous + production − use − consumption − depreciation − spoilage − waste, for every good), "
                "money conservation, no negative stocks, balance sheets, labor adds up to population, factory "
                "conditions match the ledger, and nobody consumes more than N·c.</p>")
    body.append(R.table_html(pd.DataFrame([dict(experiment=k, runs=v) for k, v in T.by_experiment.items()])))

    R.write(path, "Stage 1 · Real economy baseline",
            "citysim — central planner, four sectors, no money. Generated by "
            "<code>python -m citysim experiments stage1</code>.", "\n".join(body))
    return summary


def main(out: str | Path = ROOT / "reports" / "stage_1.html") -> dict:
    res, T = run_all()
    summary = build_report(res, T, Path(out))
    Path(out).with_name("stage_1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
