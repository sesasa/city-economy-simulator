"""Analytical checks: the simulation must reproduce theory where theory gives an answer."""
import pytest

from citysim.config import with_params
from citysim.core import runner
from citysim.layers.s1_real_economy.rules import analytical_a_min, analytical_n_min


def viable(sc):
    w = runner.run(sc).wide()
    return bool((w.labor_shortfall <= 1e-9).all() and (w.consumption_per_person >= 1 - 1e-9).all())


def bisect(sc, key, lo, hi, tol=1e-3):
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if viable(with_params(sc, **{key: mid})) else (mid, hi)
    return hi


def test_n_min_matches_formula_within_one_person(base):
    an = analytical_n_min(a=5, c=1, e=1, F_p=50, F_m=30, v_p=0.1, v_d=0.1)
    sim = bisect(with_params(base, ticks=10), "N0", 50, 400)
    assert abs(sim - an) < 1.0
    assert abs(sim - an) < 1e-2          # in fact it matches to bisection tolerance


@pytest.mark.parametrize("n", [150, 300, 1000])
def test_a_min_matches_formula(base, n):
    sim = bisect(with_params(base, ticks=10, N0=n), "a", 1, 10, tol=1e-4)
    assert sim == pytest.approx(analytical_a_min(n, c=1, e=1, F_p=50, F_m=30, v_p=0.1, v_d=0.1), abs=1e-3)


def test_stationary_at_n_min(base):
    an = 80 / 0.6
    w = runner.run(with_params(base, N0=an, ticks=300)).wide()
    assert w.surplus_share.abs().max() < 1e-9
    assert w.potential_output.iloc[-1] == pytest.approx(w.potential_output.iloc[0])
    assert (w.consumption_per_person - 1.0).abs().max() < 1e-12


def test_long_run_consumption_below_n_min(base):
    # priority planner, fixed population: settles where the freed labor can maintain the factory
    w = runner.run(with_params(base, N0=120, ticks=800)).wide()
    assert w.consumption_per_person.iloc[-1] == pytest.approx((120 - 80) / (0.4 * 120), rel=1e-3)


@pytest.mark.parametrize("n0", [150, 300])
def test_zero_investment_gives_zero_growth(base, n0):
    w = runner.run(with_params(base, s=0.0, N0=n0, ticks=300)).wide()
    assert w.surplus_share.iloc[0] > 0                 # a surplus exists...
    assert w.growth_rate.abs().max() == 0.0            # ...but nothing grows
    assert w.investment_labor.abs().max() == 0.0


def test_positive_investment_gives_growth(base):
    g = [runner.run(with_params(base, s=s, ticks=50)).wide().potential_output.iloc[-1] for s in (0.0, 0.1, 0.3)]
    assert g[0] < g[1] < g[2]


def test_long_run_reaches_plateau(base):
    from citysim.config import load_scenario
    from tests.conftest import SCENARIOS
    sc = with_params(load_scenario(SCENARIOS / "stage1_long_run.yaml"), land_improvement_elasticity=0.0)
    w = runner.run(sc).wide()
    early = w.population_growth.iloc[:100].mean()
    tail = w.population.iloc[-500:]
    assert early == pytest.approx(0.01)
    assert abs(tail.iloc[-1] / tail.iloc[0] - 1) < 0.02
    assert w.leisure_per_person.iloc[-1] < 1e-6
