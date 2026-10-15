import math

import pytest

from citysim.layers.s1_real_economy import rules as r

T = r.Tech(c=1, e=1, a=5)


def test_farm_output_and_inverse_without_land_limit():
    assert r.farm_output(30, 5, None, 0.5) == 150
    assert r.farmers_required(150, 5, None, 0.5) == 30


@pytest.mark.parametrize("farmers", [100, 500, 800, 5000])
def test_land_limit_diminishing_returns(farmers):
    out = r.farm_output(farmers, 5, 500, 0.5)
    eff = r.effective_farm_productivity(farmers, 5, 500, 0.5)
    assert out == pytest.approx(eff * farmers)
    if farmers > 500:
        assert eff == pytest.approx(5 * (500 / farmers) ** 0.5)
    assert r.farmers_required(out, 5, 500, 0.5) == pytest.approx(farmers)


def test_operating_factories_best_first():
    assert r.operating_factories(0, (1.0,), 1000) == (0, 0.0)
    assert r.operating_factories(900, (0.5, 1.0), 1000) == (1, 1000)
    assert r.operating_factories(1200, (0.5, 1.0), 1000) == (2, 1500)
    assert r.operating_factories(5000, (0.5, 1.0), 1000) == (2, 1500)


def test_labor_requirements_match_spec_rule_1():
    req = r.labor_requirements(150, T)
    assert req == pytest.approx({"farming": 30, "processing": 65, "distribution": 15, "maintenance": 30})


def test_allocate_priority():
    req = {"farming": 30, "processing": 65, "distribution": 15, "maintenance": 30}
    assert r.allocate_priority(req, 120, ["farming", "processing", "distribution", "maintenance"]) == \
        pytest.approx({"farming": 30, "processing": 65, "distribution": 15, "maintenance": 10})
    assert r.allocate_priority(req, 20, ["maintenance", "farming", "processing", "distribution"]) == \
        pytest.approx({"maintenance": 20, "farming": 0, "processing": 0, "distribution": 0})


def test_potential_output_closed_form():
    # N = 80 + 0.4 Y  ->  Y = (N - 80) / 0.4
    for n in (100, 150, 300):
        assert r.potential_output(n, T) == pytest.approx((n - 80) / 0.4, rel=1e-9)
    assert r.potential_output(70, T) == pytest.approx(0.0, abs=1e-9)   # fixed crews alone exceed 70
    assert r.potential_output(1e6, T) == 1000            # capped by factory capacity


def test_plan_surplus_and_shortage():
    p = r.make_plan(150, T, order=list(r.SECTORS))
    assert p.surplus == pytest.approx(10) and p.shortfall == 0
    p = r.make_plan(120, T, order=list(r.SECTORS))
    assert p.surplus == 0 and p.shortfall == pytest.approx(8)
    assert p.alloc["maintenance"] == pytest.approx(22)
    p = r.make_plan(120, T, order=list(r.SECTORS), shortage_rule="max_feasible")
    assert p.core_target == pytest.approx(100) and p.alloc["maintenance"] == pytest.approx(30)


def test_plan_restock_uses_surplus_without_distribution_labor():
    p = r.make_plan(150, T, order=list(r.SECTORS), restock_demand=1000)
    # extra unit costs 0.2 farm + 0.1 factory labor; 10 surplus workers -> 33.3 units
    assert p.restock == pytest.approx(10 / 0.3)
    assert p.surplus == pytest.approx(0, abs=1e-9)
    assert p.alloc["distribution"] == pytest.approx(15)


def test_investment_rules():
    ctx = r.InvestmentContext(labor=3, output=150, need=150, tech=T,
                              costs={"factory": 1000, "land_improvement": 200, "tools_processing": 200,
                                     "tools_distribution": 200}, threshold=0.9,
                              sector_labor={"farming": 30, "processing": 15, "distribution": 15})
    assert r.invest_bottleneck(ctx) == "land_improvement"     # farming uses most labor
    assert r.invest_best_return(ctx) in r.RELIEVES
    full = r.InvestmentContext(**{**ctx.__dict__, "need": 950})
    assert r.invest_bottleneck(full) == "factory"
    # saturated capital: upkeep exceeds the saving -> no project
    rich = r.Tech(c=1, e=1, a=5, land_improvement=1e4, tools_processing=1e4, tools_distribution=1e4)
    sat = r.InvestmentContext(**{**ctx.__dict__, "tech": rich})
    assert r.invest_bottleneck(sat) is None and r.invest_best_return(sat) is None


def test_factories_wanted():
    assert r.factories_wanted(800, 1000, 1000, 0.9) == 0
    assert r.factories_wanted(950, 1000, 1000, 0.9) == 1
    assert r.factories_wanted(2500, 1000, 1000, 0.9) == 2


def test_depreciation_and_demography_rules():
    assert r.unmaintained_share(22, 30) == pytest.approx(8 / 30)
    assert r.unmaintained_share(40, 30) == 0 and r.unmaintained_share(0, 0) == 0
    assert r.next_population(100, 1.0, 1.0, 0.01, 0.1) == pytest.approx(101)
    assert r.next_population(100, 0.7, 1.0, 0.01, 0.1) == pytest.approx(97)


def test_analytical_formulas():
    kw = dict(c=1, e=1, F_p=50, F_m=30, v_p=0.1, v_d=0.1)
    assert r.analytical_n_min(a=5, **kw) == pytest.approx(80 / 0.6)
    assert math.isinf(r.analytical_n_min(a=1.25, **kw))
    assert r.analytical_a_min(150, **kw) == pytest.approx(3.75)
    assert math.isinf(r.analytical_a_min(100, **kw))
