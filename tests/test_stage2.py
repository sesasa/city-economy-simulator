"""Stage 2: market economy, money and barter."""
import numpy as np
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.sfc import transaction_flow_matrix
from citysim.layers.s2_money_prices import behavior as B
from tests.conftest import SCENARIOS


@pytest.fixture
def market():
    return load_scenario(SCENARIOS / "stage2_market.yaml")


# ---- rule functions ------------------------------------------------------------------
def test_trade_completion():
    assert B.trade_completion(5, 8, "money") == 1.0
    assert B.trade_completion(5, 1, "barter") == pytest.approx(1 / 16)
    assert B.trade_completion(5, 8, "barter") == pytest.approx(1 - (15 / 16) ** 8)


def test_markup_and_gaps():
    assert B.inventory_gap(0, 10) == 1.0 and B.inventory_gap(10, 10) == 0.0 and B.inventory_gap(30, 10) == -1.0
    assert B.order_gap(10, 5) == pytest.approx(0.5)
    m = B.markup_update(0.2, 1.0, 0.05, 0.2, 0.05, -0.5, 3)
    assert m == pytest.approx(0.25)
    assert B.markup_update(1.0, 0.0, 0.05, 0.2, 0.05, -0.5, 3) == pytest.approx(0.96)   # drifts back to normal


def test_wage_update_has_a_normal_unemployment_band():
    # 5% unemployment is normal: no change
    assert B.wage_update(1.0, 95.0, 100.0, 0.1, 0.1, 0.05) == pytest.approx(1.0)
    assert B.wage_update(1.0, 120.0, 100.0, 0.1, 0.1, 0.05) > 1.0
    assert B.wage_update(1.0, 50.0, 100.0, 0.1, 0.1, 0.05) < 1.0


def test_mobility_conserves_people_and_follows_jobs():
    pools = np.array([30.0, 70.0, 20.0, 30.0])
    new = B.mobility(pools, income=np.ones(4), jobs=pools, share_moving=0.1, sensitivity=3)
    assert new.sum() == pytest.approx(150)
    assert new == pytest.approx(pools)          # equal pay, jobs where people are: nobody moves
    new = B.mobility(pools, income=np.array([1, 2, 1, 1.0]), jobs=pools, share_moving=0.1, sensitivity=3)
    assert new[1] > pools[1]


def test_consumption_spending_bounds():
    s = B.consumption_spending(income=np.array([10.0]), cash=np.array([50.0]), price=2.0, need=np.array([10.0]),
                               c_max=np.array([15.0]), alpha_income=0.9, alpha_wealth=0.1)
    assert s[0] == pytest.approx(20.0)            # 0.9*10 + 0.1*40 = 13, raised to the cost of need (10 * 2)
    s = B.consumption_spending(np.array([100.0]), np.array([200.0]), 2.0, np.array([10.0]), np.array([15.0]), 0.9, 0.1)
    assert s[0] == pytest.approx(30.0)            # c_max * price caps it
    s = B.consumption_spending(np.array([1.0]), np.array([5.0]), 2.0, np.array([10.0]), np.array([15.0]), 0.9, 0.1)
    assert s[0] == pytest.approx(5.0)             # can't spend more than cash


def test_allocate_by_price_respects_stocks():
    sales = B.allocate_by_price(10, prices=[1.0, 2.0], stocks=[3.0, 100.0], choosiness=4)
    assert sales.sum() == pytest.approx(10) and sales[0] == pytest.approx(3.0)


# ---- whole-economy behaviour --------------------------------------------------------
def test_market_feeds_the_base_town(market):
    w = runner.run(market).wide()
    assert w.consumption_per_person.min() > 0.98
    assert w.consumption_per_person.iloc[-50:].mean() > 1.0


def test_money_is_neutral_when_prices_start_in_line(market):
    a = runner.run(with_params(market, M0=100, ticks=150)).wide()
    b = runner.run(with_params(market, M0=10000, ticks=150)).wide()
    np.testing.assert_allclose(a.consumption_per_person, b.consumption_per_person, rtol=1e-9)
    np.testing.assert_allclose(b.price_level / a.price_level, 100.0, rtol=1e-9)


def test_barter_with_little_search_collapses(market):
    w = runner.run(with_params(market, exchange="barter", barter_meetings=8, ticks=100)).wide()
    assert w.consumption_per_person.iloc[-20:].mean() < 0.05
    w = runner.run(with_params(market, exchange="barter", barter_meetings=256, ticks=150)).wide()
    assert w.consumption_per_person.iloc[-20:].mean() > 0.95


def test_transaction_flow_matrix_closes(market):
    sim = runner.build(with_params(market, ticks=60))
    for _ in range(60):
        sim.step()
    m = next(l for l in sim.layers if l.name == "market")
    t = transaction_flow_matrix(sim.world.ledger.journal, m.sector_of, {"cash"},
                                ["households", "farm", "factory", "distribution", "maintenance"])
    assert t["total"].abs().max() < 1e-9                     # rows: every payment is received by someone
    assert t.drop(columns="total").sum(axis=0).abs().max() < 1e-9   # columns: flows = change in money


def test_market_runs_with_generic_layers(market):
    runner.run(with_params(market, ticks=200, **{"demography.enabled": True, "land_limits.enabled": True,
                                                 "harvest_shocks.enabled": True,
                                                 "harvest_shocks.shocks": [{"start": 50, "duration": 3, "multiplier": 0.7}]}))


def test_market_physics_invariant_catches_free_bread(market):
    from citysim.core.invariants import InvariantViolation
    from citysim.core.phases import Phase
    sim = runner.build(with_params(market, ticks=3))

    def magic(world):
        world.ledger.produce("firm:factory:0", "final", 1000.0, "baked")
    sim.scheduler.table[Phase.PRODUCTION].append(("magic", magic))
    with pytest.raises(InvariantViolation, match="physics"):
        sim.step()


def test_planner_unchanged_by_stage2(base):
    """Stage 1 golden outputs still hold with Stage 2 installed (also covered by test_regression)."""
    assert "market" not in base.layers
