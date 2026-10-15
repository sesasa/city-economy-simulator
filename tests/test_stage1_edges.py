"""Edge cases found in review: conversion rates, bread needs, town death, land limits, storage."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from tests.conftest import SCENARIOS


@pytest.mark.parametrize("kw", [dict(e=1.3), dict(e=0.8, a=6), dict(c=1.5, a=8), dict(e=1.3, **{"storage.enabled": True})])
def test_other_conversion_rates_and_needs(base, kw):
    w = runner.run(with_params(base, ticks=200, **kw)).wide()
    c = kw.get("c", 1.0)
    assert (w.consumption_per_person - c).abs().max() < 1e-9


def test_town_can_die_without_errors(base):
    w = runner.run(with_params(base, N0=60, ticks=300, **{"demography.enabled": True})).wide()
    assert w.population.iloc[-1] < 1e-6


def test_land_limits_bite_in_a_big_town(base):
    # 4,000 people (5 factories) need 800 farmers at a = 5; with 500 good-land places each farmer harvests less
    on = runner.run(with_params(base, N0=4000, s=0.0, ticks=5, initial_factories=5, **{"land_limits.enabled": True})).wide()
    off = runner.run(with_params(base, N0=4000, s=0.0, ticks=5, initial_factories=5)).wide()
    assert off.effective_farm_productivity.iloc[0] == pytest.approx(5.0)
    assert on.effective_farm_productivity.iloc[0] < 5.0
    assert on["labor_share/farming"].iloc[0] > off["labor_share/farming"].iloc[0]


def test_storage_draws_cover_the_shock():
    sc = load_scenario(SCENARIOS / "stage1_shock_storage.yaml")
    w = runner.run(sc).wide()
    assert w["storage_flows/drawn"].loc[60:62].sum() > 0
    assert (w.consumption_per_person.loc[55:80] >= 1 - 1e-9).all()
    # spoilage is 10% of the stock carried into the tick
    s = w.storage_stock
    assert w["storage_flows/spoiled"].loc[30] == pytest.approx(0.1 * s.loc[29])


def test_unpaid_capital_is_caught(base):
    """A layer that builds capital without the labor to pay for it breaks an invariant."""
    from citysim.core.invariants import InvariantViolation
    from citysim.core.phases import Phase
    sim = runner.build(with_params(base, ticks=3))

    def free_tools(world):
        world.ledger.produce("farm", "land_improvement", 5.0, "investment")
    sim.scheduler.table[Phase.INVESTMENT].append(("free", free_tools))
    with pytest.raises(InvariantViolation, match="investment_paid"):
        sim.step()


def test_extra_harvest_is_caught(base):
    from citysim.core.invariants import InvariantViolation
    from citysim.core.phases import Phase
    sim = runner.build(with_params(base, ticks=3))

    def bumper(world):
        world.ledger.produce("farm", "raw", 10.0, "harvest")
        world.ledger.remove("farm", "raw", 10.0, "waste")
    sim.scheduler.table[Phase.PRODUCTION].append(("bumper", bumper))
    with pytest.raises(InvariantViolation, match="physics"):
        sim.step()
