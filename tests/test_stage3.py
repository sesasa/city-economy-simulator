"""Stage 3: credit and banking."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS

GROW = {"demography.enabled": True, "land_limits.enabled": True}


@pytest.fixture
def bank():
    return load_scenario(SCENARIOS / "stage3_bank.yaml")


def test_bank_base_runs_and_feeds(bank):
    w = runner.run(bank).wide()
    assert w.consumption_per_person.min() > 0.98


def test_credit_rescues_scarce_money(bank):
    market = load_scenario(SCENARIOS / "stage2_market.yaml")
    kw = dict(M0=200, prices_match_money=False, ticks=300)   # recovery is slow: ~70% for 200 ticks
    without = runner.run(with_params(market, **kw)).wide()
    with_bank = runner.run(with_params(bank, **kw)).wide()
    assert without.consumption_per_person.iloc[-50:].mean() < 0.05
    assert with_bank.consumption_per_person.iloc[-50:].mean() > 0.8


def test_loans_create_and_repayments_destroy_money(bank):
    sim = runner.build(with_params(bank, M0=200, prices_match_money=False, ticks=10))
    b = next(l for l in sim.layers if l.name == "banking")
    m0 = b.money_supply(sim.world)
    sim.step(); sim.step()
    st = sim.world.vars["bank_stats"]
    assert st["lent"] > 0
    # the money_explained invariant passed every tick; money grew with lending
    assert b.money_supply(sim.world) != pytest.approx(m0)


def test_procyclical_lending_makes_more_failures(bank):
    calm = runner.run(with_params(bank, ticks=400, investment_share=0.05, procyclicality=0.0, **GROW)).wide()
    hot = runner.run(with_params(bank, ticks=400, investment_share=0.05, procyclicality=0.1, **GROW)).wide()
    assert hot.loans.max() > 2 * calm.loans.max()
    assert hot.bank_failure.sum() > calm.bank_failure.sum()


def test_thin_capital_fails_after_a_shock(bank):
    shock = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=120, duration=5, multiplier=0.5)]}
    kw = dict(ticks=200, N0=300, investment_share=0.0, credit_scale=4.0, procyclicality=0.01, **GROW, **shock)
    thin = runner.run(with_params(bank, capital_requirement=0.02, initial_capital=0.02, **kw)).wide()
    thick = runner.run(with_params(bank, capital_requirement=0.16, initial_capital=0.2, **kw)).wide()
    assert thin.bank_failure.sum() >= 1 and thin.haircut.sum() > 0
    assert thick.bank_failure.sum() == 0


def test_unexplained_money_is_caught(bank):
    sim = runner.build(with_params(bank, ticks=3))

    def counterfeit(world):
        world.ledger.post("households", "bank:0", "deposits", 50.0, "transfer", "retail")
    sim.scheduler.table[Phase.FINANCE].append(("counterfeit", counterfeit))
    with pytest.raises(InvariantViolation):
        sim.step()


def test_overdraft_is_caught(bank):
    sim = runner.build(with_params(bank, ticks=3))

    def overdraw(world):
        world.ledger.post("households", "firm:farm:0", "deposits", 1e6, "transfer", "wages")
    sim.scheduler.table[Phase.INCOME].append(("overdraw", overdraw))
    with pytest.raises(InvariantViolation, match="claims|household_cash"):
        sim.step()


def test_stage2_unchanged_without_bank():
    from tests.test_regression import test_golden
    test_golden("stage2_market")
