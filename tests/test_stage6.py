"""Stage 6: central bank and monetary policy."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS

DEMAND = {"demand_shocks.enabled": True, "demand_shocks.shocks": [dict(start=100, duration=30, multiplier=0.5)]}


@pytest.fixture
def s6():
    return load_scenario(SCENARIOS / "stage6_central_bank.yaml")


def test_taylor_rule_cuts_in_a_slump_and_respects_the_floor(s6):
    w = runner.run(with_params(s6, ticks=140, seed=1, **DEMAND)).wide()
    assert w.shadow_rate.loc[105:130].min() < 0          # the rule wants a negative rate ...
    assert w.policy_rate.min() >= 0                       # ... but the floor holds
    assert w.loans.iloc[-1] >= 0


def test_no_floor_softens_the_slump(s6):
    lost = {}
    for lb in (0.0, -0.05):
        tot = 0.0
        for seed in (1, 2, 3):
            w = runner.run(with_params(s6, ticks=180, seed=seed, **DEMAND, **{"central_bank.lower_bound": lb})).wide()
            tot += w.unemployment_rate.loc[100:180].sum()
        lost[lb] = tot
    assert lost[-0.05] < lost[0.0]


def test_money_rule_follows_its_path(s6):
    sim = runner.build(with_params(s6, ticks=60, **{"central_bank.rule": "money_growth", "central_bank.money_growth": 0.005}))
    cb = sim.world.state["economy"]["central_bank"]
    for _ in range(60):
        sim.step()
    assert cb.bank.money_supply(sim.world) >= cb.path(60) * (1 - 1e-9)       # topped up to the path ...
    assert sim.world.state["economy"]["policy_rate"] >= 0                       # ... and leaned on above it


def test_lender_of_last_resort_stops_a_panic_failure(s6):
    kw = dict(ticks=180, seed=1, panic_start=150, panic_duration=6)
    no = runner.run(with_params(s6, **kw, **{"central_bank.lender_of_last_resort": False})).wide()
    yes = runner.run(with_params(s6, **kw)).wide()
    assert no.bank_failure.sum() > 0
    assert yes.bank_failure.sum() == 0 and yes.lolr_lent.sum() > 0
    assert yes.emergency_loans.iloc[-1] < yes.emergency_loans.max()      # repaid once calm


def test_bailout_replaces_bail_in(s6):
    kw = dict(ticks=400, seed=2, procyclicality=0.05, capital_requirement=0.04, initial_capital=0.05)
    bailin = runner.run(with_params(s6, **kw)).wide()
    bailout = runner.run(with_params(s6, **kw, **{"central_bank.bailouts": True})).wide()
    assert bailin.haircut.sum() > 0
    assert bailout.bailout_cost.iloc[-1] > 0 and bailout.haircut.sum() < bailin.haircut.sum()


def test_unexplained_central_bank_money_is_caught(s6):
    sim = runner.build(with_params(s6, ticks=3))

    def counterfeit(world):
        world.ledger.post("households", "bank:0", "deposits", 50.0, "transfer", "deposit_interest_typo")
    sim.scheduler.table[Phase.FINANCE].append(("counterfeit", counterfeit))
    with pytest.raises(InvariantViolation):
        sim.step()
