"""Stage 7: financial markets and risk sharing."""
import numpy as np
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS


@pytest.fixture
def s7():
    return load_scenario(SCENARIOS / "stage7_markets.yaml")


def test_shares_conserved_and_dividends_follow_shares(s7):
    sim = runner.build(with_params(s7, ticks=60, seed=1))
    for _ in range(60):
        sim.step()
    sm = next(l for l in sim.layers if l.name == "stock_market")
    assert sm.held.sum() == pytest.approx(sm.n)
    assert sim.world.ledger.balance("households", "shares") == pytest.approx(sm.n)
    assert (sm.held >= 0).all() and (sm.debt >= 0).all()


def test_leverage_inflates_bubbles(s7):
    peak = {}
    for mr in (1.0, 0.2):
        peak[mr] = np.mean([runner.run(with_params(s7, seed=s, ticks=300, margin_requirement=mr)).wide().price_to_value.max()
                            for s in (1, 2, 3)])
    assert peak[0.2] > peak[1.0]


def test_no_borrowing_without_margin(s7):
    w = runner.run(with_params(s7, seed=1, ticks=150, margin_requirement=1.0)).wide()
    assert w.margin_debt.max() == pytest.approx(0.0, abs=1e-9)


def test_futures_transfer_money_both_ways(s7):
    w = runner.run(with_params(s7, seed=1, ticks=150, futures_hedge=1.0,
                               **{"harvest_shocks.enabled": True, "harvest_shocks.volatility": 0.2})).wide()
    assert (w.hedge_pnl > 0).any() and (w.hedge_pnl < 0).any()


def test_thin_insurer_cannot_pay_a_disaster(s7):
    kw = dict(seed=1, ticks=170, insurance_cover=1.0,
              **{"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=150, duration=5, multiplier=0.4)]})
    w = runner.run(with_params(s7, **kw)).wide()
    assert w.unpaid_claims.loc[150:160].sum() > 0
    assert w.insurer_capital.min() >= 0


def test_share_theft_is_caught(s7):
    sim = runner.build(with_params(s7, ticks=30, seed=1))
    sm = next(l for l in sim.layers if l.name == "stock_market")

    def steal(world):
        sm.held[0] += 5.0
    sim.scheduler.table[Phase.INVESTMENT].append(("steal", steal))
    with pytest.raises(InvariantViolation):
        for _ in range(30):
            sim.step()
