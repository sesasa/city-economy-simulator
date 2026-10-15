import pytest

from citysim.config import with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation, TickContext, check_all, ENGINE_INVARIANTS
from citysim.core.phases import Phase


def test_direct_stock_mutation_is_caught(base):
    """Changing a stock without a posting breaks goods conservation; the run stops with tick + trace."""
    sim = runner.build(with_params(base, ticks=5))
    sim.step()

    def cheat(world):
        world.ledger.balances[("factory", "final")] += 1.0    # goods from nowhere
    sim.scheduler.table[Phase.PRODUCTION].append(("cheat", cheat))
    with pytest.raises(InvariantViolation) as e:
        sim.step()
    assert e.value.tick == 2
    msg = str(e.value)
    assert "goods_conservation" in msg and "double_entry" in msg and "Trace" in msg


def test_negative_stock_is_caught(base):
    sim = runner.build(with_params(base, ticks=3))

    def overdraw(world):
        world.ledger.transfer("farm", "factory", "raw", 1e6, "too much")
    sim.scheduler.table[Phase.PRODUCTION].insert(0, ("overdraw", overdraw))
    with pytest.raises(InvariantViolation, match="no_negative_stocks"):
        sim.step()


def test_labor_invariant_is_caught(base):
    sim = runner.build(with_params(base, ticks=3))

    def steal(world):
        world.vars["plan"].alloc["farming"] += 5
    sim.scheduler.table[Phase.PRODUCTION].insert(0, ("steal", steal))
    with pytest.raises(InvariantViolation, match="labor_adds_up"):
        sim.step()


def test_money_conservation_detects_unexplained_money(base):
    sim = runner.build(with_params(base, ticks=1))
    w = sim.world
    w.ledger.register_item("coin", "money")
    w.begin_tick()
    opening = w.ledger.snapshot()
    w.ledger.post("households", "ext:creation", "coin", 10, "creation")   # explicit creation: fine
    check_all(w, TickContext(0, opening, w.ledger.journal), ENGINE_INVARIANTS)
    w.ledger.balances[("households", "coin")] += 1                          # unexplained
    with pytest.raises(InvariantViolation, match="money_conservation"):
        check_all(w, TickContext(0, opening, w.ledger.journal), ENGINE_INVARIANTS)
