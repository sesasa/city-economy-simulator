"""Stage 11: the modern town and the internet."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS


@pytest.fixture(scope="module")
def town():
    return runner.run(load_scenario(SCENARIOS / "stage11_town.yaml")).wide()


@pytest.fixture(scope="module")
def web():
    return runner.run(load_scenario(SCENARIOS / "stage11_internet.yaml")).wide()


def test_town_settles_into_a_quasi_steady_state(town):
    calm = town[(town.year >= 1986) & (town.year < 1993)]
    assert 0.02 < calm.unemployment.mean() < 0.08
    assert calm.unemployment.std() < 0.03
    assert 0.0 < calm.inflation.mean() < 0.04
    assert calm.real_gdp.pct_change(4).mean() > 0


def test_internet_brings_a_boom_a_bust_and_lasting_gains(town, web):
    w = web.set_index("year")
    assert w.loc[:1993, "startups/founded"].max() == 0                     # nothing before the browser
    peak = w.net_index.idxmax()
    assert 1997 <= peak <= 2001
    assert w.loc[peak:peak + 3, "net_index"].min() < 0.5 * w.net_index.max()  # the crash
    assert w["startups/failed"].iloc[-1] > 0.3 * w["startups/founded"].iloc[-1]   # many fail
    assert w.real_gdp.iloc[-4:].mean() > town.set_index("year").real_gdp.iloc[-4:].mean()   # lasting gains


def test_central_bank_hikes_into_the_boom_and_cuts_after(web):
    w = web.set_index("year")
    assert w.loc[1999:2000.9, "policy_rate"].max() > w.loc[1996:1997, "policy_rate"].mean()
    assert w.loc[2001.5:2004, "policy_rate"].min() < w.loc[1999:2000.9, "policy_rate"].max() - 0.02


def test_startups_cannot_spend_money_they_do_not_have():
    sim = runner.build(with_params(load_scenario(SCENARIOS / "stage11_internet.yaml"), ticks=80))
    for _ in range(79):
        sim.step()
    net = next(l for l in sim.layers if l.name == "internet")
    s = net.alive()[0]

    def overspend(world):
        world.ledger.post("hh:workers", s.acc, "deposits", world.ledger.balance(s.acc, "deposits") + 50.0, "transfer", "startup wages")
    sim.scheduler.table[Phase.FINANCE].append(("overspend", overspend))
    with pytest.raises(InvariantViolation):
        sim.step()


def test_carriers_overbuild_and_the_boldest_go_bankrupt(web):
    w = web.set_index("year")
    assert w.loc[1999:2003, "network/utilization"].min() < 0.3               # most of the fibre lies dark
    assert w.loc[1999:2006, "network/price"].min() < 0.5                      # a price war
    assert 1 <= w.carrier_bankruptcies.iloc[-1] <= 2                          # the bold fail, the cautious survives
    first = w[w.carrier_bankruptcies > 0].index.min()
    assert 2001 <= first <= 2004
    assert w.carrier_defaults.sum() > 0                                       # and the bank takes the loss


def test_town_imports_only_what_its_factories_cannot_make(web):
    w = web.set_index("year")
    assert w.loc[1998:2001, "imports"].max() > 0                             # the boom outruns local factories
    assert w.loc[1986:1992, "imports"].mean() < 0.01 * w.loc[1986:1992, "gdp"].mean()
    assert (w.exports >= 0).all() and (w.foreign_deposits >= -1e-6).all()


def test_winner_takes_most_among_portals(web):
    w = web.set_index("year")
    assert w.loc[2008, "leader_share/portal"] > 0.3
