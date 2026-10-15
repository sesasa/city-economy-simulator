"""Stage 5: government and public goods."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS


@pytest.fixture
def s5():
    return load_scenario(SCENARIOS / "stage5_government.yaml")


def test_government_base_runs_and_helps(s5):
    gov = runner.run(with_params(s5, ticks=300)).wide()
    none = runner.run(with_params(s5, ticks=300, **{"government.enabled": False})).wide()
    assert gov.tax_revenue.iloc[-50:].mean() > 0
    assert gov.gini_wealth.iloc[-1] < none.gini_wealth.iloc[-1]
    assert gov.roads.iloc[-1] == pytest.approx(5, rel=0.05)


def test_roads_cut_distribution_labor(s5):
    no = runner.run(with_params(s5, ticks=200, road_target=0, seed=1)).wide()
    yes = runner.run(with_params(s5, ticks=200, road_target=20, seed=1)).wide()
    assert yes["labor_share/distribution"].iloc[-50:].mean() < no["labor_share/distribution"].iloc[-50:].mean()


def test_basic_income_stops_the_starvation_spiral(s5):
    w = runner.run(with_params(s5, ticks=250, basic_income=0.2, **{"demography.enabled": True})).wide()
    assert w.population.iloc[-1] > 300


def test_thin_bank_crowds_out_private_credit(s5):
    kw = dict(ticks=300, basic_income=0.3, income_tax=0.1, profit_tax=0.2, initial_capital=0.03, investment_share=0.05)
    free = sum(runner.run(with_params(s5, seed=s, bond_risk_weight=0.0, **kw)).wide().loans.iloc[-100:].mean() for s in (1, 2, 3))
    costly = sum(runner.run(with_params(s5, seed=s, bond_risk_weight=1.0, **kw)).wide().loans.iloc[-100:].mean() for s in (1, 2, 3))
    assert costly < free           # averaged over seeds, as in the experiment


def test_government_cannot_overspend(s5):
    sim = runner.build(with_params(s5, ticks=3))

    def splurge(world):
        world.ledger.post("households", "gov", "deposits", 1e6, "transfer", "benefits")
    sim.scheduler.table[Phase.INCOME].append(("splurge", splurge))
    with pytest.raises(InvariantViolation):
        sim.step()
