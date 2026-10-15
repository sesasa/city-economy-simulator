"""Stage 9: technology and long-run growth."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS


@pytest.fixture
def plateau():
    return load_scenario(SCENARIOS / "stage9_plateau.yaml")


@pytest.fixture
def market():
    return load_scenario(SCENARIOS / "stage9_market.yaml")


def test_research_breaks_the_plateau(plateau):
    none = runner.run(with_params(plateau, ticks=1400, **{"technology.enabled": False})).wide()
    tech = runner.run(with_params(plateau, ticks=1400)).wide()
    assert none.population.pct_change().iloc[-50:].mean() < 0.001          # Stage 1 plateau
    assert tech.population.pct_change().iloc[-50:].mean() > 0.005          # still growing
    assert tech.population.iloc[-1] > 3 * none.population.iloc[-1]


def test_researchers_eat_but_do_not_produce(plateau):
    sim = runner.build(with_params(plateau, ticks=5, research_share=0.2))
    for _ in range(5):
        sim.step()
    plan = sim.world.vars["plan"]                 # made before this tick's births
    N = plan.people
    assert plan.workers == pytest.approx(N * 0.8, rel=0.02)
    assert plan.need == pytest.approx(N * 1.0)        # everyone eats


def test_too_much_research_starves(plateau):
    some = runner.run(with_params(plateau, ticks=600, research_share=0.02, lbd_exponent=0.0)).wide()
    lots = runner.run(with_params(plateau, ticks=600, research_share=0.5, lbd_exponent=0.0)).wide()
    assert lots.population.iloc[-1] < some.population.iloc[-1]


def test_market_research_moves_the_frontier_and_firms_follow(market):
    w = runner.run(with_params(market, ticks=200, seed=1)).wide()
    assert w.frontier.iloc[-1] > 1.05
    assert 1.0 < w.tech_level.iloc[-1] <= w.frontier.iloc[-1] * 1.1     # learning by doing can add a little
    assert w.researchers.iloc[-50:].mean() > 0


def test_no_diffusion_no_adoption(market):
    w = runner.run(with_params(market, ticks=150, seed=1, diffusion=0.0, lbd_exponent=0.0)).wide()
    assert w.frontier.iloc[-1] > 1.0
    assert w.tech_level.max() <= 1.0 + 1e-9           # entrants start below average; nobody adopts


def test_using_more_than_the_frontier_is_caught(market):
    sim = runner.build(with_params(market, ticks=5, seed=1))
    m = sim.world.state["economy"]["market_layer"]

    def cheat(world):
        m.firms[0].adopted = 10.0
    sim.scheduler.table[Phase.FINANCE].append(("cheat", cheat))
    with pytest.raises(InvariantViolation):
        sim.step()
