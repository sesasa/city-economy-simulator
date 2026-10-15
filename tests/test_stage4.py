"""Stage 4: agent households, many firms, owners, services, entry and exit."""
import numpy as np
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.layers.s4_competition.agent_market import gini, top_share
from tests.conftest import SCENARIOS


@pytest.fixture
def s4():
    return load_scenario(SCENARIOS / "stage4_competition.yaml")


def test_gini_and_top_share():
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    assert gini(np.array([0, 0, 0, 1.0])) == pytest.approx(0.75)
    assert gini(np.array([1.0, 3.0]), np.array([3.0, 1.0])) == pytest.approx(gini(np.array([1, 1, 1, 3.0])))
    assert top_share(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 91.0]), np.ones(10), 0.1) == pytest.approx(0.91)


def test_base_runs_feeds_and_is_reproducible(s4):
    a = runner.run(with_params(s4, ticks=200)).frame
    b = runner.run(with_params(s4, ticks=200)).frame
    assert a.equals(b)                                 # seeded randomness: same seed, same history
    w = runner.run(with_params(s4, ticks=300)).wide()
    assert w.consumption_per_person.iloc[-50:].mean() > 0.9
    assert w.unemployment_rate.iloc[-50:].mean() < 0.1


def test_more_competition_lowers_markups(s4):
    one = runner.run(with_params(s4, ticks=300, farm_firms=1, distribution_firms=1, maintenance_firms=1,
                                 **{"entry_exit.enabled": False})).wide()
    many = runner.run(with_params(s4, ticks=300, farm_firms=8, distribution_firms=8, maintenance_firms=4,
                                  **{"entry_exit.enabled": False})).wide()
    assert one["markups/distribution"].iloc[-50:].mean() > many["markups/distribution"].iloc[-50:].mean()
    assert one["concentration/distribution"].iloc[-1] == pytest.approx(1.0)


def test_concentrated_ownership_raises_inequality(s4):
    few = runner.run(with_params(s4, ticks=300, owners_per_firm=1)).wide()
    public = runner.run(with_params(s4, ticks=300, ownership="public")).wide()
    assert few.gini_wealth.iloc[-1] > public.gini_wealth.iloc[-1] + 0.3


def test_services_cure_satiation_unemployment(s4):
    without = runner.run(with_params(s4, ticks=300, **{"services.enabled": False})).wide()
    with_s = runner.run(with_params(s4, ticks=300)).wide()
    assert without.unemployment_rate.iloc[-50:].mean() > 3 * with_s.unemployment_rate.iloc[-50:].mean()


def test_entry_and_exit_happen(s4):
    w = runner.run(with_params(s4, ticks=300)).wide()
    assert w.entries.iloc[-1] > 0


def test_unequal_market_with_demography_starves_the_poor(s4):
    """Finding that motivates Stage 5: without a safety net the poor go hungry and the town shrinks."""
    w = runner.run(with_params(s4, ticks=150, **{"demography.enabled": True})).wide()
    assert w.hungry_share.iloc[10:50].mean() > 0.5
    assert w.population.iloc[-1] < 0.5 * w.population.iloc[0]
