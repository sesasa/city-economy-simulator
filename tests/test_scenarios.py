"""Every scenario runs with invariants on; config, toggles, sweeps and seeds behave."""
import pytest

from citysim.config import load_scenario, parse_range, resolve_param, with_params
from citysim.core import runner
from citysim.layers import REGISTRY
from tests.conftest import SCENARIOS

ALL = sorted(p for p in SCENARIOS.glob("*.yaml") if "cities:" not in p.read_text())   # regions: test_stage8


@pytest.mark.parametrize("path", ALL, ids=[p.stem for p in ALL])
def test_scenario_runs_with_invariants(path):
    sc = load_scenario(path)
    res = runner.run(with_params(sc, ticks=min(sc.ticks, 400)))
    assert res.frame.tick.max() == min(sc.ticks, 400)


def test_inheritance(base):
    sc = load_scenario(SCENARIOS / "stage1_shock_no_storage.yaml")
    assert sc.ticks == 150 and sc.layers["storage"]["enabled"] is False
    assert sc.layers["harvest_shocks"]["shocks"][0]["multiplier"] == 0.7
    assert sc.layers["real_economy"]["farm_output_per_worker"] == 5


def test_param_resolution(base):
    assert resolve_param(base, "a") == ("real_economy", "farm_output_per_worker")
    assert resolve_param(base, "s") == ("investment", "investment_share")
    assert resolve_param(base, "storage.enabled") == ("storage", "enabled")
    assert resolve_param(base, "ticks") == (None, "ticks")
    with pytest.raises(KeyError):
        resolve_param(base, "nonsense")
    assert parse_range("a=1:10:0.5")[1][:3] == [1.0, 1.5, 2.0] and parse_range("a=1:10:0.5")[1][-1] == 10.0


def test_requires_is_enforced():
    from citysim.config import Scenario
    with pytest.raises(ValueError, match="requires"):
        runner.build(Scenario(layers={"investment": {}}))
    with pytest.raises(ValueError, match="listed after"):
        runner.build(Scenario(layers={"investment": {}, "real_economy": {}}))


def test_unknown_param_rejected(base):
    with pytest.raises(Exception):
        runner.build(with_params(base, **{"real_economy.farm_output_per_worker": 5}).model_copy(
            update={"layers": {"real_economy": {"no_such_param": 1}}}))


@pytest.mark.parametrize("layer", ["investment", "demography", "storage", "land_limits", "harvest_shocks"])
def test_each_layer_toggles(base, layer):
    on = with_params(base, **{f"{layer}.enabled": True}, ticks=30)
    off = with_params(base, **{f"{layer}.enabled": False}, ticks=30)
    runner.run(on); runner.run(off)


def test_absent_layer_equals_disabled_layer(base):
    """Removing a disabled layer from the scenario changes nothing."""
    a = runner.run(with_params(base, ticks=100)).frame
    data = base.model_dump()
    for k in ("demography", "storage", "land_limits", "harvest_shocks"):
        data["layers"].pop(k)
    from citysim.config import Scenario
    b = runner.run(Scenario.model_validate({**data, "ticks": 100})).frame
    assert a.equals(b)


def test_land_limits_inactive_below_limit(base):
    """With fewer than land_limit farmers, the land_limits layer must not change anything."""
    a = runner.run(with_params(base, ticks=100)).wide()
    b = runner.run(with_params(base, ticks=100, **{"land_limits.enabled": True})).wide()
    assert a[a.columns].equals(b[a.columns])


def test_sweep_checks_every_point(base):
    table, _ = runner.sweep(with_params(base, ticks=20), {"a": [3, 5], "s": [0.0, 0.3]})
    assert len(table) == 4 and "final_population" in table


def test_seeds_do_not_matter_in_stage1(base):
    """Stage 1 is deterministic: the seeded RNG exists but no rule draws from it."""
    t = runner.multi_seed(with_params(base, ticks=50), [1, 2, 3])
    cols = [c for c in t.columns if c.startswith("final_")]
    assert (t[cols].nunique() == 1).all()


def test_agent_representation_not_yet_available(base):
    sc = with_params(base, **{"real_economy.representation": {"households": "agents"}})
    with pytest.raises(NotImplementedError):
        runner.build(sc)


def test_registry_has_stage1_layers():
    assert {"real_economy", "investment", "demography", "storage", "land_limits", "harvest_shocks"} <= set(REGISTRY)
