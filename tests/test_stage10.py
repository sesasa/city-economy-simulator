"""Stage 10: the modern economy and comparing allocation systems."""
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.core.invariants import InvariantViolation
from citysim.core.phases import Phase
from tests.conftest import SCENARIOS


def test_modern_economy_looks_modern():
    w = runner.run(with_params(load_scenario(SCENARIOS / "stage10_modern.yaml"), seed=1, ticks=250)).wide().iloc[-100:]
    assert 0.005 < w["labor_share/farm"].mean() < 0.05
    assert w["labor_share/services"].mean() > w["labor_share/factory"].mean()
    assert (w.cash_hoarded / w.money_supply).mean() < 0.1           # money is mostly bank deposits


def test_planner_serves_with_free_labor():
    w = runner.run(with_params(load_scenario(SCENARIOS / "stage10_planned.yaml"), ticks=150)).wide()
    assert w.services_per_person.iloc[-50:].mean() > 0.5
    assert w.consumption_per_person.iloc[-50:].mean() == pytest.approx(1.0)


def test_planned_services_cannot_use_working_labor():
    sim = runner.build(with_params(load_scenario(SCENARIOS / "stage10_planned.yaml"), ticks=3))
    layer = next(l for l in sim.layers if l.name == "planned_services")

    def overreach(world):
        layer.last["labor"] = layer.last["free"] + 100.0
    sim.scheduler.table[Phase.FINANCE].append(("overreach", overreach))
    with pytest.raises(InvariantViolation):
        sim.step()


def test_every_layer_can_be_switched_off():
    from citysim.experiments.stage10 import ABLATION, MODERN
    for kw in ABLATION.values():
        runner.run(with_params(load_scenario(MODERN), seed=2, ticks=30, **kw))
