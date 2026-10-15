"""Build worlds from scenarios and run them: single runs, sweeps, multi-seed, compare."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable

import pandas as pd

from ..config import Scenario, with_params
from .invariants import ENGINE_INVARIANTS, Invariant, TickContext, check_all
from .layer import Layer
from .metrics import Metric, Recorder, wide
from .phases import Scheduler
from .world import World


@dataclass
class Simulation:
    scenario: Scenario
    world: World
    layers: list[Layer]
    scheduler: Scheduler
    invariants: list[Invariant]
    recorder: Recorder
    check_invariants: bool = True

    def step(self) -> None:
        w = self.world
        w.tick += 1
        w.begin_tick()
        opening = w.ledger.snapshot()
        self.scheduler.run_tick(w)
        if self.check_invariants:
            check_all(w, TickContext(w.tick, opening, w.ledger.journal), self.invariants)
        self.recorder.record(w)

    def run(self, ticks: int | None = None) -> "RunResult":
        n = self.scenario.ticks if ticks is None else ticks
        for _ in range(n):
            self.step()
        return RunResult(self.scenario, self.recorder.frame(), self.world, self.recorder.catalog())


@dataclass
class RunResult:
    scenario: Scenario
    frame: pd.DataFrame
    world: World
    catalog: pd.DataFrame = field(default_factory=pd.DataFrame)

    def wide(self) -> pd.DataFrame:
        return wide(self.frame)

    def series(self, metric: str, group: str = "") -> pd.Series:
        d = self.frame[(self.frame.metric == metric) & (self.frame.group == group)]
        return d.set_index("tick")["value"]


def build(scenario: Scenario, check_invariants: bool = True) -> Simulation:
    from ..layers import REGISTRY

    world = World(seed=scenario.seed)
    layers: list[Layer] = []
    active = scenario.active_layers()
    for name in active:
        if name not in REGISTRY:
            raise KeyError(f"unknown layer {name!r}; known: {sorted(REGISTRY)}")

    def offers(n: str) -> set[str]:
        return {n, *REGISTRY[n].provides}

    for i, name in enumerate(active):
        cls = REGISTRY[name]
        everywhere = set().union(*(offers(n) for n in active))
        missing = [r for r in cls.requires if r not in everywhere]
        if missing:
            raise ValueError(f"layer {name!r} requires {missing}, which are not enabled")
        before = set().union(set(), *(offers(n) for n in active[:i]))
        late = [r for r in cls.requires if r not in before]
        if late:
            raise ValueError(f"layer {name!r} must be listed after {late}")
        layers.append(cls(cls.Params.model_validate(scenario.layers[name] or {})))
    for layer in layers:
        layer.setup(world)
    invariants = list(ENGINE_INVARIANTS) + [inv for l in layers for inv in l.invariants()]
    metrics: list[Metric] = []
    for l in layers:
        metrics += l.metrics()
    sim = Simulation(scenario, world, layers, Scheduler(layers), invariants, Recorder(metrics),
                     check_invariants)
    # Check the initial state too (tick 0), with an empty journal.
    if check_invariants:
        check_all(world, TickContext(0, world.ledger.snapshot(), []), invariants)
    return sim


def run(scenario: Scenario, check_invariants: bool = True) -> RunResult:
    return build(scenario, check_invariants).run()


def sweep(scenario: Scenario, grid: dict[str, Iterable[Any]],
          reduce: Callable[[RunResult], dict[str, float]] | None = None,
          keep_frames: bool = False) -> tuple[pd.DataFrame, list[RunResult]]:
    """Run every point of a parameter grid. Returns a summary table (one row per point)
    and, if `keep_frames`, the full results. Invariants are checked at every point."""
    keys = list(grid)
    rows: list[dict[str, Any]] = []
    results: list[RunResult] = []
    for values in product(*(list(grid[k]) for k in keys)):
        point = dict(zip(keys, values))
        res = run(with_params(scenario, **point))
        row = dict(point)
        row.update((reduce or default_reduce)(res))
        rows.append(row)
        if keep_frames:
            results.append(res)
    return pd.DataFrame(rows), results


def multi_seed(scenario: Scenario, seeds: Iterable[int],
               reduce: Callable[[RunResult], dict[str, float]] | None = None) -> pd.DataFrame:
    table, _ = sweep(scenario, {"seed": list(seeds)}, reduce)
    return table


def compare(scenarios: list[Scenario]) -> list[RunResult]:
    return [run(s) for s in scenarios]


def default_reduce(res: RunResult) -> dict[str, float]:
    """Final value of every scalar metric."""
    d = res.frame
    last = d[(d.tick == d.tick.max()) & (d.group == "")]
    return dict(zip("final_" + last.metric, last.value))
