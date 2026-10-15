"""Stage 8: several cities stepped together, linked by trade, money and migration.

A region scenario YAML looks like::

    name: two_cities
    seed: 42
    ticks: 300
    cities:
      grainbridge: {scenario: stage5_government.yaml, params: {farm_output_per_worker: 7}}
      millford:    {scenario: stage5_government.yaml, params: {factory_variable_labor: 0.05}}
    trade: {currency: union, transport_cost: 0.1}

Each city is a full simulation with its own ledger, layers and invariants. The region runs
all cities phase by phase in lockstep and calls the trade network's hooks after every city
has finished a phase. City metrics are recorded with the city name as a prefix
("millford.consumption_per_person"); region metrics have no prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import load_raw, load_scenario, with_params
from .core import runner
from .core.invariants import InvariantViolation, TickContext, check_all
from .core.metrics import Recorder, wide
from .core.phases import Phase


class CityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    params: dict[str, Any] = Field(default_factory=dict)


class RegionScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "region"
    description: str = ""
    seed: int = 0
    ticks: int = Field(100, ge=0)
    cities: dict[str, CityConfig]
    trade: dict[str, Any] = Field(default_factory=dict)


def load_region(path: str | Path) -> RegionScenario:
    data = load_raw(path)
    sc = RegionScenario.model_validate(data)
    base = Path(path).parent
    for c in sc.cities.values():                 # city scenario paths are relative to the region file
        c.scenario = str((base / c.scenario).resolve())
    return sc


def with_region(sc: RegionScenario, **kw) -> RegionScenario:
    """Override region fields: ticks/seed, trade params ("trade.x" or plain trade keys),
    and city params ("city.param")."""
    data = sc.model_dump()
    for k, v in kw.items():
        if k in ("ticks", "seed", "name"):
            data[k] = v
        elif k.startswith("trade."):
            data["trade"][k[6:]] = v
        elif "." in k and k.split(".", 1)[0] in data["cities"]:
            city, param = k.split(".", 1)
            data["cities"][city]["params"][param] = v
        else:
            data["trade"][k] = v
    return RegionScenario.model_validate(data)


@dataclass
class RegionResult:
    scenario: RegionScenario
    frames: dict[str, pd.DataFrame]          # per city, tidy
    region_frame: pd.DataFrame

    def wide(self) -> pd.DataFrame:
        parts = [wide(self.region_frame)]
        for name, f in self.frames.items():
            w = wide(f)
            w.columns = [f"{name}.{c}" for c in w.columns]
            parts.append(w)
        return pd.concat(parts, axis=1)

    def city(self, name: str) -> pd.DataFrame:
        return wide(self.frames[name])


class Region:
    def __init__(self, sc: RegionScenario, check_invariants: bool = True):
        from .layers.s8_trade.trade import TradeNetwork

        self.scenario = sc
        self.check = check_invariants
        self.names = list(sc.cities)
        self.sims: dict[str, runner.Simulation] = {}
        for i, (name, c) in enumerate(sc.cities.items()):
            city_sc = with_params(load_scenario(c.scenario), seed=sc.seed * 100 + i, ticks=sc.ticks, **c.params)
            self.sims[name] = runner.build(city_sc, check_invariants)
        self.trade = TradeNetwork(self, sc.trade)
        self.recorder = Recorder(self.trade.metrics())
        self.tick = 0

    def world(self, name: str):
        return self.sims[name].world

    def step(self) -> None:
        self.tick += 1
        openings = {}
        for name, sim in self.sims.items():
            w = sim.world
            w.tick = self.tick
            w.begin_tick()
            openings[name] = w.ledger.snapshot()
        hooks = self.trade.hooks()
        for phase in Phase:
            if phase == Phase.RECORD:
                continue
            for sim in self.sims.values():
                sim.world.phase = phase
                for _n, fn in sim.scheduler.table[phase]:
                    fn(sim.world)
            if phase in hooks:
                hooks[phase](self)
        if self.check:
            for name, sim in self.sims.items():
                w = sim.world
                try:
                    check_all(w, TickContext(w.tick, openings[name], w.ledger.journal), sim.invariants)
                except InvariantViolation as e:
                    for v in e.violations:
                        v.message = f"{name}: {v.message}"
                    raise InvariantViolation(e.tick, e.violations, w.ledger.journal) from None
            bad = self.trade.check(self)
            if bad:
                raise RuntimeError(f"region invariant violated at tick {self.tick}: " + "; ".join(bad))
        for sim in self.sims.values():
            sim.recorder.record(sim.world)
        self.recorder.record(self)

    def run(self, ticks: int | None = None) -> RegionResult:
        n = self.scenario.ticks if ticks is None else ticks
        for _ in range(n):
            self.step()
        return RegionResult(self.scenario, {k: s.recorder.frame() for k, s in self.sims.items()}, self.recorder.frame())


def run_region(sc: RegionScenario, check_invariants: bool = True) -> RegionResult:
    return Region(sc, check_invariants).run()
