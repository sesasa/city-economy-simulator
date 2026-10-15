"""Tick phases and the scheduler that runs layer hooks in a fixed order."""
from __future__ import annotations

from enum import IntEnum
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .layer import Layer
    from .world import World


class Phase(IntEnum):
    """Fixed phase order within one tick (SPEC: 'Phase order within one tick')."""

    SHOCKS_POLICY = 1   # exogenous shocks and policy (incl. the Stage 1 planner)
    PRODUCTION = 2      # farming first, then processing
    MAINTENANCE = 3     # maintenance and depreciation
    INCOME = 4          # wages, profits, interest, taxes, transfers
    DISTRIBUTION = 5    # goods move to consumers, purchases settle
    INVESTMENT = 6      # investment and capital formation
    FINANCE = 7         # loans, repayments, defaults, asset trades
    DEMOGRAPHY = 8      # births, deaths, migration, job switching
    RECORD = 9          # invariant checks and metric recording (engine-owned)


Hook = Callable[["World"], None]


class Scheduler:
    """Collects hooks from layers and runs them phase by phase.

    Within a phase, hooks run in the order the layers are listed in the scenario.
    Phase.RECORD is reserved for the engine (invariants + metrics).
    """

    def __init__(self, layers: list["Layer"]):
        self.table: dict[Phase, list[tuple[str, Hook]]] = {p: [] for p in Phase}
        for layer in layers:
            for phase, fn in layer.hooks().items():
                if phase == Phase.RECORD:
                    raise ValueError(f"layer {layer.name!r} may not hook Phase.RECORD")
                self.table[Phase(phase)].append((layer.name, fn))

    def run_tick(self, world: "World") -> None:
        for phase in Phase:
            if phase == Phase.RECORD:
                continue
            world.phase = phase
            for _name, fn in self.table[phase]:
                fn(world)

    def describe(self) -> list[tuple[str, str]]:
        return [(p.name, n) for p in Phase for n, _ in self.table[p]]
