"""Agent groups: a common interface for sector aggregates and (later) individual agents.

Every sector or household population is an `AgentGroup`. Stage 1 only uses
`representation="aggregate"`: one record standing for the whole group, with
float-valued head counts. The `agents` representation will store per-agent
attributes as numpy arrays (one array per attribute, not one object per agent)
so that later stages can swap it in without touching the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Representation = Literal["aggregate", "agents"]


@dataclass
class AgentGroup:
    """Base interface. `account` is the ledger account holding the group's stocks."""

    name: str
    account: str
    representation: Representation = "aggregate"

    def size(self) -> float:
        raise NotImplementedError

    def set_size(self, n: float) -> None:
        raise NotImplementedError


@dataclass
class AggregateGroup(AgentGroup):
    """A whole sector or population as a single aggregate (float head count)."""

    count: float = 0.0
    labor: float = 0.0          # workers assigned this tick
    attrs: dict[str, float] = field(default_factory=dict)

    def size(self) -> float:
        return self.count

    def set_size(self, n: float) -> None:
        if n < 0:
            raise ValueError(f"{self.name}: negative size {n}")
        self.count = float(n)


@dataclass
class ArrayGroup(AgentGroup):
    """Placeholder for individual agents stored as numpy arrays (used from Stage 4)."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def size(self) -> float:
        first = next(iter(self.arrays.values()), None)
        return 0.0 if first is None else float(len(first))

    def set_size(self, n: float) -> None:  # pragma: no cover - not used in Stage 1
        raise NotImplementedError("agent-mode resizing arrives with the agent-based stages")


def make_group(name: str, account: str, representation: Representation, count: float = 0.0) -> AgentGroup:
    if representation == "aggregate":
        return AggregateGroup(name=name, account=account, count=count)
    raise NotImplementedError(
        f"{name}: representation='agents' is not implemented in Stage 1 "
        "(see DECISIONS.md D-03); use 'aggregate'"
    )
