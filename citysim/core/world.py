"""World state shared by all layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agents import AgentGroup
from .ledger import Ledger, open_standard_external_accounts
from .phases import Phase


@dataclass
class World:
    """Tick counter, agents/sector aggregates, goods stocks (via the ledger), RNG.

    * `ledger` is the single source of truth for every stock of goods and money.
    * `groups` holds sector aggregates / agent populations by name.
    * `state` is per-layer persistent state, namespaced by layer name.
    * `vars` is a per-tick blackboard layers use to pass plans and results to
      later phases; it is cleared at the start of every tick. Keys are
      documented by the layers that write them.
    """

    seed: int = 0
    tick: int = 0
    phase: Phase = Phase.SHOCKS_POLICY
    ledger: Ledger = field(default_factory=Ledger)
    groups: dict[str, AgentGroup] = field(default_factory=dict)
    state: dict[str, dict[str, Any]] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        open_standard_external_accounts(self.ledger)

    # ---- helpers -----------------------------------------------------------
    def add_group(self, group: AgentGroup) -> AgentGroup:
        if group.name in self.groups:
            raise ValueError(f"group {group.name!r} already registered")
        self.groups[group.name] = group
        self.ledger.open_account(group.account)
        return group

    def stock(self, holder: str, item: str) -> float:
        return self.ledger.balance(holder, item)

    def layer_state(self, layer_name: str) -> dict[str, Any]:
        return self.state.setdefault(layer_name, {})

    def begin_tick(self) -> None:
        self.vars = {}
        self.ledger.begin_tick(self.tick)
