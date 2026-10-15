"""Layer base class: the plugin interface every economic mechanism implements."""
from __future__ import annotations

from typing import Callable, ClassVar, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from .phases import Phase

if TYPE_CHECKING:
    from .invariants import Invariant
    from .metrics import Metric
    from .world import World


class LayerParams(BaseModel):
    """Base for layer parameter models. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Layer:
    """A toggleable mechanism.

    Subclasses set `name`, `requires`, `Params` and optionally `aliases`
    (short symbols such as 'a' -> 'farm_output_per_worker' used by the CLI).
    """

    name: ClassVar[str] = "layer"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = []      # capabilities other layers may require (e.g. "economy")
    Params: ClassVar[type[LayerParams]] = LayerParams
    aliases: ClassVar[dict[str, str]] = {}

    def __init__(self, params: LayerParams):
        self.p = params

    def setup(self, world: "World") -> None:
        """Register agents, accounts and goods; initialise stocks."""

    def hooks(self) -> dict[Phase, Callable[["World"], None]]:
        return {}

    def invariants(self) -> list["Invariant"]:
        return []

    def metrics(self) -> list["Metric"]:
        return []
