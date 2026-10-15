"""Scenario configuration: YAML files validated by pydantic, with inheritance.

A scenario YAML looks like::

    inherits: stage1_base.yaml     # optional, path relative to this file
    name: my_scenario
    seed: 42
    ticks: 300
    layers:                        # order = hook order within each phase
      real_economy: {farm_output_per_worker: 5}
      investment:   {investment_share: 0.3}
      demography:   {enabled: false}

Child values are deep-merged over the parent. A layer is active when it is
listed and its `enabled` flag is true (the default).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "scenario"
    description: str = ""
    seed: int = 0
    ticks: int = Field(100, ge=0)
    layers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def active_layers(self) -> list[str]:
        return [n for n, p in self.layers.items() if (p or {}).get("enabled", True)]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_raw(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("inherits", None)
    if parent:
        data = deep_merge(load_raw(path.parent / parent), data)
    return data


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate(load_raw(path))


def scenario_from_dict(data: dict) -> Scenario:
    return Scenario.model_validate(copy.deepcopy(data))


# ---------------------------------------------------------------------------
# Parameter paths: "a", "farm_output_per_worker", "real_economy.farm_output_per_worker",
# "ticks", "seed", "demography.enabled"
# ---------------------------------------------------------------------------

def resolve_param(scenario: Scenario, key: str) -> tuple[str | None, str]:
    """Return (layer_name or None for top-level, param_name)."""
    from .layers import REGISTRY

    if key in ("ticks", "seed", "name"):
        return None, key
    if "." in key:
        layer, param = key.split(".", 1)
        if layer not in REGISTRY:
            raise KeyError(f"unknown layer {layer!r} in {key!r}")
        param = REGISTRY[layer].aliases.get(param, param)
        _check_param(layer, param)
        return layer, param
    hits: list[tuple[str, str]] = []
    for lname, cls in REGISTRY.items():
        if key in cls.aliases:
            hits.append((lname, cls.aliases[key]))
        elif key in cls.Params.model_fields and key != "enabled":
            hits.append((lname, key))
    in_scenario = [h for h in hits if h[0] in scenario.layers]
    hits = in_scenario or hits
    if len(hits) != 1:
        raise KeyError(f"parameter {key!r} is {'unknown' if not hits else 'ambiguous: ' + str(hits)}; "
                       "use layer.param")
    return hits[0]


def _check_param(layer: str, param: str) -> None:
    from .layers import REGISTRY

    if param not in REGISTRY[layer].Params.model_fields:
        raise KeyError(f"layer {layer!r} has no parameter {param!r}")


def with_params(scenario: Scenario, **overrides: Any) -> Scenario:
    """Copy of `scenario` with parameters overridden (keys as accepted by resolve_param)."""
    data = scenario.model_dump()
    for key, value in overrides.items():
        layer, param = resolve_param(scenario, key)
        if layer is None:
            data[param] = value
        else:
            data["layers"].setdefault(layer, {})[param] = value
    return Scenario.model_validate(data)


def parse_range(spec: str) -> tuple[str, list[float]]:
    """'a=1:10:0.5' -> ('a', [1.0, 1.5, ..., 10.0]); 'a=1,2,5' -> ('a', [1, 2, 5])."""
    key, _, rng = spec.partition("=")
    if not rng:
        raise ValueError(f"bad --param {spec!r}; expected name=start:stop:step or name=v1,v2")
    if ":" in rng:
        start, stop, step = (float(x) for x in rng.split(":"))
        n = int(round((stop - start) / step))
        values = [round(start + i * step, 12) for i in range(n + 1)]
    else:
        values = [float(x) for x in rng.split(",")]
    return key.strip(), values
