"""Bridge to the Grainbridge town twin (twin/index.html + twin/engine.js).

`scenario_to_twin` turns a Stage 1 scenario into the parameter object the JS
engine takes; tests/test_twin_parity.py uses it to check that the twin follows
the same rules as the Python model. `build_standalone` writes a single
self-contained HTML file you can open locally (reports/grainbridge.html).
"""
from __future__ import annotations

from pathlib import Path

from .config import Scenario
from .layers import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "twin"


def _params(sc: Scenario, layer: str) -> tuple[bool, dict]:
    raw = sc.layers.get(layer)
    p = REGISTRY[layer].Params.model_validate(raw or {})
    return (raw is not None and p.enabled), p.model_dump()


def scenario_to_twin(sc: Scenario) -> dict:
    _, re = _params(sc, "real_economy")
    inv_on, inv = _params(sc, "investment")
    dem_on, dem = _params(sc, "demography")
    sto_on, sto = _params(sc, "storage")
    land_on, land = _params(sc, "land_limits")
    shock_on, sh = _params(sc, "harvest_shocks")
    return {
        "a": re["farm_output_per_worker"], "c": re["consumption_per_person"], "e": re["conversion_rate"],
        "F_p": re["factory_fixed_crew"], "v_p": re["factory_variable_labor"], "v_d": re["distribution_labor"],
        "F_m": re["maintenance_fixed_crew"], "delta": re["depreciation_rate"], "capacity": re["factory_capacity"],
        "N0": re["initial_population"], "initial_factories": re["initial_factories"],
        "scrap_condition": re["scrap_condition"],
        "land_elasticity": re["land_improvement_elasticity"], "tool_elasticity": re["tool_elasticity"],
        "land_maintenance": re["land_improvement_maintenance"], "tool_maintenance": re["tool_maintenance"],
        "priority": re["priority"], "shortage_rule": re["shortage_rule"],
        "investment": {"enabled": inv_on, "s": inv["investment_share"], "rule": inv["rule"],
                       "threshold": inv["bottleneck_threshold"], "factory_cost": inv["factory_cost"],
                       "land_cost": inv["land_improvement_cost"], "tool_cost": inv["tool_cost"]},
        "demography": {"enabled": dem_on, "n": dem["population_growth"], "mu": dem["starvation_rate"]},
        "storage": {"enabled": sto_on, "spoilage": sto["storage_spoilage"],
                    "target_ticks": sto["storage_target_ticks"], "initial_stock": sto["initial_stock"]},
        "land_limits": {"enabled": land_on, "limit": land["land_limit"], "beta": land["land_exponent"]},
        "shocks": sh["shocks"] if shock_on else [],
    }


def build_standalone(out: Path = ROOT / "reports" / "grainbridge.html") -> Path:
    """Inline engine.js into index.html and add the document shell for local viewing."""
    page = (TWIN / "index.html").read_text(encoding="utf-8")
    engine = (TWIN / "engine.js").read_text(encoding="utf-8")
    page = page.replace('<script src="engine.js"></script>', f"<script>\n{engine}\n</script>")
    html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head><body>\n" + page + "\n</body></html>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
