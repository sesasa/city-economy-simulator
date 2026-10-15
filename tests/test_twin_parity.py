"""The Grainbridge twin (twin/engine.js) must follow the same rules as the Python model.

Runs the JS engine under node for each Stage 1 scenario (and a few variants) and compares
every tick. Skipped when node is not installed.
"""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from citysim.config import load_scenario, with_params
from citysim.core import runner
from citysim.twin import TWIN, scenario_to_twin
from tests.conftest import SCENARIOS

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

CASES = {
    "base": ("stage1_base.yaml", {}),
    "below_n_min": ("stage1_base.yaml", {"N0": 120, "ticks": 400}),
    "collapse_max_feasible": ("stage1_base.yaml", {"N0": 120, "shortage_rule": "max_feasible",
                                                   "demography.enabled": True, "ticks": 300}),
    "shock_storage": ("stage1_shock_storage.yaml", {}),
    "shock_storage_demography": ("stage1_shock_storage.yaml", {"demography.enabled": True}),
    "shock_no_storage": ("stage1_shock_no_storage.yaml", {}),
    "growth_s0": ("stage1_base.yaml", {"s": 0.0, "demography.enabled": True, "ticks": 400}),
    "best_return": ("stage1_base.yaml", {"investment.rule": "best_return", "N0": 300}),
    "long_run": ("stage1_long_run.yaml", {}),
    "conversion_1_3": ("stage1_base.yaml", {"e": 1.3, "storage.enabled": True}),
    "conversion_0_8": ("stage1_base.yaml", {"e": 0.8, "a": 6, "demography.enabled": True}),
    "hungry_people": ("stage1_base.yaml", {"c": 1.5, "a": 8, "demography.enabled": True}),
    "custom_priority": ("stage1_base.yaml", {"N0": 110, "priority": ["maintenance", "farming", "processing", "distribution"]}),
    "overlapping_shocks": ("stage1_shock_storage.yaml", {"demography.enabled": True, "harvest_shocks.shocks": [
        {"start": 60, "duration": 3, "multiplier": 0.7}, {"start": 61, "duration": 4, "multiplier": 0.6}]}),
    "town_dies": ("stage1_base.yaml", {"N0": 60, "demography.enabled": True, "ticks": 300}),
}
COLS = ["population", "consumption_per_person", "surplus_share", "labor_shortfall", "total_output",
        "raw_output", "potential_output", "investment_labor", "factories", "capital/factory",
        "capital/land_improvement", "capital/tools_processing", "capital/tools_distribution",
        "capital/factory_wip", "unmaintained_share", "effective_farm_productivity",
        "labor_share/farming", "labor_share/maintenance"]


def js_run(sc, tmp_path):
    f = tmp_path / "params.json"
    f.write_text(json.dumps(scenario_to_twin(sc)))
    out = subprocess.run([NODE, str(TWIN / "parity_runner.js"), str(f), str(sc.ticks)],
                         capture_output=True, text=True, check=True)
    return pd.DataFrame(json.loads(out.stdout)).set_index("tick")


@pytest.mark.parametrize("case", list(CASES))
def test_twin_matches_python(case, tmp_path):
    path, kw = CASES[case]
    sc = with_params(load_scenario(SCENARIOS / path), **kw)
    py = runner.run(sc).wide()
    js = js_run(sc, tmp_path)
    cols = COLS + (["storage_stock"] if "storage_stock" in py else [])
    worst = {}
    for c in cols:
        a, b = py[c].to_numpy(float), js[c].to_numpy(float)
        # labor quantities are differences of population-sized numbers: scale by population
        scale = py["population"].to_numpy(float) if c in ("labor_shortfall", "investment_labor") else np.abs(a)
        assert (np.isnan(a) == np.isnan(b)).all(), f"{case}: NaN pattern differs in {c}"
        rel = np.abs(a - b) / np.maximum(1.0, scale)
        worst[c] = float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0
    bad = {c: v for c, v in worst.items() if v > 1e-9}
    assert not bad, f"twin diverges from the model in {case}: {bad}"


def test_standalone_build_inlines_engine(tmp_path):
    from citysim.twin import build_standalone
    out = build_standalone(tmp_path / "grainbridge.html")
    html = out.read_text()
    assert html.startswith("<!doctype html>") and "class Town" in html and 'src="engine.js"' not in html
