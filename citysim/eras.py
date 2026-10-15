"""Data for "Grainbridge through the ages" (twin/eras.html): one recorded pair of runs per stage.

Each era plays back a Python run of that stage's economy ("with" the new institution or event)
next to a run without it, so the page can show what the new layer changes. Everything comes
from the real model with all invariants checked; the page only draws it.

    python -m citysim eras        # writes twin/eras_data.js
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_scenario, with_params
from .core.runner import run
from .region import load_region, run_region, with_region

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios"
STEP = 2            # keep every 2nd tick
SECTORS = ["farm", "factory", "distribution", "maintenance", "services", "public", "research", "building"]


def pack(w: pd.DataFrame, cols: dict[str, str], ticks: int) -> dict:
    w = w.loc[1:ticks:STEP]
    out = {}
    for key, c in cols.items():
        s = w[c] if c in w else pd.Series(0.0, index=w.index)
        out[key] = [round(float(x), 4) if np.isfinite(x) else 0.0 for x in s.fillna(0.0)]
    jobs = {}
    for s in SECTORS:
        c = f"labor_share/{s}" if f"labor_share/{s}" in w else f"labor_share/{ {'farm': 'farming', 'factory': 'processing'}.get(s, s)}"
        if c in w:
            jobs[s] = [round(float(x), 4) for x in w[c].fillna(0.0)]
    out["jobs"] = jobs
    return out


BASE = {"bread": "consumption_per_person", "services": "services_per_person", "jobless": "unemployment_rate",
        "hungry": "hungry_share", "people": "population", "price": "price_level"}


def town(path, ticks, extra=None, **kw):
    w = run(with_params(load_scenario(SCEN / path), ticks=ticks, **kw)).wide()
    return pack(w, {**BASE, **(extra or {})}, ticks)


def eras() -> list[dict]:
    out = []
    # 2 · money and prices: a drought hits a market town; prices carry the news
    drought = {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=100, duration=6, multiplier=0.4)]}
    out.append(dict(stage=2, key="money", ticks=220, event=100,
                    a=town("stage2_market.yaml", 220, **drought), b=town("stage2_market.yaml", 220),
                    a_label="drought in year 100", b_label="no drought"))
    # 3 · a bank: an optimistic lender, boom and bust
    grow = {"demography.enabled": True, "land_limits.enabled": True}
    bank = {"loans": "loans", "defaults": "defaults", "failure": "bank_failure", "run": "bank_run"}
    out.append(dict(stage=3, key="bank", ticks=400, event=None,
                    a=town("stage3_bank.yaml", 400, bank, investment_share=0.05, procyclicality=0.1, **grow),
                    b=town("stage3_bank.yaml", 400, bank, investment_share=0.05, procyclicality=0.0, **grow),
                    a_label="bank loosens its standards in good times", b_label="cautious bank"))
    # 4 · many firms and services
    comp = {"firms": "firms/farm", "gini": "gini_income"}
    out.append(dict(stage=4, key="firms", ticks=300, event=None,
                    a=town("stage4_competition.yaml", 300, comp), b=town("stage4_competition.yaml", 300, comp, **{"services.enabled": False}),
                    a_label="shops sell services too", b_label="bread only"))
    # 5 · government: a safety net as families grow
    gov = {"gini": "gini_income", "debt": "public_debt"}
    out.append(dict(stage=5, key="government", ticks=300, event=None,
                    a=town("stage5_government.yaml", 300, gov, basic_income=0.2, **{"demography.enabled": True}),
                    b=town("stage5_government.yaml", 300, gov, **{"government.enabled": False, "demography.enabled": True}),
                    a_label="taxes pay a basic income", b_label="no government"))
    # 6 · central bank: a sudden wish to save
    slump = {"demand_shocks.enabled": True, "demand_shocks.shocks": [dict(start=150, duration=30, multiplier=0.5)]}
    cb = {"rate": "policy_rate"}
    out.append(dict(stage=6, key="central_bank", ticks=260, event=150,
                    a=town("stage6_central_bank.yaml", 260, cb, seed=1, **{"central_bank.lower_bound": -0.05}, **slump),
                    b=town("stage6_central_bank.yaml", 260, cb, seed=1, **{"central_bank.rule": "fixed_rate"}, **slump),
                    a_label="central bank cuts the interest rate", b_label="rate never moves"))
    # 7 · stock market: borrowed money and a bubble
    sm = {"share": "share_price", "value": "fundamental_value", "margin": "margin_debt", "fire": "forced_sales"}
    out.append(dict(stage=7, key="stocks", ticks=400, event=None,
                    a=town("stage7_markets.yaml", 400, sm, seed=2, margin_requirement=0.2),
                    b=town("stage7_markets.yaml", 400, sm, seed=2, margin_requirement=1.0),
                    a_label="buy shares with 80% borrowed", b_label="no borrowing for shares"))
    # 8 · two towns: Grainbridge's harvest fails
    def region(**kw):
        w = run_region(with_region(load_region(SCEN / "stage8_two_cities.yaml"), seed=1, ticks=240, **kw)).wide()
        cols = {}
        for c in ("grainbridge", "millford"):
            for k, m in BASE.items():
                cols[f"{c}.{k}"] = f"{c}.{m}"
        cols["rate"] = "exchange_rate/grainbridge"
        ship = [x for x in w.columns if x.startswith("shipments")]
        w["shipped"] = w[ship].sum(axis=1) if ship else 0.0
        cols["shipped"] = "shipped"
        return pack(w, cols, 240)
    fail = {"grainbridge.harvest_shocks.enabled": True,
            "grainbridge.harvest_shocks.shocks": [dict(start=150, duration=10, multiplier=0.4)]}
    out.append(dict(stage=8, key="trade", ticks=240, event=150, a=region(**fail), b=region(currency="union", **fail),
                    a_label="each town has its own money", b_label="one money for both towns"))
    # 9 · technology
    tech = {"tech": "tech_level", "wage": "real_wage"}
    out.append(dict(stage=9, key="technology", ticks=600, event=None,
                    a=town("stage9_market.yaml", 600, tech, seed=1),
                    b=town("stage9_market.yaml", 600, tech, seed=1, **{"technology.enabled": False}),
                    a_label="3% of people do research", b_label="no research"))
    # 10 · systems
    sysc = {"tech": "tech_level", "gini": "gini_income"}
    out.append(dict(stage=10, key="systems", ticks=600, event=300,
                    a=town("stage10_modern.yaml", 600, sysc, seed=1, **_harvest(300)),
                    b=town("stage10_modern.yaml", 600, sysc, seed=1, **{"government.enabled": False, "central_bank.enabled": False,
                                                                          "risk_sharing.enabled": False}, **_harvest(300)),
                    c=town("stage10_planned.yaml", 600, sysc, **_harvest(300)),
                    a_label="mixed economy", b_label="free market", c_label="central planning"))
    return out


def _harvest(t):
    return {"harvest_shocks.enabled": True, "harvest_shocks.shocks": [dict(start=t, duration=5, multiplier=0.5)]}


def build(out: Path = ROOT / "twin" / "eras_data.js") -> Path:
    data = eras()
    out.write_text("window.ERAS = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
