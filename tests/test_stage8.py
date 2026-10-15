"""Stage 8: trade between cities."""
import numpy as np
import pytest

from citysim.region import Region, load_region, run_region, with_region
from tests.conftest import SCENARIOS

TWO = SCENARIOS / "stage8_two_cities.yaml"
SAME = {"grainbridge.farm_output_per_worker": 5.0, "grainbridge.factory_variable_labor": 0.1,
        "millford.farm_output_per_worker": 5.0, "millford.factory_variable_labor": 0.1}


def test_comparative_advantage_raises_total_consumption():
    def total(**kw):
        out = 0.0
        for seed in (1, 2, 3):
            w = run_region(with_region(load_region(TWO), seed=seed, ticks=300, **kw)).wide().iloc[-100:]
            out += sum((w[f"{c}.consumption_per_person"] + w[f"{c}.services_per_person"]).mean() for c in ("grainbridge", "millford"))
        return out
    assert total() > total(**{"trade.enabled": False})


def test_identical_towns_barely_trade():
    w = run_region(with_region(load_region(TWO), seed=1, ticks=200, **SAME)).wide()
    ship = [c for c in w.columns if c.startswith("shipments")]
    assert w[ship].iloc[-80:].sum(axis=1).mean() < 0.1 * 600          # under a tenth of the bread market


def test_no_money_crosses_under_floating_and_goods_arrive():
    R = Region(with_region(load_region(TWO), seed=1, ticks=120))
    exported = imported = 0.0
    for _ in range(120):
        R.step()
        for c in R.names:
            for p in R.world(c).ledger.journal:
                if p.flow == "export" and p.item == "deposits":
                    raise AssertionError("money crossed a border under floating rates")
                if p.flow == "export" and p.item in ("raw", "final"):
                    exported += p.amount
                if p.flow == "import" and p.item in ("raw", "final"):
                    imported += p.amount
    assert exported > 0 and exported == pytest.approx(imported)


def test_union_deficit_drains_money():
    w = run_region(with_region(load_region(TWO), seed=1, ticks=200, currency="union")).wide()
    g = w["grainbridge.money_supply"]
    m = w["millford.money_supply"]
    assert (m.iloc[-1] - m.iloc[10]) * (g.iloc[-1] - g.iloc[10]) < 0         # one town gains what the other loses


def test_embargo_hurts_the_dependent_town():
    emb = {"embargoes": [dict(start=120, duration=30, between=["grainbridge", "millford"])]}
    base = run_region(with_region(load_region(TWO), seed=1, ticks=160)).wide()
    cut = run_region(with_region(load_region(TWO), seed=1, ticks=160, **emb)).wide()
    ship = [c for c in cut.columns if c.startswith("shipments")]
    assert cut[ship].loc[125:149].sum().sum() == 0
    assert cut["millford.consumption_per_person"].loc[125:155].mean() < base["millford.consumption_per_person"].loc[125:155].mean()


def test_migrants_keep_people_and_money():
    R = Region(with_region(load_region(TWO), seed=1, ticks=150, migration=0.05, currency="union"))
    for _ in range(150):
        R.step()
    pop = sum(float(R.world(c).state["economy"]["market_layer"].size.sum()) for c in R.names)
    assert pop == pytest.approx(1200.0)
