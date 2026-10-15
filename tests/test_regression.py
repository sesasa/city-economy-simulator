"""Golden outputs: Stage 1 base scenarios at a fixed seed. Later stages, with their layers
off, must reproduce these exactly. Regenerate only deliberately: python -m tests.make_golden"""
import numpy as np
import pandas as pd
import pytest

from citysim.config import load_scenario
from citysim.core import runner
from tests.conftest import ROOT, SCENARIOS
from tests.make_golden import GOLDEN, REGION_GOLDEN, region_frame

GOLDEN_DIR = ROOT / "tests" / "golden"


@pytest.mark.parametrize("name", list(GOLDEN))
def test_golden(name):
    expected = pd.read_csv(GOLDEN_DIR / f"{name}.csv", keep_default_na=False)
    sc = load_scenario(SCENARIOS / f"{name}.yaml")
    got = runner.run(sc).frame
    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected)
    assert (got.metric.values == expected.metric.values).all()
    assert (got.group.values == expected.group.astype(str).values).all()
    pd.testing.assert_series_equal(got.value, expected.value, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", list(REGION_GOLDEN))
def test_region_golden(name):
    expected = pd.read_csv(GOLDEN_DIR / f"{name}.csv", keep_default_na=False)
    actual = region_frame(name)
    assert len(actual) == len(expected)
    assert list(actual["metric"]) == list(expected["metric"])
    np.testing.assert_allclose(actual["value"].to_numpy(float), expected["value"].to_numpy(float), rtol=1e-9, atol=1e-9)
