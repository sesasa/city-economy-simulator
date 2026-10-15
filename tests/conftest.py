from pathlib import Path

import pytest

from citysim.config import load_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


@pytest.fixture
def base():
    return load_scenario(SCENARIOS / "stage1_base.yaml")
