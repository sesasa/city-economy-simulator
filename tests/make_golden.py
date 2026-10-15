"""Write golden outputs for the regression test (run deliberately, never from tests)."""
from citysim.config import load_scenario
from citysim.core import runner
from citysim.core.metrics import write_frame

GOLDEN = ["stage1_base", "stage1_long_run", "stage1_shock_storage", "stage1_shock_no_storage", "stage2_market", "stage3_bank", "stage4_competition", "stage5_government", "stage6_central_bank", "stage7_markets", "stage9_plateau", "stage9_market", "stage10_modern", "stage10_planned", "stage11_town", "stage11_internet"]

REGION_GOLDEN = ["stage8_two_cities"]          # region runs (Stage 8+): tidy frame with a "city" column


def region_frame(name: str, ticks: int = 150):
    import pandas as pd
    from citysim.region import load_region, run_region, with_region
    from tests.conftest import SCENARIOS
    res = run_region(with_region(load_region(SCENARIOS / f"{name}.yaml"), ticks=ticks))
    parts = [res.region_frame.assign(city="")] + [f.assign(city=c) for c, f in res.frames.items()]
    return pd.concat(parts, ignore_index=True)[["city", "tick", "metric", "group", "value"]]


if __name__ == "__main__":
    from tests.conftest import ROOT, SCENARIOS
    for name in GOLDEN:
        res = runner.run(load_scenario(SCENARIOS / f"{name}.yaml"))
        print(name, write_frame(res.frame, ROOT / "tests" / "golden" / f"{name}.csv"))
    for name in REGION_GOLDEN:
        print(name, write_frame(region_frame(name), ROOT / "tests" / "golden" / f"{name}.csv"))
