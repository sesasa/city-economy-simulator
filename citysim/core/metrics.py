"""Metric registry and time-series recorder (tidy rows: tick, metric, group, value)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING, Union

import pandas as pd

if TYPE_CHECKING:
    from .world import World

MetricValue = Union[float, dict[str, float]]


@dataclass
class Metric:
    """A named observable. `fn` returns a scalar, or a {group: value} dict."""

    name: str
    fn: Callable[["World"], MetricValue]
    description: str = ""
    unit: str = ""


class Recorder:
    def __init__(self, metrics: list[Metric]):
        names = [m.name for m in metrics]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate metric names: {sorted(dupes)}")
        self.metrics = metrics
        self._rows: list[tuple[int, str, str, float]] = []

    def record(self, world: "World") -> None:
        t = world.tick
        rows = self._rows
        for m in self.metrics:
            v = m.fn(world)
            if isinstance(v, dict):
                for g, x in v.items():
                    rows.append((t, m.name, g, float(x)))
            else:
                rows.append((t, m.name, "", float(v)))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=["tick", "metric", "group", "value"])

    def catalog(self) -> pd.DataFrame:
        return pd.DataFrame([(m.name, m.unit, m.description) for m in self.metrics],
                            columns=["metric", "unit", "description"])


def write_frame(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, float_format="%.17g")
    return path


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a tidy frame to one column per metric[/group], indexed by tick."""
    key = df["metric"].where(df["group"] == "", df["metric"] + "/" + df["group"])
    return df.assign(key=key).pivot(index="tick", columns="key", values="value")
