"""Stage 1 behavioural rules as pure, documented, swappable functions.

Nothing in here touches the world or the ledger; the layers call these and
then post the resulting flows. Symbols follow SPEC.md:
a (farm output per worker), c (consumption per person), e (final units per raw unit),
F_p / F_m (fixed factory / maintenance crews), v_p / v_d (variable labor per final unit).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import accumulate
from typing import Callable, Optional

SECTORS = ("farming", "processing", "distribution", "maintenance")
CAPITAL_TYPES = ("factory", "land_improvement", "tools_processing", "tools_distribution")
EPS = 1e-12


# ---------------------------------------------------------------------------
# Technology: what the current capital stock lets the city do
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tech:
    c: float
    e: float
    a: float                              # nominal farm output per worker (before capital, land, shocks)
    land_improvement: float = 0.0         # L, units of land improvement
    tools_processing: float = 0.0         # T_p
    tools_distribution: float = 0.0       # T_d
    land_elasticity: float = 0.25         # a_x = a (1+L)^theta_L
    tool_elasticity: float = 0.25         # v_eff = v (1+T)^-theta_T
    v_p: float = 0.1
    v_d: float = 0.1
    F_p: float = 50.0
    F_m: float = 30.0
    capacity: float = 1000.0              # final units per factory per tick at condition 1
    conditions: tuple[float, ...] = (1.0,)  # one entry per factory, in [0, 1]
    land_maintenance: float = 1.0         # workers per unit of land improvement per tick
    tool_maintenance: float = 1.0         # workers per unit of tools per tick
    land_limit: Optional[float] = None    # None: no diminishing returns
    land_exponent: float = 0.5            # beta

    # derived quantities -------------------------------------------------
    @property
    def a_x(self) -> float:
        """Farm output per worker after land improvements (before land limits)."""
        return self.a * (1.0 + self.land_improvement) ** self.land_elasticity

    @property
    def v_p_eff(self) -> float:
        return self.v_p * (1.0 + self.tools_processing) ** (-self.tool_elasticity)

    @property
    def v_d_eff(self) -> float:
        return self.v_d * (1.0 + self.tools_distribution) ** (-self.tool_elasticity)

    @property
    def total_capacity(self) -> float:
        return self.capacity * sum(self.conditions)

    @cached_property
    def cumulative_capacity(self) -> list[float]:
        """Capacity of the best k factories, k = 1..n (best condition first)."""
        return [self.capacity * x for x in accumulate(sorted(self.conditions, reverse=True))]

    def with_capital(self, kind: str, delta: float) -> "Tech":
        """Tech after adding `delta` units of a capital type (used to evaluate investments)."""
        if kind == "factory":
            return replace(self, conditions=self.conditions + (delta,))
        return replace(self, **{kind: getattr(self, kind) + delta})


# ---------------------------------------------------------------------------
# Farming with optional land limits (SPEC rule 6)
# ---------------------------------------------------------------------------

def farm_output(farmers: float, a_x: float, land_limit: Optional[float], beta: float) -> float:
    """Raw output. Above `land_limit` farmers, productivity is a_x * (land_limit/farmers)^beta."""
    if farmers <= 0:
        return 0.0
    if land_limit is None or farmers <= land_limit:
        return a_x * farmers
    return a_x * land_limit ** beta * farmers ** (1.0 - beta)


def farmers_required(raw: float, a_x: float, land_limit: Optional[float], beta: float) -> float:
    """Inverse of farm_output: farmers needed to harvest `raw` units."""
    if raw <= 0:
        return 0.0
    if a_x <= 0:
        return math.inf
    if land_limit is None or raw <= a_x * land_limit:
        return raw / a_x
    if beta >= 1.0:
        return math.inf
    return land_limit * (raw / (a_x * land_limit)) ** (1.0 / (1.0 - beta))


def effective_farm_productivity(farmers: float, a_x: float, land_limit: Optional[float], beta: float) -> float:
    if land_limit is None or farmers <= land_limit:
        return a_x
    return a_x * (land_limit / farmers) ** beta


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def operating_factories(output: float, conditions: tuple[float, ...], capacity: float,
                        cumulative: Optional[list[float]] = None) -> tuple[int, float]:
    """Run the fewest factories (best condition first) that cover `output`.

    Returns (number operated, their combined capacity). If output exceeds the
    total capacity, every factory runs. `cumulative` may pass a precomputed
    Tech.cumulative_capacity.
    """
    if output <= EPS or not conditions:
        return 0, 0.0
    cum = cumulative if cumulative is not None else \
        [capacity * x for x in accumulate(sorted(conditions, reverse=True))]
    k = bisect.bisect_left(cum, output * (1 - 1e-12))
    if k >= len(cum):
        return len(cum), cum[-1]
    return k + 1, cum[k]


def capacity_of_best(k: int, conditions: tuple[float, ...], capacity: float) -> float:
    return capacity * sum(sorted(conditions, reverse=True)[:k])


# ---------------------------------------------------------------------------
# Labor requirements (SPEC rule 1)
# ---------------------------------------------------------------------------

def maintenance_required(tech: Tech) -> float:
    """F_m per existing factory (idle ones too), plus per-unit upkeep of land
    improvements and tools (DECISIONS D-07)."""
    return (tech.F_m * len(tech.conditions)
            + tech.land_maintenance * tech.land_improvement
            + tech.tool_maintenance * (tech.tools_processing + tech.tools_distribution))


def production_labor(output: float, tech: Tech) -> tuple[float, float]:
    """(farmers, factory workers) needed to produce `output` final units."""
    farmers = farmers_required(output / tech.e, tech.a_x, tech.land_limit, tech.land_exponent)
    k, _ = operating_factories(output, tech.conditions, tech.capacity, tech.cumulative_capacity)
    return farmers, tech.F_p * k + tech.v_p_eff * output


def labor_requirements(output: float, tech: Tech, distributed: Optional[float] = None) -> dict[str, float]:
    """Workers each sector needs to produce `output` and distribute `distributed` (default: all of it)."""
    farmers, factory = production_labor(output, tech)
    return {
        "farming": farmers,
        "processing": factory,
        "distribution": tech.v_d_eff * (output if distributed is None else distributed),
        "maintenance": maintenance_required(tech),
    }


def total_labor(output: float, tech: Tech) -> float:
    return sum(labor_requirements(output, tech).values())


# ---------------------------------------------------------------------------
# Shortage allocation (SPEC rule 2) — swappable
# ---------------------------------------------------------------------------

def allocate_priority(required: dict[str, float], workers: float, order: list[str]) -> dict[str, float]:
    """Fill sectors fully, one at a time, in `order` until workers run out."""
    left = workers
    alloc: dict[str, float] = {}
    for s in order:
        give = min(required[s], max(left, 0.0))
        alloc[s] = give
        left -= give
    return alloc


def potential_output(workers: float, tech: Tech, iters: int = 60) -> float:
    """Largest final output the city could sustain with all its labor and capital
    fully maintained and no investment (capped by factory capacity)."""
    cap = tech.total_capacity
    if total_labor(0.0, tech) > workers or cap <= 0:
        return 0.0
    if total_labor(cap, tech) <= workers:
        return cap
    lo, hi = 0.0, cap
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if total_labor(mid, tech) <= workers:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    workers: float                     # population at planning time
    need: float                        # N * c
    core_target: float                 # final units planned for consumption
    restock: float                     # extra final units planned for storage
    required: dict[str, float]         # labor needed for core_target (+ restock)
    alloc: dict[str, float]            # labor assigned per sector
    surplus: float                     # unassigned labor (investment + leisure)
    shortfall: float                   # required - available, >= 0
    factories_operated: int
    restock_labor: float = 0.0
    population: float = -1.0           # people fed, when some workers are elsewhere (research, Stage 9)

    @property
    def people(self) -> float:
        return self.population if self.population >= 0 else self.workers

    @property
    def target(self) -> float:
        return self.core_target + self.restock


def _max_extra(base: float, limit: float, budget: float, tech: Tech, iters: int = 60) -> tuple[float, float]:
    """Largest extra output x <= limit whose extra production labor fits in `budget`."""
    if limit <= EPS or budget <= EPS:
        return 0.0, 0.0
    f0, p0 = production_labor(base, tech)

    def extra(x: float) -> float:
        f, p = production_labor(base + x, tech)
        return (f - f0) + (p - p0)

    if extra(limit) <= budget:
        return limit, extra(limit)
    lo, hi = 0.0, limit
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if extra(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo, extra(lo)


def make_plan(workers: float, tech: Tech, *, order: list[str], shortage_rule: str = "priority",
              restock_demand: float = 0.0, eaters: float | None = None) -> Plan:
    """Central planner (SPEC rules 1-3).

    1. Target consumption output N*c, capped by factory capacity
       (`max_feasible` additionally caps it at what labor can sustain).
    2. Compute labor requirements; if they exceed N, allocate by priority `order`.
    3. If labor is left, produce extra final goods for storage (restock), then
       the remainder is the surplus (split into investment and leisure later).
    """
    need = (workers if eaters is None else eaters) * tech.c
    pop = -1.0 if eaters is None else eaters
    core = min(need, tech.total_capacity)
    if shortage_rule == "max_feasible":
        core = min(core, potential_output(workers, tech))
    elif shortage_rule != "priority":
        raise ValueError(f"unknown shortage_rule {shortage_rule!r}")

    required = labor_requirements(core, tech)
    total_req = sum(required.values())
    if total_req > workers * (1 + 1e-12):
        alloc = allocate_priority(required, workers, order)
        k, _ = operating_factories(core, tech.conditions, tech.capacity, tech.cumulative_capacity)
        return Plan(workers, need, core, 0.0, required, alloc, 0.0,
                    max(total_req - workers, 0.0), k, population=pop)

    alloc = dict(required)
    surplus = workers - total_req
    restock, rl = _max_extra(core, min(restock_demand, tech.total_capacity - core), surplus, tech)
    if restock > 0:
        f, p = production_labor(core + restock, tech)
        alloc["farming"], alloc["processing"] = f, p
        required = dict(alloc)
        surplus = workers - sum(alloc.values())
    k, _ = operating_factories(core + restock, tech.conditions, tech.capacity, tech.cumulative_capacity)
    return Plan(workers, need, core, restock, required, alloc, max(surplus, 0.0), 0.0, k, rl, population=pop)


# ---------------------------------------------------------------------------
# Investment allocation (SPEC rules 4-5) — swappable
# ---------------------------------------------------------------------------

@dataclass
class InvestmentContext:
    labor: float                         # labor-ticks available for investment this tick
    output: float                        # planned final output
    need: float                          # N * c
    tech: Tech
    costs: dict[str, float]              # labor-ticks per unit of each capital type
    threshold: float                     # factory-capacity utilization that triggers building
    factory_in_progress: bool = False
    sector_labor: dict[str, float] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        cap = self.tech.total_capacity
        return math.inf if cap <= 0 else self.need / cap

    def net_saving(self, kind: str) -> float:
        """Labor saved per tick (net of new upkeep) by putting this tick's labor into `kind`."""
        units = self.labor / self.costs[kind]
        before = total_labor(self.output, self.tech)
        after = total_labor(self.output, self.tech.with_capital(kind, units))
        return before - after


def factories_wanted(need: float, total_capacity: float, capacity: float, threshold: float) -> float:
    """Whole factories to build so that need / capacity drops below `threshold`."""
    gap = need / threshold - total_capacity
    return 0.0 if gap <= 0 else float(math.floor(gap / capacity) + 1)


# Which labor-saving capital relieves which sector.
RELIEVES = {"land_improvement": "farming", "tools_processing": "processing",
            "tools_distribution": "distribution"}


def invest_bottleneck(ctx: InvestmentContext) -> Optional[str]:
    """Invest in the sector closest to its limit.

    * Factory capacity is a hard limit: if demand / capacity >= threshold (or a
      factory is half built) build factory capacity.
    * Otherwise labor is the binding limit, so relieve the sector that uses the
      most labor, provided the capital saves more labor than its own upkeep.
      If none does, the investment labor stays idle (becomes leisure).
    """
    if ctx.factory_in_progress or ctx.utilization >= ctx.threshold:
        return "factory"
    ranked = sorted(RELIEVES, key=lambda k: -ctx.sector_labor.get(RELIEVES[k], 0.0))
    for kind in ranked:
        if ctx.net_saving(kind) > EPS:
            return kind
    return None


def invest_best_return(ctx: InvestmentContext) -> Optional[str]:
    """Alternative: same factory trigger, otherwise the capital with the largest net labor saving."""
    if ctx.factory_in_progress or ctx.utilization >= ctx.threshold:
        return "factory"
    best = max(RELIEVES, key=ctx.net_saving)
    return best if ctx.net_saving(best) > EPS else None


INVESTMENT_RULES: dict[str, Callable[[InvestmentContext], Optional[str]]] = {
    "bottleneck": invest_bottleneck,
    "best_return": invest_best_return,
}


# ---------------------------------------------------------------------------
# Depreciation and demography
# ---------------------------------------------------------------------------

def unmaintained_share(maintenance_labor: float, maintenance_required_: float) -> float:
    """Share of capital left unmaintained this tick."""
    if maintenance_required_ <= EPS:
        return 0.0
    return min(1.0, max(0.0, 1.0 - maintenance_labor / maintenance_required_))


def next_population(n: float, consumption_per_person: float, c: float,
                    growth: float, starvation_rate: float) -> float:
    """SPEC rule 7: grow at `growth` when fully fed, otherwise shrink in proportion
    to the shortfall: N' = N * (1 - starvation_rate * (1 - cons/c))."""
    if consumption_per_person >= c * (1 - 1e-9):
        return n * (1.0 + growth)
    shortfall = 1.0 - consumption_per_person / c
    return n * max(0.0, 1.0 - starvation_rate * shortfall)


def analytical_n_min(a: float, c: float, e: float, F_p: float, F_m: float, v_p: float, v_d: float) -> float:
    """SPEC benchmark: N_min = (F_p + F_m) / (1 - c/(e a) - v_p c - v_d c); inf if not viable."""
    denom = 1.0 - c / (e * a) - v_p * c - v_d * c
    return math.inf if denom <= 0 else (F_p + F_m) / denom


def analytical_a_min(n: float, c: float, e: float, F_p: float, F_m: float, v_p: float, v_d: float) -> float:
    """Minimum farm productivity for a city of n people (inverse of the N_min formula)."""
    denom = 1.0 - v_p * c - v_d * c - (F_p + F_m) / n
    return math.inf if denom <= 0 else c / (e * denom)
