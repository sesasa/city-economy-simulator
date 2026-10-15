"""Stage 1 toggleable mechanisms layered on top of `real_economy`.

* investment     — a share s of surplus labor builds capital (SPEC rules 3-5)
* demography     — population grows when fed, shrinks under shortfall (rule 7)
* storage        — unconsumed final goods are stored and spoil (rule 8)
* land_limits    — diminishing returns above land_limit farmers (rule 6)
* harvest_shocks — scheduled multipliers on realised harvests (experiment 5)
"""
from __future__ import annotations

import numpy as np

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.invariants import Invariant, Violation, close
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase
from . import rules
from .real_economy import CAPITAL_HOLDER, NAME as BASE


def _base(world) -> dict:
    """State of whichever base economy is active (planner or market): world.state["economy"]."""
    return world.state["economy"]


# ---------------------------------------------------------------------------
class InvestmentParams(LayerParams):
    investment_share: float = Field(0.3, ge=0, le=1, description="s: share of surplus labor invested")
    rule: Literal["bottleneck", "best_return"] = "bottleneck"
    bottleneck_threshold: float = Field(0.9, gt=0, description="factory utilization that triggers a new factory")
    factory_cost: float = Field(1000.0, gt=0, description="labor-ticks to build one factory")
    land_improvement_cost: float = Field(200.0, gt=0, description="labor-ticks per unit of land improvement")
    tool_cost: float = Field(200.0, gt=0, description="labor-ticks per unit of tools")


class Investment(Layer):
    name = "investment"
    requires = ["planner"]
    Params = InvestmentParams
    aliases = {"s": "investment_share"}
    p: InvestmentParams

    def setup(self, world) -> None:
        world.layer_state(self.name).update(cumulative={k: 0.0 for k in rules.CAPITAL_TYPES})

    def invest(self, world) -> None:
        """Spend s * surplus labor. The rule picks a project; a factory takes only the labor
        needed to bring capacity utilization back under the threshold, and any labor left
        goes to the rule's next pick. Labor no project wants stays leisure."""
        plan, L = world.vars["plan"], world.ledger
        budget = self.p.investment_share * plan.surplus
        spent = {k: 0.0 for k in rules.CAPITAL_TYPES}
        costs = {"factory": self.p.factory_cost, "land_improvement": self.p.land_improvement_cost,
                 "tools_processing": self.p.tool_cost, "tools_distribution": self.p.tool_cost}
        for _ in range(len(rules.CAPITAL_TYPES)):
            if budget <= rules.EPS:
                break
            tech = _base(world)["tech_fn"](world)   # capital after depreciation and earlier spending
            wip = world.stock("factory", "factory_wip")
            ctx = rules.InvestmentContext(
                labor=budget, output=plan.core_target, need=plan.need, tech=tech, costs=costs,
                threshold=self.p.bottleneck_threshold, factory_in_progress=wip > rules.EPS,
                sector_labor={"farming": plan.alloc["farming"],
                              "processing": tech.v_p_eff * plan.core_target,
                              "distribution": plan.alloc["distribution"]})
            choice = rules.INVESTMENT_RULES[self.p.rule](ctx)
            if choice is None or (choice != "factory" and spent[choice] > 0):
                break
            units = budget / costs[choice]
            if choice == "factory":
                wanted = rules.factories_wanted(plan.need, tech.total_capacity, tech.capacity,
                                                self.p.bottleneck_threshold)
                units = min(units, max(wanted, 1.0) - wip)
                L.produce("factory", "factory_wip", units, "factory construction")
                done = int(world.stock("factory", "factory_wip") + 1e-9)
                if done:
                    L.remove("factory", "factory_wip", min(float(done), world.stock("factory", "factory_wip")),
                             "use", "factories completed")
                    L.produce("factory", "factory", float(done), "factories completed")
                    st = _base(world)
                    st["conditions"] = st["conditions"] + [1.0] * done
            else:
                L.produce(CAPITAL_HOLDER[choice], choice, units, "investment")
            labor = units * costs[choice]
            spent[choice] += labor
            budget -= labor
        world.vars["investment_labor"] = sum(spent.values())
        world.vars["investment_costs"] = costs
        world.vars["investment_by_type"] = spent
        cum = world.layer_state(self.name)["cumulative"]
        for k, v in spent.items():
            cum[k] += v

    def hooks(self):
        return {Phase.INVESTMENT: self.invest}

    def invariants(self):
        def paid_for(world, ctx):
            """Every unit of capital built must be matched by investment labor at its cost."""
            costs = world.vars.get("investment_costs")
            spent = world.vars.get("investment_by_type")
            if not costs or spent is None:
                return []
            built = {k: 0.0 for k in costs}
            for p_ in ctx.journal:
                if p_.flow == "production" and p_.note in ("investment", "factory construction"):
                    built["factory" if p_.item == "factory_wip" else p_.item] += p_.amount
            out = []
            for k, units in built.items():
                if not close(units * costs[k], spent[k], spent[k]):
                    out.append(Violation("investment_paid", f"{k}: built {units:.9g} units for {spent[k]:.9g} labor", (k,)))
            return out
        return [Invariant("investment_paid", paid_for)]

    def metrics(self):
        return [Metric("investment_labor", lambda w: w.vars["investment_labor"], "labor building capital", "workers"),
                Metric("investment_by_type", lambda w: w.vars["investment_by_type"], "investment labor by capital type",
                       "workers")]


# ---------------------------------------------------------------------------
class DemographyParams(LayerParams):
    population_growth: float = Field(0.01, ge=0, description="n: growth per tick when fully fed")
    starvation_rate: float = Field(0.1, ge=0, le=1,
                                   description="mu: N' = N(1 - mu * shortfall share) when underfed")


class Demography(Layer):
    name = "demography"
    requires = ["economy"]
    Params = DemographyParams
    aliases = {"n": "population_growth"}
    p: DemographyParams

    def step(self, world) -> None:
        hh = world.groups["households"]
        c = world.vars["tech"].c
        new = rules.next_population(hh.size(), world.vars["consumption_per_person"], c,
                                    self.p.population_growth, self.p.starvation_rate)
        hh.set_size(new)
        world.vars["population_expected"] = new

    def hooks(self):
        return {Phase.DEMOGRAPHY: self.step}


# ---------------------------------------------------------------------------
class StorageParams(LayerParams):
    storage_spoilage: float = Field(0.1, ge=0, le=1, description="share of stored final goods lost per tick")
    storage_target_ticks: float = Field(1.0, ge=0, description="planner keeps this many ticks of consumption in store")
    initial_stock: float = Field(0.0, ge=0)


class Storage(Layer):
    name = "storage"
    requires = ["planner"]
    Params = StorageParams
    p: StorageParams
    ACC = "warehouse"

    def setup(self, world) -> None:
        world.ledger.open_account(self.ACC)
        st = _base(world)
        st["demand_providers"].append(self.restock_demand)
        st["supply_sources"].append(self.draw)
        st["leftover_sinks"].append(self.absorb)
        if self.p.initial_stock:
            world.ledger.produce(self.ACC, "final", self.p.initial_stock, "initial endowment")

    def target(self, world) -> float:
        return self.p.storage_target_ticks * world.groups["households"].size() * _base(world)["c"]

    def restock_demand(self, world) -> float:
        """Units needed so that, after this tick's spoilage, the store is back at target."""
        after_spoil = world.stock(self.ACC, "final") * (1 - self.p.storage_spoilage)
        return max(0.0, self.target(world) - after_spoil)

    def spoil(self, world) -> None:
        lost = world.stock(self.ACC, "final") * self.p.storage_spoilage
        world.ledger.remove(self.ACC, "final", lost, "spoilage", "storage spoilage")
        world.vars["storage_spoiled"] = lost

    def draw(self, world, wanted: float) -> float:
        take = min(wanted, world.stock(self.ACC, "final"))
        world.ledger.transfer(self.ACC, "households", "final", take, "drawn from storage")
        world.vars["storage_drawn"] = take
        return take

    def absorb(self, world, units: float) -> float:
        world.ledger.transfer("factory", self.ACC, "final", units, "into storage")
        world.vars["storage_added"] = units
        return units

    def hooks(self):
        return {Phase.MAINTENANCE: self.spoil}

    def metrics(self):
        return [Metric("storage_stock", lambda w: w.stock(self.ACC, "final"), "final goods in store", "units"),
                Metric("storage_flows", lambda w: {"spoiled": w.vars.get("storage_spoiled", 0.0),
                                                   "drawn": w.vars.get("storage_drawn", 0.0),
                                                   "added": w.vars.get("storage_added", 0.0)},
                       "storage spoilage, draws and additions", "units"),
                Metric("restock_labor", lambda w: w.vars["plan"].restock_labor, "labor producing for storage",
                       "workers")]


# ---------------------------------------------------------------------------
class LandLimitsParams(LayerParams):
    land_limit: float = Field(500.0, gt=0, description="farmers before diminishing returns start")
    land_exponent: float = Field(0.5, ge=0, lt=1, description="beta")


class LandLimits(Layer):
    name = "land_limits"
    requires = ["economy"]
    Params = LandLimitsParams
    aliases = {"beta": "land_exponent"}
    p: LandLimitsParams

    def setup(self, world) -> None:
        st = _base(world)
        st["land_limit"] = self.p.land_limit
        st["land_exponent"] = self.p.land_exponent

    def metrics(self):
        return [Metric("land_pressure", lambda w: w.vars["farmers"] / self.p.land_limit,
                       "farmers / land_limit", "ratio")]


# ---------------------------------------------------------------------------
class Shock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=1)
    duration: int = Field(1, ge=1)
    multiplier: float = Field(0.7, ge=0)


class HarvestShocksParams(LayerParams):
    shocks: list[Shock] = Field(default_factory=list)
    volatility: float = Field(0.0, ge=0, description="random weather: each tick's harvest x lognormal noise with this sd "
                                                     "(mean 1; own random stream, Stage 7)")


class HarvestShocks(Layer):
    """Weather: multiplies realised harvests. The planner does not anticipate it (DECISIONS D-11)."""

    name = "harvest_shocks"
    requires = ["economy"]
    Params = HarvestShocksParams
    p: HarvestShocksParams

    def setup(self, world) -> None:
        self.weather = np.random.default_rng([world.seed, 7001])

    def apply(self, world) -> None:
        m = 1.0
        for s in self.p.shocks:
            if s.start <= world.tick < s.start + s.duration:
                m *= s.multiplier
        sd = self.p.volatility
        if sd > 0:
            m *= float(np.exp(sd * self.weather.standard_normal() - sd * sd / 2))
        world.vars["harvest_multiplier"] = m

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.apply}

    def metrics(self):
        return [Metric("harvest_multiplier", lambda w: w.vars["harvest_multiplier"], "weather multiplier", "x")]
