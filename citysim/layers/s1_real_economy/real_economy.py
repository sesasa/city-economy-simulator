"""Stage 1 base layer: four sectors run by a central planner, no money.

Ledger accounts (holders): farm, factory, distribution, maintenance, households.
Items: raw, final (goods); factory, factory_wip, land_improvement,
tools_processing, tools_distribution (capital).

Per tick:
  Phase 1  plan: target output, labor requirements, allocation, surplus
  Phase 2  farming, then processing
  Phase 3  maintenance; unmaintained capital depreciates
  Phase 5  distribution and consumption; unused raw and final goods are wasted
           (or handed to a registered sink such as the storage layer)

Extension points other layers use (in world.state['real_economy']):
  demand_providers : [fn(world) -> extra final units wanted this tick]  (storage restock)
  supply_sources   : [fn(world, wanted) -> units delivered to households] (storage draw)
  leftover_sinks   : [fn(world, units) -> units absorbed from factory leftovers] (storage)
  land_limit / land_exponent : diminishing returns (set by the land_limits layer)

world.vars written each tick: plan, tech, harvest_multiplier (read), raw_output,
final_output, unmaintained_share, delivered, consumption_per_person,
investment_labor (written by the investment layer), wasted_raw, wasted_final.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from ...core.agents import make_group
from ...core.invariants import Invariant, Violation, close
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase
from . import rules
from .rules import SECTORS, Tech

NAME = "real_economy"
HOLDER = {"farming": "farm", "processing": "factory", "distribution": "distribution",
          "maintenance": "maintenance"}
CAPITAL_HOLDER = {"factory": "factory", "factory_wip": "factory", "land_improvement": "farm",
                  "tools_processing": "factory", "tools_distribution": "distribution"}


class RealEconomyParams(LayerParams):
    farm_output_per_worker: float = Field(5.0, gt=0, description="a: raw units per farmer per tick")
    consumption_per_person: float = Field(1.0, gt=0, description="c: final units each person needs per tick")
    conversion_rate: float = Field(1.0, gt=0, description="e: final units per raw unit")
    factory_fixed_crew: float = Field(50.0, ge=0, description="F_p: workers to run one factory at all")
    factory_variable_labor: float = Field(0.1, ge=0, description="v_p: workers per final unit processed")
    distribution_labor: float = Field(0.1, ge=0, description="v_d: workers per final unit distributed")
    maintenance_fixed_crew: float = Field(30.0, ge=0, description="F_m: workers to maintain one factory")
    depreciation_rate: float = Field(0.05, ge=0, le=1, description="delta: share of unmaintained capital lost per tick")
    factory_capacity: float = Field(1000.0, gt=0, description="max final units per factory per tick")
    initial_population: float = Field(150.0, ge=0, description="N0")
    initial_factories: int = Field(1, ge=0)
    scrap_condition: float = Field(0.01, ge=0, lt=1, description="factories below this condition are scrapped")
    # capital technology (DECISIONS D-08)
    land_improvement_elasticity: float = Field(0.25, ge=0, description="a_x = a (1+L)^theta")
    tool_elasticity: float = Field(0.25, ge=0, description="v_eff = v (1+T)^-theta")
    land_improvement_maintenance: float = Field(1.0, ge=0, description="workers per unit of land improvement")
    tool_maintenance: float = Field(1.0, ge=0, description="workers per unit of tools")
    # planner
    priority: list[Literal["farming", "processing", "distribution", "maintenance"]] = \
        Field(default_factory=lambda: list(SECTORS))
    shortage_rule: Literal["priority", "max_feasible"] = "priority"
    representation: dict[str, Literal["aggregate", "agents"]] = Field(
        default_factory=lambda: {s: "aggregate" for s in (*SECTORS, "households")})


class RealEconomy(Layer):
    name = NAME
    requires: list[str] = []
    provides = ["economy", "planner"]
    Params = RealEconomyParams
    aliases = {"a": "farm_output_per_worker", "c": "consumption_per_person", "e": "conversion_rate",
               "F_p": "factory_fixed_crew", "v_p": "factory_variable_labor", "v_d": "distribution_labor",
               "F_m": "maintenance_fixed_crew", "delta": "depreciation_rate", "N0": "initial_population"}
    p: RealEconomyParams

    # ---- setup ----------------------------------------------------------------
    def setup(self, world) -> None:
        p = self.p
        if sorted(p.priority) != sorted(SECTORS):
            raise ValueError(f"priority must be a permutation of {SECTORS}")
        L = world.ledger
        for item in ("raw", "final"):
            L.register_item(item, "good")
        for item in CAPITAL_HOLDER:
            L.register_item(item, "capital")
        rep = p.representation
        world.add_group(make_group("households", "households", rep.get("households", "aggregate"),
                                   p.initial_population))
        for s in SECTORS:
            world.add_group(make_group(s, HOLDER[s], rep.get(s, "aggregate")))
        st = world.layer_state(NAME)
        world.state["economy"] = st          # shared view for economy-generic layers
        st.update(conditions=[1.0] * p.initial_factories, demand_providers=[], supply_sources=[],
                  leftover_sinks=[], land_limit=None, land_exponent=0.5,
                  c=p.consumption_per_person, tech_fn=self.tech)
        L.begin_tick(0)
        if p.initial_factories:
            L.produce("factory", "factory", float(p.initial_factories), "initial endowment")
        st["prev_potential"] = None

    # ---- helpers ----------------------------------------------------------------
    def population(self, world) -> float:
        return world.groups["households"].size()

    def tech(self, world) -> Tech:
        p, st = self.p, world.layer_state(NAME)
        return Tech(
            c=p.consumption_per_person, e=p.conversion_rate, a=p.farm_output_per_worker * st.get("tech_level", 1.0),
            land_improvement=world.stock("farm", "land_improvement"),
            tools_processing=world.stock("factory", "tools_processing"),
            tools_distribution=world.stock("distribution", "tools_distribution"),
            land_elasticity=p.land_improvement_elasticity, tool_elasticity=p.tool_elasticity,
            v_p=p.factory_variable_labor / st.get("tech_level", 1.0), v_d=p.distribution_labor / st.get("tech_level", 1.0),
            F_p=p.factory_fixed_crew, F_m=p.maintenance_fixed_crew, capacity=p.factory_capacity,
            conditions=tuple(st["conditions"]),
            land_maintenance=p.land_improvement_maintenance, tool_maintenance=p.tool_maintenance,
            land_limit=st["land_limit"], land_exponent=st["land_exponent"])

    # ---- phases -------------------------------------------------------------------
    def plan(self, world) -> None:
        st = world.layer_state(NAME)
        tech = self.tech(world)
        restock = sum(fn(world) for fn in st["demand_providers"])
        research = st.get("research_labor", 0.0)        # Stage 9: researchers don't produce but do eat
        N = self.population(world)
        plan = rules.make_plan(N - min(research, N), tech, order=list(self.p.priority),
                               shortage_rule=self.p.shortage_rule, restock_demand=restock,
                               eaters=N if research > 0 else None)
        world.vars["plan"] = plan
        world.vars["tech"] = tech
        world.vars["farmers"] = plan.alloc["farming"]
        if st["prev_potential"] is None:   # baseline for the growth_rate metric
            st["prev_potential"] = rules.potential_output(plan.workers, tech)
        for s in SECTORS:
            world.groups[s].labor = plan.alloc[s]

    def produce(self, world) -> None:
        """Farming first, then processing. The harvest shock (if any) hits realised output only."""
        plan, tech, L = world.vars["plan"], world.vars["tech"], world.ledger
        farmers = plan.alloc["farming"]
        mult = world.vars.get("harvest_multiplier", 1.0)
        raw = rules.farm_output(farmers, tech.a_x, tech.land_limit, tech.land_exponent) * mult
        L.produce("farm", "raw", raw, "harvest")
        L.transfer("farm", "factory", "raw", raw, "raw to factory")

        labor = plan.alloc["processing"]
        k = plan.factories_operated
        if tech.F_p > 0:
            k = min(k, int(math.floor(labor / tech.F_p + 1e-9)))
        final = 0.0
        if k > 0:
            variable = max(labor - tech.F_p * k, 0.0)
            by_labor = variable / tech.v_p_eff if tech.v_p_eff > 0 else math.inf
            final = min(raw * tech.e, by_labor, rules.capacity_of_best(k, tech.conditions, tech.capacity),
                        plan.target)
        # final/e can exceed the raw stock by one rounding step when e != 1
        L.remove("factory", "raw", min(final / tech.e, world.stock("factory", "raw")), "use", "processed")
        L.produce("factory", "final", final, "processed")
        world.vars.update(raw_output=raw, final_output=final, factories_running=k,
                          effective_a=raw / farmers if farmers > 0 else tech.a_x * mult)

    def maintain(self, world) -> None:
        """Unmaintained share u of every capital stock loses depreciation_rate * u this tick.

        Factories below `scrap_condition` are scrapped (their remaining capital is
        written off as depreciation) so they stop needing a maintenance crew.
        """
        plan, L, st = world.vars["plan"], world.ledger, world.layer_state(NAME)
        u = rules.unmaintained_share(plan.alloc["maintenance"], plan.required["maintenance"])
        loss = self.p.depreciation_rate * u
        world.vars["unmaintained_share"] = u
        if loss <= 0:
            return
        kept, lost = [], 0.0
        for cond in st["conditions"]:
            after = cond * (1 - loss)
            if after < self.p.scrap_condition:
                lost += cond
            else:
                lost += cond - after
                kept.append(after)
        st["conditions"] = kept
        L.remove("factory", "factory", min(lost, world.stock("factory", "factory")), "depreciation",
                 "unmaintained / scrapped")
        for item in ("land_improvement", "tools_processing", "tools_distribution"):
            holder = CAPITAL_HOLDER[item]
            L.remove(holder, item, world.stock(holder, item) * loss, "depreciation", "unmaintained")

    def distribute(self, world) -> None:
        """Deliver up to N*c to households (limited by distribution labor), they consume it all."""
        plan, tech, L, st = world.vars["plan"], world.vars["tech"], world.ledger, world.layer_state(NAME)
        n = plan.people
        capacity = plan.alloc["distribution"] / tech.v_d_eff if tech.v_d_eff > 0 else math.inf
        wanted = min(plan.need, capacity)
        from_factory = min(world.stock("factory", "final"), wanted)
        L.transfer("factory", "households", "final", from_factory, "distribution")
        delivered = from_factory
        for source in st["supply_sources"]:
            if wanted - delivered <= rules.EPS:
                break
            delivered += source(world, wanted - delivered)
        L.remove("households", "final", world.stock("households", "final"), "consumption")
        leftover = world.stock("factory", "final")
        for sink in st["leftover_sinks"]:
            leftover -= sink(world, leftover)
        wasted_final = world.stock("factory", "final")
        wasted_raw = world.stock("factory", "raw")
        L.remove("factory", "final", wasted_final, "waste", "unused final goods")
        L.remove("factory", "raw", wasted_raw, "waste", "unprocessed raw")
        world.vars.update(delivered=delivered, consumption_per_person=delivered / n if n > 0 else 0.0,
                          wasted_final=wasted_final, wasted_raw=wasted_raw)

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.plan, Phase.PRODUCTION: self.produce,
                Phase.MAINTENANCE: self.maintain, Phase.DISTRIBUTION: self.distribute}

    # ---- invariants -------------------------------------------------------------
    def invariants(self):
        def labor_adds_up(world, ctx):
            plan = world.vars.get("plan")
            if plan is None:
                return []
            used = sum(plan.alloc.values()) + plan.surplus
            inv = world.vars.get("investment_labor", 0.0)
            out = []
            if not close(used, plan.workers, plan.workers):
                out.append(Violation("labor_adds_up", f"allocated {used:.9g} != workers {plan.workers:.9g}"))
            if inv > plan.surplus * (1 + 1e-9) + 1e-12:
                out.append(Violation("labor_adds_up", f"investment {inv} exceeds surplus {plan.surplus}"))
            if any(v < -1e-12 for v in plan.alloc.values()):
                out.append(Violation("labor_adds_up", f"negative allocation {plan.alloc}"))
            return out

        def factory_conditions_match_ledger(world, ctx):
            conds = world.layer_state(NAME)["conditions"]
            book = world.stock("factory", "factory")
            bad = [] if close(sum(conds), book) else [
                Violation("factory_conditions", f"sum of factory conditions {sum(conds)} != ledger {book}",
                          ("factory",), ("factory",))]
            if any(c < -1e-12 or c > 1 + 1e-12 for c in conds):
                bad.append(Violation("factory_conditions", f"condition out of [0,1]: {conds}", ("factory",)))
            return bad

        def no_overconsumption(world, ctx):
            plan = world.vars.get("plan")
            if plan is None:
                return []
            d = world.vars.get("delivered", 0.0)
            return [] if d <= plan.need * (1 + 1e-9) + 1e-12 else [
                Violation("no_overconsumption", f"delivered {d} > need {plan.need}", ("final",))]

        def physics(world, ctx):
            """The ledger's production and consumption must be what the technology allows.

            Recomputed independently from the plan and the technology, so a layer that
            posts more output than its labor, inputs or capacity allow is caught.
            """
            plan, tech = world.vars.get("plan"), world.vars.get("tech")
            if plan is None:
                return []
            flows = {}
            for p_ in ctx.journal:
                flows[(p_.item, p_.flow)] = flows.get((p_.item, p_.flow), 0.0) + p_.amount
            out = []
            mult = world.vars.get("harvest_multiplier", 1.0)
            harvest = flows.get(("raw", "production"), 0.0)
            expect = rules.farm_output(plan.alloc["farming"], tech.a_x, tech.land_limit, tech.land_exponent) * mult
            if not close(harvest, expect, expect):
                out.append(Violation("physics", f"harvest {harvest:.9g} != farmers' output {expect:.9g}", ("raw",)))
            baked = flows.get(("final", "production"), 0.0)
            used = flows.get(("raw", "use"), 0.0)
            k = min(plan.factories_operated, int(math.floor(plan.alloc["processing"] / tech.F_p + 1e-9))) if tech.F_p > 0 else plan.factories_operated
            by_labor = max(plan.alloc["processing"] - tech.F_p * k, 0.0) / tech.v_p_eff if tech.v_p_eff > 0 else math.inf
            cap = rules.capacity_of_best(k, tech.conditions, tech.capacity) if k > 0 else 0.0
            bound = min(used * tech.e * (1 + 1e-9) + 1e-9, by_labor * (1 + 1e-9) + 1e-9, cap * (1 + 1e-9) + 1e-9)
            if baked > bound:
                out.append(Violation("physics", f"baked {baked:.9g} > what wheat, labor and capacity allow {bound:.9g}", ("final",)))
            eaten = flows.get(("final", "consumption"), 0.0)
            carry = plan.alloc["distribution"] / tech.v_d_eff if tech.v_d_eff > 0 else math.inf
            if eaten > min(carry, plan.need) * (1 + 1e-9) + 1e-9:
                out.append(Violation("physics", f"consumed {eaten:.9g} > deliverable {min(carry, plan.need):.9g}", ("final",)))
            N = world.groups["households"].size()
            expected_N = world.vars.get("population_expected", plan.people)
            if not close(N, expected_N, expected_N):
                out.append(Violation("physics", f"population {N:.9g} changed outside the demography rule ({expected_N:.9g})"))
            return out

        return [Invariant("labor_adds_up", labor_adds_up), Invariant("physics", physics),
                Invariant("factory_conditions", factory_conditions_match_ledger),
                Invariant("no_overconsumption", no_overconsumption)]

    # ---- metrics --------------------------------------------------------------------
    def metrics(self):
        st_get = lambda w: w.layer_state(NAME)

        def labor_share(w):
            plan = w.vars["plan"]
            n = plan.workers or 1.0
            inv = w.vars.get("investment_labor", 0.0)
            out = {s: plan.alloc[s] / n for s in SECTORS}
            out["investment"] = inv / n
            out["leisure"] = (plan.surplus - inv) / n
            return out

        def capital(w):
            return {"factory": w.stock("factory", "factory"),
                    "factory_wip": w.stock("factory", "factory_wip"),
                    "land_improvement": w.stock("farm", "land_improvement"),
                    "tools_processing": w.stock("factory", "tools_processing"),
                    "tools_distribution": w.stock("distribution", "tools_distribution")}

        def potential(w):
            st = st_get(w)
            y = rules.potential_output(self.population(w), self.tech(w))
            st["potential_now"] = y
            return y

        def potential_growth(w):
            st = st_get(w)
            prev, now = st["prev_potential"], st["potential_now"]
            st["prev_potential"] = now
            return now / prev - 1.0 if prev > 0 else 0.0

        def pop_growth(w):
            n_now = self.population(w)
            return n_now / w.vars["plan"].workers - 1.0 if w.vars["plan"].workers > 0 else 0.0

        return [
            Metric("population", lambda w: self.population(w), "people at end of tick", "people"),
            Metric("population_growth", pop_growth, "growth of population this tick", "1/tick"),
            Metric("labor_share", labor_share, "share of population per activity", "share"),
            Metric("surplus_share", lambda w: w.vars["plan"].surplus / (w.vars["plan"].workers or 1),
                   "labor left after production, maintenance and restocking", "share"),
            Metric("maintenance_share", lambda w: w.vars["plan"].alloc["maintenance"] / (w.vars["plan"].workers or 1),
                   "share of labor in maintenance", "share"),
            Metric("labor_shortfall", lambda w: w.vars["plan"].shortfall, "required minus available labor", "workers"),
            Metric("consumption_per_person", lambda w: w.vars["consumption_per_person"],
                   "final goods consumed per person", "units/person"),
            Metric("leisure_per_person", lambda w: (w.vars["plan"].surplus - w.vars.get("investment_labor", 0.0))
                   / (w.vars["plan"].workers or 1), "leisure (worker-ticks) per person", "share"),
            Metric("total_output", lambda w: w.vars["final_output"], "final goods produced", "units"),
            Metric("raw_output", lambda w: w.vars["raw_output"], "raw material harvested", "units"),
            Metric("potential_output", potential, "max sustainable final output given labor and capital", "units"),
            Metric("growth_rate", potential_growth, "growth of potential output this tick", "1/tick"),
            Metric("capital", capital, "capital stock by type", "units"),
            Metric("factories", lambda w: len(st_get(w)["conditions"]), "number of factories", "count"),
            Metric("factory_utilization", lambda w: w.vars["plan"].need / max(self.tech(w).total_capacity, 1e-12),
                   "consumption need / factory capacity", "ratio"),
            Metric("unmaintained_share", lambda w: w.vars["unmaintained_share"], "share of capital unmaintained", "share"),
            Metric("effective_farm_productivity", lambda w: w.vars["effective_a"],
                   "raw output per farmer (after capital, land limits, shocks)", "units/worker"),
            Metric("waste", lambda w: {"raw": w.vars["wasted_raw"], "final": w.vars["wasted_final"]},
                   "goods discarded at end of tick", "units"),
        ]
