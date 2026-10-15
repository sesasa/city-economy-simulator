"""Stage 10: services in the planned economy.

The Stage 1 planner feeds everyone and invests a share of the surplus labor; the rest was
leisure. For a fair comparison with the market economies (where households buy services),
the planner can put a share of that free labor into services (haircuts, teaching, care):
`labor_per_unit` workers per unit, delivered to households equally and consumed at once.
"""
from __future__ import annotations

from pydantic import Field

from ...core.invariants import Invariant, Violation
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase


class PlannedServicesParams(LayerParams):
    share_of_free_labor: float = Field(0.8, ge=0, le=1, description="share of labor left after investment put into services")
    labor_per_unit: float = Field(0.5, gt=0, description="workers per unit of services")


class PlannedServices(Layer):
    name = "planned_services"
    requires = ["planner"]
    Params = PlannedServicesParams
    p: PlannedServicesParams

    def setup(self, world) -> None:
        world.ledger.register_item("services", "good")
        self.last = dict(labor=0.0, units=0.0, free=0.0)

    def serve(self, world) -> None:
        plan = world.vars.get("plan")
        if plan is None:
            return
        free = max(plan.surplus - world.vars.get("investment_labor", 0.0) - world.vars.get("storage_labor", 0.0), 0.0)
        labor = self.p.share_of_free_labor * free
        units = labor / self.p.labor_per_unit
        L = world.ledger
        if units > 0:
            L.produce("households", "services", units, "planned services")
            L.remove("households", "services", units, "consumption")
        n = plan.people
        self.last = dict(labor=labor, units=units, free=free)
        world.vars["services_per_person"] = units / n if n > 0 else 0.0
        world.vars["services_labor"] = labor

    def hooks(self):
        return {Phase.FINANCE: self.serve}

    def invariants(self):
        def within_free_labor(world, ctx):
            if self.last["labor"] > self.last["free"] * (1 + 1e-9) + 1e-9:
                return [Violation("physics", "services used more labor than was free")]
            made = sum(p.amount for p in ctx.journal if p.item == "services" and p.flow == "production")
            if made > self.last["labor"] / self.p.labor_per_unit * (1 + 1e-9) + 1e-9:
                return [Violation("physics", "services beyond the labor that made them", ("services",))]
            return []
        return [Invariant("planned_services", within_free_labor)]

    def metrics(self):
        def share(w):
            plan = w.vars.get("plan")
            return self.last["labor"] / plan.people if plan is not None and plan.people > 0 else 0.0
        return [Metric("services_per_person", lambda w: w.vars.get("services_per_person", 0.0), "services per person", "units"),
                Metric("services_labor_share", share, "share of people working in services", "share")]
