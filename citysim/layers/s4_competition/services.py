"""Stage 4: a services sector — labor-only work people want once their bread need is met.

Services are produced and consumed in the same tick (haircuts, teaching, entertainment):
`labor_per_unit` workers per unit plus a fixed crew per firm. Households spend on
services whatever they planned to spend but did not spend on bread, so satiated
households no longer hoard: their money keeps circulating.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.layer import Layer, LayerParams
from .agent_market import SERVICES


class ServicesParams(LayerParams):
    firms: int = Field(3, ge=0, description="services firms at the start (0: the sector has to emerge by entry)")
    labor_per_unit: float = Field(0.5, gt=0, description="workers per unit of service")
    fixed_crew: float = Field(2.0, ge=0, description="workers a services firm needs to open at all")
    share: float = Field(1.0, ge=0, le=1, description="share of unspent planned spending that goes to services")
    necessity_band: float = Field(0.0, ge=0, le=1, description="0: only fully fed households buy services; "
                                  "b > 0: services spending falls linearly to zero as bread shortfall reaches b")
    initial_demand: float = Field(0.1, ge=0, description="services per person each firm initially expects (in total)")


class Services(Layer):
    name = "services"
    requires = ["agents"]
    Params = ServicesParams
    p: ServicesParams

    def setup(self, world) -> None:
        m = world.state["economy"]["market_layer"]
        m.services = self.p
        world.state["economy"]["services_layer"] = self
        m.n_sec = 5
        m.sector_names = m.sector_names[:4] + ["services"]
        st = world.state["economy"]
        st["extra_categories"] = list(st.get("extra_categories", [])) + ["services"]
        for _ in range(self.p.firms):
            self.open_firm(world, m, staff_from_others=True)

    def open_firm(self, world, m, staff_from_others: bool = False, owner=None):
        p, L = self.p, world.ledger
        N = float(m.size.sum())
        wage = float(np.mean([f.wage for f in m.firms if f.sector != SERVICES])) if m.firms else 1.0
        expected = p.initial_demand * N / max(p.firms, 1)
        unit = wage * (p.labor_per_unit + p.fixed_crew / max(expected, 1.0))
        f = m.new_firm(world, SERVICES, unit * (1 + m.p.initial_markup), wage, expected, owner=owner)
        if staff_from_others:
            # a few people start out in services, taken at random from the other occupations
            staff = p.fixed_crew + p.labor_per_unit * expected
            idx = np.where(m.sector != SERVICES)[0]
            m.rng.shuffle(idx)
            n = int(np.ceil(staff / m.p.agent_size))
            take = idx[:n]
            m.sector[take] = SERVICES
            m.employer[take] = f.id
            L.post(f.account, "ext:creation", "cash", 2 * wage * staff, "creation", "initial money")
        return f
