"""Stage 9: technology and long-run growth — research, learning by doing, slow diffusion.

A technology level A multiplies farm output per worker and divides the labor needed per loaf
in processing and distribution (fixed crews, maintenance and services do not improve — so
services get relatively dearer as everything else gets cheaper: Baumol's cost disease).

* Research: `research_share` of the population works at a research institute instead of
  producing. Researchers push out the frontier A*:
      A*' = A* (1 + research_productivity x researchers / population / A*^difficulty)
  (difficulty > 0: ideas get harder to find).
* Diffusion: a user's adopted technology moves towards the frontier by `diffusion` x the gap
  each tick; under the market each firm has its own adoption speed (lognormal around
  `diffusion`), and entrants start at their sector's average less `entrant_lag`.
* Learning by doing: practice raises productivity on top of what is adopted:
      x (1 + cumulative output / lbd_scale)^lbd_exponent.

Under the Stage 1 planner the planner takes the researchers out of the workforce before
planning production (everyone still eats) and the economy has one technology level. Under
the market economy the research institute is a public employer in a sixth occupation
("research"): it hires in the labor market like any firm, its wages are paid by the
government (or by a levy on households without one), and each firm has its own level.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.invariants import Invariant, Violation
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

GOV = "gov"


class TechnologyParams(LayerParams):
    research_share: float = Field(0.0, ge=0, lt=1, description="share of the population doing research")
    research_productivity: float = Field(0.05, ge=0, description="frontier growth per tick per unit research share")
    difficulty: float = Field(0.0, ge=0, description="ideas get harder: growth / A*^difficulty")
    diffusion: float = Field(0.02, ge=0, le=1, description="share of the gap to the frontier adopted per tick")
    diffusion_spread: float = Field(0.5, ge=0, description="market: sd of log adoption speed across firms")
    entrant_lag: float = Field(0.1, ge=0, le=1, description="market: entrants start this far below their sector average")
    lbd_exponent: float = Field(0.0, ge=0, description="learning by doing: elasticity of productivity to experience")
    lbd_scale: float = Field(20000.0, gt=0, description="cumulative output that doubles the learning term's base")


class Technology(Layer):
    name = "technology"
    requires = ["economy"]
    provides = ["technology"]
    Params = TechnologyParams
    p: TechnologyParams

    def setup(self, world) -> None:
        st = world.state["economy"]
        self.st = st
        self.market = st.get("market_layer")
        self.frontier = 1.0
        self.t = dict(researchers=0.0, grant=0.0)
        self.rng = np.random.default_rng([world.seed, 9001])
        self.research = None
        if self.market is None:                       # Stage 1 planner: one economy-wide level
            self.adopted = 1.0
            self.experience = 0.0
            st["tech_level"] = 1.0
            st["research_labor"] = self.p.research_share * world.groups["households"].size()
            return
        m = self.market
        self.speed = {}
        for f in m.firms:
            self.init_firm(f, 1.0)
        self.research = None
        if self.p.research_share > 0:
            from ..s4_competition.agent_market import AgentMarket
            if not isinstance(m, AgentMarket):
                raise ValueError("research under the market economy needs the agent market (Stage 4+)")
            if m.n_sec < 5:                     # keep slot 4 for services (empty when there are none)
                m.sector_names = list(m.sector_names)[:4] + ["services"]
                m.n_sec = 5
                m.public_sectors = tuple(getattr(m, "public_sectors", ())) + (4,)   # nobody may open one
            m.RESEARCH = m.n_sec
            m.n_sec += 1
            m.sector_names = list(m.sector_names) + ["research"]
            m.public_sectors = tuple(getattr(m, "public_sectors", ())) + (m.RESEARCH,)
            wage = float(np.mean([f.wage for f in m.firms]))
            target = self.target(m)
            f = m.new_firm(world, m.RESEARCH, wage, wage, target)
            f.owners = np.array([], int)                       # a public institute: no dividends
            f.owner = -1
            self.init_firm(f, 1.0)
            self.research = f
            st["extra_categories"] = list(st.get("extra_categories", [])) + ["research_grant"]
            # a few people start out as researchers, taken at random from the other occupations
            idx = np.where(m.sector != m.RESEARCH)[0]
            m.rng.shuffle(idx)
            n = int(np.ceil(target / m.p.agent_size))
            m.sector[idx[:n]] = m.RESEARCH
            m.employer[idx[:n]] = -1                            # they leave their old jobs

    def init_firm(self, f, level: float) -> None:
        f.adopted = level
        f.experience = 0.0
        f.tech = level
        self.speed[f.id] = float(np.clip(self.p.diffusion * np.exp(self.p.diffusion_spread * self.rng.standard_normal()
                                                                    - self.p.diffusion_spread ** 2 / 2), 0.0, 1.0))

    def target(self, m) -> float:
        return self.p.research_share * float(m.size.sum())

    def lbd(self, experience: float) -> float:
        p = self.p
        return (1.0 + experience / p.lbd_scale) ** p.lbd_exponent if p.lbd_exponent > 0 else 1.0

    # ------------------------------------------------------------------ phase 1: fund research (market)
    def fund(self, world) -> None:
        f = self.research
        if f is None:
            return
        f.orders_in = self.target(self.market)               # its staffing target, seen by the market's adjust
        need = f.wage_bill - max(world.ledger.balance(f.account, self.market.money_item), 0.0)
        if need <= 1e-12:
            return
        gov = self.st.get("gov_layer")
        L, mi = world.ledger, self.market.money_item
        if gov is not None:
            got = gov.fund(world, need)
            if got > 1e-12:
                L.post(f.account, GOV, mi, got, "transfer", "research_grant")
                gov.t["research"] = gov.t.get("research", 0.0) + got
        else:
            m = self.market
            have = float(m.cash.sum())
            got = min(need, max(have, 0.0))
            if got > 1e-12:
                L.post(f.account, "households", mi, got, "transfer", "research_grant")
                m.cash -= got * m.cash / have
        self.t["grant"] += max(got, 0.0)

    # ------------------------------------------------------------------ phase 8: ideas, diffusion, practice
    def advance(self, world) -> None:
        p, st = self.p, self.st
        if self.market is None:
            N = world.groups["households"].size()
            R = st.get("research_labor", 0.0)
            self.t["researchers"] = R
            if N > 0:
                self.frontier *= 1 + p.research_productivity * (R / N) / self.frontier ** p.difficulty
            self.adopted += p.diffusion * (self.frontier - self.adopted)
            self.experience += world.vars.get("final_output", 0.0)
            st["tech_level"] = self.adopted * self.lbd(self.experience)
            st["research_labor"] = p.research_share * N          # next tick's research staff
            self.record(world)
            return
        m = self.market
        N = float(m.size.sum())
        R = self.research.hired if self.research is not None else 0.0
        self.t["researchers"] = R
        if N > 0:
            self.frontier *= 1 + p.research_productivity * (R / N) / self.frontier ** p.difficulty
        for s in range(m.n_sec):
            firms = m.of(s)
            known = [f for f in firms if hasattr(f, "adopted")]
            avg = float(np.mean([f.adopted for f in known])) if known else 1.0
            for f in firms:
                if not hasattr(f, "adopted"):                 # an entrant
                    self.init_firm(f, avg * (1 - p.entrant_lag))
                f.adopted += self.speed.get(f.id, p.diffusion) * (self.frontier - f.adopted)
                f.experience += getattr(f, "output", 0.0)
                f.tech = f.adopted * self.lbd(f.experience)
        if self.research is not None:
            self.research.price = self.research.wage
        self.record(world)

    def record(self, world) -> None:
        world.vars["tech_stats"] = self.t
        self.t = dict(researchers=0.0, grant=0.0)

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.fund, Phase.DEMOGRAPHY: self.advance}

    # ------------------------------------------------------------------ invariants
    def invariants(self):
        def frontier(world, ctx):
            out = []
            if self.frontier < 1.0 - 1e-12:
                out.append(Violation("technology", f"frontier fell below its start: {self.frontier}"))
            if self.market is not None:
                for f in self.market.firms:
                    if getattr(f, "adopted", 1.0) > self.frontier * (1 + 1e-9):
                        out.append(Violation("technology", f"{f.account} uses more than the frontier"))
            return out
        return [Invariant("technology", frontier)]

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        def level(w):
            if self.market is None:
                return w.state["economy"].get("tech_level", 1.0)
            firms = [f for f in self.market.firms if f is not self.research]
            return float(np.mean([getattr(f, "tech", 1.0) for f in firms])) if firms else 1.0

        def spread(w):
            if self.market is None:
                return 0.0
            t = [getattr(f, "tech", 1.0) for f in self.market.firms if f is not self.research]
            return float(max(t) / min(t)) if t else 1.0

        out = [Metric("frontier", lambda w: self.frontier, "best technology known", "x"),
               Metric("tech_level", level, "technology in use (average over firms)", "x"),
               Metric("tech_spread", spread, "best / worst firm technology", "ratio"),
               Metric("researchers", lambda w: w.vars.get("tech_stats", {}).get("researchers", 0.0), "people doing research", "people")]
        if self.market is not None and len(getattr(self.market, "sector_names", [])) > 4:
            def svc_price(w):
                fs = self.market.of(4)
                pl = w.vars.get("price_level", 0.0)
                return float(np.mean([f.price for f in fs])) / pl if fs and pl > 0 else 0.0
            out.append(Metric("services_price_ratio", svc_price, "price of a unit of services / price of a loaf", "ratio"))
        if self.market is not None:
            out.append(Metric("research_grant", lambda w: w.vars.get("tech_stats", {}).get("grant", 0.0), "research funding", "money"))
        return out
