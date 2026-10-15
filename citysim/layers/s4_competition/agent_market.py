"""Stage 4 base layer: the Stage 2 market with individual households and many firms.

What changes relative to `market`:
* Households are agents (numpy arrays, one entry per `agent_size` people): each has
  an occupation, an employer (or none), savings, income this tick, and a retraining
  countdown after switching occupation (no work meanwhile: structural unemployment).
* Labor market with job search: firms fire down to their labor demand; unemployed
  people of the right occupation fill vacancies, best-paying firms first, with a
  matching efficiency below one.
* Firms belong to owners (household agents) who receive their dividends, so profits
  create inequality. Ownership "public" pays dividends to everyone per head.
* Market power: a firm's normal markup rises with its sector's concentration
  (Herfindahl index of sales): mu = mu_competitive + (mu_monopoly - mu_competitive) * HHI.
* Several firms per sector (`firms` per sector), with an optional services sector
  (added by the `services` layer) and entry/exit (the `entry_exit` layer).

Every Stage 2 rule (prices, wages, investment, payments, credit hooks) is inherited.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from ...core.agents import AggregateGroup
from ...core.invariants import Invariant, Violation, close
from ...core.metrics import Metric
from ...core.phases import Phase
from ..s1_real_economy import rules
from ..s2_money_prices import behavior as B
from ..s2_money_prices.market import (DIST, FACTORY, FARM, MAINT, OUTPUT_ITEM, SECTORS, Firm, MarketEconomy,
                                      MarketParams)

SERVICES = 4
SECTOR_NAMES = list(SECTORS) + ["services"]


def gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Gini coefficient of non-negative values (weighted by people per agent)."""
    v = np.maximum(np.asarray(values, float), 0.0)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if v.size == 0 or w.sum() <= 0 or (v * w).sum() <= 0:
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cvw = np.cumsum(v * w)
    total_w, total = cw[-1], cvw[-1]
    # area under the Lorenz curve with trapezoids
    lorenz_prev = np.concatenate([[0.0], cvw[:-1]]) / total
    area = np.sum(w / total_w * (lorenz_prev + cvw / total) / 2.0)
    return float(1.0 - 2.0 * area)


class AgentMarketParams(MarketParams):
    agent_size: float = Field(1.0, gt=0, description="people per household agent")
    farm_firms: int = Field(2, ge=1)
    factory_firms: int = Field(1, ge=1)
    distribution_firms: int = Field(2, ge=1)
    maintenance_firms: int = Field(1, ge=1)
    ownership: str = Field("founders", pattern="^(founders|public)$",
                           description="founders: each firm has owner households; public: dividends per head")
    owners_per_firm: int = Field(20, ge=1, description="households sharing a firm's dividends under 'founders'")
    market_power: bool = Field(True, description="normal markup rises with sector concentration")
    competitive_markup: float = Field(0.1, description="normal markup when the sector is fully competitive")
    monopoly_markup: float = Field(0.6, description="normal markup for a monopoly")
    matching_efficiency: float = Field(0.8, ge=0, le=1, description="share of possible job matches made per tick")
    mobility_unemployed: float = Field(0.2, ge=0, le=1, description="chance an unemployed person reconsiders occupation")
    mobility_employed: float = Field(0.01, ge=0, le=1, description="chance an employed person reconsiders occupation")
    retrain_ticks: int = Field(3, ge=0, description="ticks without work after switching occupation")
    switch_horizon: float = Field(50.0, gt=0, description="ticks over which people weigh a switch against retraining")


class AgentMarket(MarketEconomy):
    name = "agent_market"
    provides = ["economy", "market", "agents"]
    Params = AgentMarketParams
    p: AgentMarketParams

    # ================================================================== setup
    def setup(self, world) -> None:
        p, L = self.p, world.ledger
        for item in ("raw", "final", "services"):
            L.register_item(item, "good")
        for item in ("factory", "factory_wip", "land_improvement", "tools_processing", "tools_distribution"):
            L.register_item(item, "capital")
        L.register_item("cash", "money")
        st = world.layer_state(self.name)
        world.state["economy"] = st
        st.update(land_limit=None, land_exponent=0.5, c=p.consumption_per_person)
        L.begin_tick(0)
        self.rng = world.rng
        self.n_sec = 4
        self.sector_names = list(SECTORS)

        # starting point: the planner's allocation and cost-based prices (as in Stage 2)
        n_agents = max(1, int(round(p.initial_population / p.agent_size)))
        N = n_agents * p.agent_size
        world.add_group(AggregateGroup(name="households", account="households", count=N))
        tech0 = self._tech(st, conditions=(1.0,) * p.initial_factories, L=0, Tp=0, Td=0)
        plan = rules.make_plan(N, tech0, order=list(rules.SECTORS))
        need = N * p.consumption_per_person
        w = p.initial_wage
        mk = 1 + p.initial_markup
        raw0 = need / p.conversion_rate
        p_r = w * plan.alloc["farming"] / max(raw0, 1e-9) * mk
        p_m = w * mk
        unit_f = (p_r * raw0 + w * plan.alloc["processing"] + p_m * plan.required["maintenance"]) / max(need, 1e-9)
        p_w = unit_f * mk
        p_f = (p_w + w * p.distribution_labor) * mk
        prices = {FARM: p_r, FACTORY: p_w, DIST: p_f, MAINT: p_m}
        expected = {FARM: raw0, FACTORY: need, DIST: need, MAINT: plan.required["maintenance"]}
        counts = {FARM: p.farm_firms, FACTORY: p.factory_firms, DIST: p.distribution_firms, MAINT: p.maintenance_firms}
        cost = {FARM: w * plan.alloc["farming"],
                FACTORY: w * plan.alloc["processing"] + p_r * raw0 + p_m * plan.required["maintenance"],
                DIST: w * plan.alloc["distribution"] + p_w * need,
                MAINT: w * plan.alloc["maintenance"]}
        firm_need = {s: max(p.cash_buffer, 1.0) * cost[s] for s in range(4)}
        hh_need = p_f * need
        total = hh_need + sum(firm_need.values())
        scale = (p.initial_money / total) if p.initial_money else 1.0
        self.money_scale = scale
        ps = scale if p.prices_match_money else 1.0

        # households (agents)
        alloc = np.array([plan.alloc["farming"], plan.alloc["processing"], plan.alloc["distribution"],
                          plan.alloc["maintenance"]], float)
        shares = alloc / alloc.sum()
        per_sector = np.floor(shares * n_agents).astype(int)
        per_sector[np.argsort(-(shares * n_agents - per_sector))[: n_agents - per_sector.sum()]] += 1
        self.sector = np.repeat(np.arange(4), per_sector)
        self.rng.shuffle(self.sector)
        self.size = np.full(n_agents, p.agent_size, float)
        self.employer = np.full(n_agents, -1, int)
        self.retrain = np.zeros(n_agents, int)
        L.open_account("households")
        hh_money = hh_need * scale
        L.post("households", "ext:creation", "cash", hh_money, "creation", "initial money")
        self.cash = np.full(n_agents, hh_money / n_agents)
        self.income = np.zeros(n_agents)
        self.hunger = np.zeros(n_agents)
        self.pop_carry = 0.0

        # firms
        self.firms: list[Firm] = []
        self.next_firm_id = 0
        fac_left = p.initial_factories
        for s in range(4):
            for j in range(counts[s]):
                f = self.new_firm(world, s, prices[s] * ps, w * ps, expected[s] / counts[s])
                L.post(f.account, "ext:creation", "cash", firm_need[s] * scale / counts[s], "creation", "initial money")
                if s == FACTORY:
                    n_fac = fac_left // (counts[s] - j) if counts[s] - j > 0 else 0
                    n_fac = max(n_fac, 1 if fac_left > 0 else 0)
                    fac_left -= n_fac
                    f.conditions = [1.0] * n_fac
                    if n_fac:
                        L.produce(f.account, "factory", float(n_fac), "initial endowment")
        # start everyone in their occupation employed, split across that sector's firms
        for s in range(4):
            idx = np.where(self.sector == s)[0]
            firms = self.of(s)
            for k, i in enumerate(idx):
                self.employer[i] = firms[k % len(firms)].id
        self.pools_at_hire = self.pools()
        st["firms"] = self.firms
        st["market_layer"] = self
        st.setdefault("credit_providers", [])
        st.setdefault("investment_finance", [])
        st.setdefault("extra_categories", ["equity", "liquidation"])
        st["tech_fn"] = lambda world: self._tech(st)
        self.prev_price_level = p_f * ps
        self.prev_potential = None
        self.last = {}
        self.entries = 0
        self.exits = 0

    def new_firm(self, world, sector: int, price: float, wage: float, expected: float, owner: int | None = None) -> Firm:
        p = self.p
        acc = f"firm:{self.sector_names[sector]}:{self.next_firm_id}"
        world.ledger.open_account(acc)
        f = Firm(self.next_firm_id, sector, acc, price, wage, expected)
        f.markup = p.initial_markup
        f.unit_cost = price / (1 + p.initial_markup)
        if p.ownership == "public":
            f.owners = np.array([], int)
        else:
            k = min(p.owners_per_firm, len(self.size))
            others = self.rng.choice(len(self.size), size=k, replace=False)
            if owner is not None:
                others = np.concatenate([[owner], others[others != owner][: k - 1]])
            f.owners = np.asarray(others, int)
        f.owner = int(f.owners[0]) if len(f.owners) else -1
        f.loss_streak = 0
        self.next_firm_id += 1
        self.firms.append(f)
        return f

    # ================================================================== helpers
    def pools(self) -> np.ndarray:
        return np.bincount(self.sector, weights=self.size, minlength=self.n_sec).astype(float)

    def hhi(self, sector: int) -> float:
        firms = self.of(sector)
        sales = np.array([f.revenue for f in firms])
        if sales.sum() <= 0:
            sales = np.ones(len(firms))
        sh = sales / sales.sum()
        return float((sh ** 2).sum())

    def normal_unit_cost(self, world, f: Firm) -> float:
        if f.sector == SERVICES:
            sv = self.services
            return f.wage * (sv.labor_per_unit + sv.fixed_crew / max(f.expected, 1.0))
        return super().normal_unit_cost(world, f)

    def pay_households(self, world, f: Firm, amount: float, category: str, weights: np.ndarray) -> None:
        """Wages go to the firm's employees, builders' wages to idle people of its occupation,
        dividends to its owner (or everyone per head under public ownership)."""
        if amount <= 0:
            return
        world.ledger.post("households", f.account, self.money_item, amount, "transfer", category)
        if len(self.size) == 0:
            return              # nobody left: the money stays on the households account (an estate)
        if category == "wages":
            w = np.where(self.employer == f.id, self.size, 0.0)
        elif category == "investment_wages":
            w = np.where((self.sector == f.sector) & (self.employer < 0) & (self.retrain == 0), self.size, 0.0)
            if w.sum() <= 0:
                w = np.where(self.sector == f.sector, self.size, 0.0)
        elif category in ("dividends", "liquidation") and world.state["economy"].get("dividend_weights"):
            w = world.state["economy"]["dividend_weights"]()          # shareholders (Stage 7 stock market)
        elif category in ("dividends", "liquidation") and len(getattr(f, "owners", ())):
            w = np.zeros_like(self.size)
            np.add.at(w, f.owners, 1.0)
        else:
            w = self.size.copy()
        w = w / w.sum() if w.sum() > 0 else self.size / self.size.sum()
        self.cash += amount * w
        self.income += amount * w

    # ================================================================== phase 1
    def plan_and_hire(self, world) -> None:
        L = world.ledger
        self._ledger = L
        st = world.state["economy"]
        for f in self.firms:
            f.labor_demand = f.hired = f.wage_bill = f.costs = f.revenue = 0.0
            f.orders_in = f.sold = f.order_out = f.maint_req = f.maint_got = f.builders = 0.0
            f.op_costs = f.output = 0.0
        self.income = np.zeros_like(self.size)
        world.state["economy"]["spending_multiplier"] = 1.0
        self.sync_population(world)
        q = self.trade_share()
        world.vars["trade_completion"] = q
        self.plan_firms(world)
        for f in self.of(SERVICES):
            sv = self.services
            f.plan = max(f.expected, 0.0)
            f.labor_demand = (sv.fixed_crew + sv.labor_per_unit * f.plan) if f.plan > 1e-9 else 0.0
        self.hire_agents(world, q)
        world.vars["farmers"] = sum(f.hired for f in self.of(FARM))
        world.vars["tech"] = self._tech(st)

    def hire_agents(self, world, q: float) -> None:
        p = self.p
        rng = self.rng
        self.pools_at_hire = self.pools()
        for s in range(self.n_sec):
            firms = sorted(self.of(s), key=lambda g: -g.wage)
            for f in firms:
                target = f.labor_demand * q
                emp = np.where(self.employer == f.id)[0]
                have = self.size[emp].sum()
                if have > target + 1e-9:
                    rng.shuffle(emp)
                    keep = np.cumsum(self.size[emp]) <= target + 1e-9
                    self.employer[emp[~keep]] = -1
            unemp = np.where((self.sector == s) & (self.employer < 0) & (self.retrain == 0))[0]
            rng.shuffle(unemp)
            vac = {f.id: max(0.0, f.labor_demand * q - self.size[self.employer == f.id].sum()) for f in firms}
            total_v = sum(vac.values())
            can_fill = p.matching_efficiency * min(total_v, self.size[unemp].sum())
            pos = 0
            for f in firms:
                want = min(vac[f.id], max(can_fill, 0.0))
                if want <= 1e-9 or pos >= len(unemp):
                    continue
                sizes = self.size[unemp[pos:]]
                n = int(np.searchsorted(np.cumsum(sizes), want + 1e-9, side="right"))
                if n == 0 and want >= 0.5 * sizes[0]:
                    n = 1
                take = unemp[pos:pos + n]
                self.employer[take] = f.id
                pos += n
                can_fill -= self.size[take].sum()
            for f in firms:
                f.hired = float(self.size[self.employer == f.id].sum())
                f.wage_bill = f.hired * f.wage

    def sync_population(self, world) -> None:
        """Births add agents (into occupations in proportion to their size), deaths remove the hungriest."""
        p = self.p
        target = world.groups["households"].size()
        self.pop_carry += target - float(self.size.sum())
        k = int(self.pop_carry / p.agent_size)
        rng = self.rng
        if k > 0:
            pools = self.pools()
            probs = pools / pools.sum() if pools.sum() > 0 else np.full(self.n_sec, 1 / self.n_sec)
            new_sec = rng.choice(self.n_sec, size=k, p=probs)
            self.sector = np.concatenate([self.sector, new_sec])
            self.size = np.concatenate([self.size, np.full(k, p.agent_size)])
            self.employer = np.concatenate([self.employer, np.full(k, -1, int)])
            self.retrain = np.concatenate([self.retrain, np.zeros(k, int)])
            self.cash = np.concatenate([self.cash, np.zeros(k)])
            self.income = np.concatenate([self.income, np.zeros(k)])
            self.hunger = np.concatenate([self.hunger, np.zeros(k)])
            for obj, name, fill in world.state["economy"].get("agent_arrays", ()):    # e.g. shareholdings (Stage 7)
                a = getattr(obj, name)
                setattr(obj, name, np.concatenate([a, np.full(k, 0 if fill == "inherit" else fill, a.dtype)]))
            self.pop_carry -= k * p.agent_size
        elif k < 0:
            n = min(-k, len(self.size))
            w = self.hunger + 1e-6
            die = rng.choice(len(self.size), size=n, replace=False, p=w / w.sum())
            alive = np.ones(len(self.size), bool)
            alive[die] = False
            inherit = self.cash[die].sum()
            remap = np.cumsum(alive) - 1
            n_alive = int(alive.sum())
            for f in self.firms:
                if len(getattr(f, "owners", ())) and n_alive:
                    o = f.owners
                    heirs = rng.integers(n_alive, size=len(o))          # a dead owner's share passes to an heir
                    f.owners = np.where(alive[o], remap[o], heirs).astype(int)
                    f.owner = int(f.owners[0])
            for name in ("sector", "size", "employer", "retrain", "cash", "income", "hunger"):
                setattr(self, name, getattr(self, name)[alive])
            for obj, name, fill in world.state["economy"].get("agent_arrays", ()):
                a = getattr(obj, name)
                kept = a[alive]
                if fill == "inherit" and n_alive:        # holdings and debts pass to heirs per head
                    kept = kept + a[~alive].sum() * self.size / self.size.sum()
                setattr(obj, name, kept)
            if len(self.size):
                self.cash += inherit * self.size / self.size.sum()      # heirs keep the money
            self.pop_carry += n * p.agent_size
        world.groups["households"].set_size(float(self.size.sum()))

    # ================================================================== migration (Stage 8)
    def remove_agents(self, world, idx: np.ndarray) -> float:
        """Take agents out of the town (they emigrate); returns the money they take with them.
        The caller moves that money out of the ledger. Their firm ownership passes to heirs."""
        idx = np.unique(np.asarray(idx, int))
        if len(idx) == 0:
            return 0.0
        alive = np.ones(len(self.size), bool)
        alive[idx] = False
        n_alive = int(alive.sum())
        if n_alive == 0:
            return 0.0
        money = float(self.cash[idx].sum())
        remap = np.cumsum(alive) - 1
        for f in self.firms:
            if len(getattr(f, "owners", ())):
                o = f.owners
                heirs = self.rng.integers(n_alive, size=len(o))
                f.owners = np.where(alive[o], remap[o], heirs).astype(int)
                f.owner = int(f.owners[0])
        for name in ("sector", "size", "employer", "retrain", "cash", "income", "hunger"):
            setattr(self, name, getattr(self, name)[alive])
        for obj, name, fill in world.state["economy"].get("agent_arrays", ()):
            a = getattr(obj, name)
            kept = a[alive]
            if fill == "inherit":
                kept = kept + a[~alive].sum() * self.size / self.size.sum()
            setattr(obj, name, kept)
        world.groups["households"].set_size(float(self.size.sum()))
        return money

    def add_agents(self, world, sectors: np.ndarray, cash: np.ndarray) -> None:
        """Immigrants arrive unemployed in the given occupations with the given money
        (the caller has already put that money on the households account)."""
        k = len(sectors)
        if k == 0:
            return
        p = self.p
        self.sector = np.concatenate([self.sector, np.asarray(sectors, int)])
        self.size = np.concatenate([self.size, np.full(k, p.agent_size)])
        self.employer = np.concatenate([self.employer, np.full(k, -1, int)])
        self.retrain = np.concatenate([self.retrain, np.zeros(k, int)])
        self.cash = np.concatenate([self.cash, np.asarray(cash, float)])
        self.income = np.concatenate([self.income, np.zeros(k)])
        self.hunger = np.concatenate([self.hunger, np.zeros(k)])
        for obj, name, fill in world.state["economy"].get("agent_arrays", ()):
            a = getattr(obj, name)
            setattr(obj, name, np.concatenate([a, np.full(k, 0 if fill == "inherit" else fill, a.dtype)]))
        world.groups["households"].set_size(float(self.size.sum()))

    # ================================================================== phase 5
    def consume(self, world) -> None:
        cash_before = self.cash.copy()
        super().consume(world)
        if self.of(SERVICES):
            self.buy_services(world, cash_before)

    def buy_services(self, world, cash_before: np.ndarray) -> None:
        """What households planned to spend but did not spend on bread goes to services."""
        p, L, sv = self.p, world.ledger, self.services
        q = world.vars["trade_completion"]
        k = world.state["economy"].get("spending_multiplier", 1.0)
        planned = min(p.alpha_income * k, 1.0) * self.income + min(p.alpha_wealth * k, 1.0) * np.maximum(cash_before - self.income, 0.0)
        spent_on_bread = cash_before - self.cash
        # necessities first: people who did not get their bread keep their money for bread
        if sv.necessity_band > 0:     # smooth: a little short of bread, a little less on services (Stage 10)
            keep = np.clip(1.0 - self.hunger / sv.necessity_band, 0.0, 1.0)
        else:                          # strict: only fully fed households buy services (Stages 4-9)
            keep = (self.hunger < 0.01).astype(float)
        budget = keep * np.clip((planned - spent_on_bread) * sv.share, 0.0, self.cash)
        firms = self.of(SERVICES)
        prices = np.array([f.price for f in firms])
        price = float(prices.mean())
        desired = budget / price
        total = float(desired.sum())
        for f in firms:
            f.orders_in += total / len(firms)
        capacity = np.array([max(f.hired - sv.fixed_crew, 0.0) / sv.labor_per_unit for f in firms])
        sales = B.allocate_by_price(total * q, prices, capacity, choosiness=4.0)
        cost = float((sales * prices).sum())
        if cost > budget.sum() > 0:
            sales *= budget.sum() / cost
        paid = 0.0
        for f, x in zip(firms, sales):
            if x <= 1e-12:
                f.output = 0.0
                continue
            L.produce(f.account, "services", x, "served")
            L.transfer(f.account, "households", "services", x, "services")
            L.post(f.account, "households", self.money_item, x * f.price, "transfer", "services")
            f.revenue += x * f.price
            f.sold += x
            f.output = x
            paid += x * f.price
        if total > 0:
            self.cash -= paid * desired / total
        L.remove("households", "services", L.balance("households", "services"), "consumption")
        N = float(self.size.sum())
        world.vars["services_per_person"] = float(sales.sum()) / N if N > 0 else 0.0
        world.vars["services_spending"] = paid

    # ================================================================== phase 8
    def adjust(self, world) -> None:
        p = self.p
        lam = p.expectation_speed
        lo, hi = p.markup_bounds
        normal = {s: (p.competitive_markup + (p.monopoly_markup - p.competitive_markup) * self.hhi(s))
                  if p.market_power else p.normal_markup for s in range(self.n_sec)}
        for f in self.firms:
            f.last_profit = f.revenue - f.op_costs - getattr(f, "interest_paid", 0.0)
            f.loss_streak = getattr(f, "loss_streak", 0) + 1 if f.last_profit < 0 else 0
            f.op_costs_prev = f.op_costs
            f.expected = float(B.adaptive(f.expected, f.orders_in, lam))
            f.unit_cost = float(B.adaptive(f.unit_cost, self.normal_unit_cost(world, f), lam)) if f.unit_cost > 0 \
                else self.normal_unit_cost(world, f)
            if f.sector == MAINT:
                gap = B.order_gap(f.orders_in, f.hired)
            elif f.sector == SERVICES:
                sv = self.services
                gap = B.order_gap(f.orders_in, max(f.hired - sv.fixed_crew, 0.0) / sv.labor_per_unit)
            elif f.sector not in OUTPUT_ITEM:            # e.g. research (Stage 9): staffed to its orders
                gap = B.order_gap(f.orders_in, f.hired)
            else:
                gap = B.inventory_gap(self.stock(world, f, OUTPUT_ITEM[f.sector]), p.inventory_target * f.expected)
            f.markup = B.markup_update(f.markup, gap, p.price_speed, normal[f.sector], p.markup_pull, lo, hi)
            if f.unit_cost > 0:
                f.price = (1.0 + f.markup) * f.unit_cost
        pools = self.pools()
        income = np.zeros(self.n_sec)
        demand = np.zeros(self.n_sec)
        for s in range(self.n_sec):
            firms = self.of(s)
            if not firms:
                continue
            pool = float(pools[s])
            D = sum(f.labor_demand for f in firms)
            for f in firms:
                share = f.labor_demand / D * pool if D > 0 else pool / len(firms)
                f.wage = float(B.wage_update(f.wage, f.labor_demand, share, p.wage_speed, p.max_step,
                                             p.normal_unemployment))
            wbar = float(np.mean([f.wage for f in firms]))
            income[s] = float(B.expected_income(wbar, D, pool))
            demand[s] = D
        # people reconsider their occupation: mostly the unemployed
        rng = self.rng
        unemployed = self.employer < 0
        chance = np.where(unemployed, p.mobility_unemployed, p.mobility_employed)
        movers = np.where(rng.random(len(self.size)) < chance)[0]
        weights = np.maximum(demand, 1e-9) * np.maximum(income, 0.0) ** p.mobility_sensitivity
        if weights.sum() > 0 and len(movers):
            choice = rng.choice(self.n_sec, size=len(movers), p=weights / weights.sum())
            # switch only if the new occupation pays more even after the ticks lost retraining
            keep_time = max(1.0 - p.retrain_ticks / p.switch_horizon, 0.0)
            worth = income[choice] * keep_time > income[self.sector[movers]]
            ok = (choice != self.sector[movers]) & worth
            switch = movers[ok]
            self.sector[switch] = choice[ok]
            self.employer[switch] = -1
            self.retrain[switch] = p.retrain_ticks
        self.retrain = np.maximum(self.retrain - 1, 0)
        world.vars["expected_income"] = income

    # ================================================================== invariants and metrics
    def invariants(self):
        out = [i for i in super().invariants() if i.name != "labor_pools"]

        def employment(world, ctx):
            v = []
            if world.tick == 0:
                return v                   # nobody has been hired yet
            known = {f.id for f in self.firms}
            if ((self.employer >= 0) & ~np.isin(self.employer, list(known))).any():
                v.append(Violation("employment", "someone works for a firm that does not exist"))
            for f in self.firms:
                # people can leave during the tick (job switching, exit) but never join outside hiring
                if float(self.size[self.employer == f.id].sum()) > f.hired * (1 + 1e-9) + 1e-9:
                    v.append(Violation("employment", f"{f.account}: more employees than it hired ({f.hired})"))
            for s in range(self.n_sec):
                used = sum(f.hired + f.builders for f in self.of(s))
                if used > self.pools_at_hire[s] * (1 + 1e-9) + 1e-9:
                    v.append(Violation("labor_pools", f"{self.sector_names[s]}: {used:.6g} used, pool {self.pools_at_hire[s]:.6g}"))
            if (self.size <= 0).any():
                v.append(Violation("employment", "agent with no people"))
            return v

        def services_physics(world, ctx):
            if not self.of(SERVICES):
                return []
            sv = self.services
            v = []
            for f in self.of(SERVICES):
                made = sum(p_.amount for p_ in ctx.journal if p_.flow == "production" and p_.item == "services"
                           and p_.debit == f.account)
                if made > max(f.hired - sv.fixed_crew, 0.0) / sv.labor_per_unit * (1 + 1e-9) + 1e-9:
                    v.append(Violation("physics", f"{f.account}: served {made:.6g} beyond its staff", ("services",)))
            return v

        return out + [Invariant("employment", employment), Invariant("services_physics", services_physics)]

    def metrics(self):
        base = {m.name: m for m in super().metrics()}

        def shares(w):
            N = float(self.size.sum()) or 1.0
            out = {}
            for s in range(self.n_sec):
                out[self.sector_names[s]] = sum(f.hired for f in self.of(s)) / N
            out["building"] = sum(f.builders for f in self.firms) / N
            out["retraining"] = float(self.size[self.retrain > 0].sum()) / N
            if w.state["economy"].get("gov_layer") is not None:
                out["public"] = w.vars.get("public_workers", 0.0) / N
            out["unemployed"] = max(0.0, 1.0 - sum(out.values()))
            return out

        for name in ("labor_share", "unemployment_rate", "occupations", "population", "expected_income"):
            base.pop(name, None)
        firm_count = lambda w: {self.sector_names[s]: len(self.of(s)) for s in range(self.n_sec)}
        extra = [
            Metric("population", lambda w: float(self.size.sum()), "people", "people"),
            Metric("labor_share", shares, "share of people per activity", "share"),
            Metric("unemployment_rate", lambda w: shares(w)["unemployed"] + shares(w)["retraining"],
                   "share of people without work (incl. retraining)", "share"),
            Metric("occupations", lambda w: {self.sector_names[s]: float(self.pools()[s]) for s in range(self.n_sec)},
                   "people per occupation", "people"),
            Metric("firms", firm_count, "number of firms per sector", "count"),
            Metric("concentration", lambda w: {self.sector_names[s]: self.hhi(s) for s in range(self.n_sec) if self.of(s)},
                   "Herfindahl index of sales per sector (1 = monopoly)", "index"),
            Metric("markups", lambda w: {self.sector_names[s]: float(np.mean([f.markup for f in self.of(s)]))
                                         for s in range(self.n_sec) if self.of(s)}, "average markup", "share"),
            Metric("gini_income", lambda w: gini(self.income / self.size, self.size), "Gini of income per person", "index"),
            Metric("gini_wealth", lambda w: gini(self.cash / self.size, self.size), "Gini of money held per person", "index"),
            Metric("top10_wealth_share", lambda w: top_share(self.cash, self.size, 0.1), "money held by the richest 10%", "share"),
            Metric("profit_share", lambda w: sum(max(f.last_profit, 0.0) for f in self.firms) /
                   max(sum(f.revenue for f in self.of(DIST)) + sum(f.revenue for f in self.of(SERVICES)), 1e-9),
                   "profits / household spending", "share"),
            Metric("hungry_share", lambda w: float(self.size[self.hunger > 0.01].sum() / max(self.size.sum(), 1e-12)),
                   "share of people who did not get all the bread they need", "share"),
            Metric("services_per_person", lambda w: w.vars.get("services_per_person", 0.0), "services bought per person", "units"),
            Metric("entries", lambda w: self.entries, "firms that entered so far", "count"),
            Metric("exits", lambda w: self.exits, "firms that exited so far", "count"),
            Metric("expected_income", lambda w: {self.sector_names[s]: float(w.vars["expected_income"][s])
                                                 for s in range(self.n_sec)}, "wage x chance of a job", "money"),
        ]
        return list(base.values()) + extra


def top_share(values, weights, top: float) -> float:
    v = np.asarray(values, float) / np.maximum(np.asarray(weights, float), 1e-12)
    w = np.asarray(weights, float)
    if w.size == 0 or w.sum() <= 0:
        return 0.0
    order = np.argsort(-v)
    cw = np.cumsum(w[order])
    k = cw <= top * cw[-1] + 1e-9
    tot = (v * w).sum()
    return float((v[order][k] * w[order][k]).sum() / tot) if tot > 0 else 0.0
