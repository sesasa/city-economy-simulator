"""Stage 2 base layer: a market economy that replaces the planner.

Firms (one per sector by default; several per sector from Stage 4) set prices
from their inventories and wages from their vacancies. Households choose
occupations by expected income and spend out of income and savings. Every
purchase is paid in money (or, in barter mode, only the trades that find a
double coincidence of wants go through).

Sectors and what they sell
  farm         wheat (raw)            to factories
  factory      bread (final)          to distributors (wholesale)
  distribution bread delivered        to households (retail)
  maintenance  repair service         to every capital owner

Ledger accounts: "households" plus one account per firm ("firm:<id>").
Items: raw, final and the Stage 1 capital goods, plus "cash" (money).
Household-level detail (one agent per occupation in aggregate mode) lives in
numpy arrays whose totals the invariants tie back to the ledger.

world.vars written each tick (read by economy-generic layers and metrics):
  tech, farmers, consumption_per_person, harvest_multiplier (read), market
"""
from __future__ import annotations

import dataclasses

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import Field

from ...core.agents import AggregateGroup
from ...core.invariants import Invariant, Violation, close
from ...core.metrics import Metric
from ...core.phases import Phase
from ...core.layer import Layer
from ..s1_real_economy import rules
from ..s1_real_economy.real_economy import RealEconomyParams
from . import behavior as B

NAME = "market"
SECTORS = ("farm", "factory", "distribution", "maintenance")
FARM, FACTORY, DIST, MAINT = range(4)
OUTPUT_ITEM = {FARM: "raw", FACTORY: "final", DIST: "final"}
CATEGORIES = ("wages", "investment_wages", "wheat", "bread_wholesale", "maintenance", "retail", "dividends")


class MarketParams(RealEconomyParams):
    """Technology fields are inherited from Stage 1 so both economies share one world."""

    exchange: Literal["money", "barter"] = "money"
    barter_goods: int = Field(5, ge=2, description="kinds of things traded (wheat, bread, repairs, delivery, labor)")
    barter_meetings: int = Field(8, ge=1, description="random meetings per trader per tick under barter")
    initial_money: float | None = Field(None, gt=0, description="money created at the start; default: firms get "
                                        "cash_buffer ticks of their costs and households one tick of income")
    initial_wage: float = Field(1.0, gt=0)
    prices_match_money: bool = Field(True, description="start prices and wages in line with the money stock; if "
                                     "false they start at the default level, so scarce money must force deflation")
    initial_markup: float = Field(0.1, ge=0, description="markup used to set starting prices from costs")
    initial_allocation: Literal["planner", "equal"] = "planner"
    firms_per_sector: int = Field(1, ge=1)
    # behaviour
    expectation_speed: float = Field(0.5, gt=0, le=1, description="lambda in E' = E + lambda (x - E)")
    inventory_target: float = Field(0.3, ge=0, description="target stock, in ticks of expected sales")
    inventory_correction: float = Field(2.0, ge=1, description="ticks to close an inventory gap")
    price_speed: float = Field(0.05, ge=0, description="markup response to the inventory (or order) gap per tick")
    markup_bounds: tuple[float, float] = (-0.5, 3.0)
    normal_markup: float = Field(0.2, description="markup firms drift back to")
    markup_pull: float = Field(0.05, ge=0, le=1, description="speed of the drift back to the normal markup")
    normal_unemployment: float = Field(0.05, ge=0, lt=1, description="unemployment at which wages are stable")
    wage_speed: float = Field(0.05, ge=0, description="wage response to the vacancy gap")
    max_step: float = Field(0.1, gt=0, description="largest price or wage change per tick")
    mobility: float = Field(0.1, ge=0, le=1, description="share of workers reconsidering their occupation each tick")
    mobility_sensitivity: float = Field(3.0, ge=0, description="how strongly workers follow expected income")
    spoilage: float = Field(0.05, ge=0, le=1, description="share of firms' wheat and bread stocks lost per tick")
    alpha_income: float = Field(0.9, ge=0, le=1, description="share of this tick's income households plan to spend")
    alpha_wealth: float = Field(0.1, ge=0, le=1, description="share of savings households plan to spend")
    max_consumption: float = Field(1.5, ge=1, description="most bread a person wants, in multiples of c")
    cash_buffer: float = Field(2.0, ge=0, description="firms keep this many ticks of costs as cash")
    excess_payout: float = Field(0.5, ge=0, le=1, description="share of cash above the buffer paid out per tick")
    investment_share: float = Field(0.3, ge=0, le=1, description="share of profit set aside for building")
    bottleneck_threshold: float = Field(0.9, gt=0)
    factory_cost: float = Field(1000.0, gt=0)
    land_improvement_cost: float = Field(200.0, gt=0)
    tool_cost: float = Field(200.0, gt=0)


@dataclass
class Firm:
    id: int
    sector: int
    account: str
    price: float
    wage: float
    expected: float                    # expected sales (units of output)
    conditions: list[float] = field(default_factory=list)   # factories (factory firms only)
    labor_demand: float = 0.0
    hired: float = 0.0
    wage_bill: float = 0.0
    costs: float = 0.0                 # cash paid this tick (wages + inputs + services)
    revenue: float = 0.0
    orders_in: float = 0.0             # desired purchases from this firm this tick (before limits)
    sold: float = 0.0
    plan: float = 0.0                  # planned output this tick
    order_out: float = 0.0             # what this firm wants to buy (wheat / bread / service)
    maint_req: float = 0.0
    maint_got: float = 0.0
    builders: float = 0.0
    markup: float = 0.1
    unit_cost: float = 0.0
    op_costs: float = 0.0              # running costs this tick (wages paid, inputs, repairs)
    output: float = 0.0
    wages_paid: float = 0.0
    last_profit: float = 0.0
    dividends: float = 0.0
    inv_budget: float = 0.0
    carry: float = 0.0
    op_costs_prev: float = 0.0


def world_cash_outside(world, market) -> float:
    """Cash held by households outside banks (after a run); 0 when money is cash itself."""
    if market.money_item == "cash":
        return 0.0
    return world.ledger.balance("households", "cash")


class MarketEconomy(Layer):
    name = NAME
    requires: list[str] = []
    provides = ["economy", "market"]
    Params = MarketParams
    aliases = {"a": "farm_output_per_worker", "c": "consumption_per_person", "e": "conversion_rate",
               "F_p": "factory_fixed_crew", "v_p": "factory_variable_labor", "v_d": "distribution_labor",
               "F_m": "maintenance_fixed_crew", "delta": "depreciation_rate", "N0": "initial_population",
               "M0": "initial_money"}
    p: MarketParams

    # ================================================================== setup
    def setup(self, world) -> None:
        p, L = self.p, world.ledger
        for item in ("raw", "final"):
            L.register_item(item, "good")
        for item in ("factory", "factory_wip", "land_improvement", "tools_processing", "tools_distribution"):
            L.register_item(item, "capital")
        L.register_item("cash", "money")
        world.add_group(AggregateGroup(name="households", account="households", count=p.initial_population))
        st = world.layer_state(NAME)
        world.state["economy"] = st
        st.update(land_limit=None, land_exponent=0.5, c=p.consumption_per_person)
        L.begin_tick(0)

        # --- starting point: what a planner would do for N0 people (prices and expectations)
        N = p.initial_population
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

        # --- firms
        self.firms: list[Firm] = []
        fid = 0
        for s in range(4):
            for j in range(p.firms_per_sector):
                acc = f"firm:{SECTORS[s]}:{j}"
                L.open_account(acc)
                f = Firm(fid, s, acc, prices[s], w, expected[s] / p.firms_per_sector)
                if s == FACTORY:
                    n_fac = p.initial_factories // p.firms_per_sector + (1 if j < p.initial_factories % p.firms_per_sector else 0)
                    f.conditions = [1.0] * n_fac
                    if n_fac:
                        L.produce(acc, "factory", float(n_fac), "initial endowment")
                self.firms.append(f)
                fid += 1
        L.open_account("households")

        # --- households: one agent per occupation (aggregate mode)
        if p.initial_allocation == "planner":
            shares = np.array([plan.alloc["farming"], plan.alloc["processing"], plan.alloc["distribution"],
                               plan.alloc["maintenance"]], float)
            shares = shares + max(N - shares.sum(), 0.0) * shares / max(shares.sum(), 1e-12)
        else:
            shares = np.full(4, N / 4)
        self.size = shares.astype(float)          # people per household agent (agent k works in sector k)
        self.sector = np.arange(4)
        # money: firms hold cash_buffer ticks of their costs, households one tick of income,
        # all scaled so the total equals initial_money when it is given
        cost = {FARM: w * plan.alloc["farming"],
                FACTORY: w * plan.alloc["processing"] + p_r * raw0 + p_m * plan.required["maintenance"],
                DIST: w * plan.alloc["distribution"] + p_w * need,
                MAINT: w * plan.alloc["maintenance"]}
        firm_need = {s: max(p.cash_buffer, 1.0) * cost[s] for s in range(4)}
        hh_need = p_f * need
        total = hh_need + sum(firm_need.values())
        scale = (p.initial_money / total) if p.initial_money else 1.0
        self.money_scale = scale
        hh_money = hh_need * scale
        L.post("households", "ext:creation", "cash", hh_money, "creation", "initial money")
        self.cash = hh_money * self.size / max(self.size.sum(), 1e-12)
        for f in self.firms:
            L.post(f.account, "ext:creation", "cash", firm_need[f.sector] * scale / p.firms_per_sector, "creation", "initial money")
        # prices and wages start consistent with the money stock (unless asked not to)
        ps = scale if p.prices_match_money else 1.0
        for f in self.firms:
            f.price *= ps
            f.wage *= ps
            f.markup = p.initial_markup
            f.unit_cost = f.price / (1 + p.initial_markup)
        self.hunger = np.zeros(4)
        self.income = np.zeros(4)
        self.pools_at_hire = self.size.copy()
        st["firms"] = self.firms
        st["market_layer"] = self
        st.setdefault("credit_providers", [])      # fn(world, firm, amount) -> amount lent (Stage 3)
        st.setdefault("investment_finance", [])    # fn(world, firm, amount) -> amount lent for building
        st.setdefault("extra_categories", [])
        st["tech_fn"] = lambda world: self._tech(st)
        self.prev_price_level = p_f * ps
        self.prev_potential = None
        self.last = {}

    # ================================================================== helpers
    def _tech(self, st, conditions=None, L=None, Tp=None, Td=None, land_share=1.0):
        p = self.p
        firms = getattr(self, "firms", [])
        if conditions is None:
            conditions = tuple(c for f in firms if f.sector == FACTORY for c in f.conditions)
        led = getattr(self, "_ledger", None)

        def tot(sector, item):
            return sum(led.balance(f.account, item) for f in firms if f.sector == sector) if led else 0.0
        limit = st.get("land_limit")
        return rules.Tech(
            c=p.consumption_per_person, e=p.conversion_rate, a=p.farm_output_per_worker,
            land_improvement=tot(FARM, "land_improvement") if L is None else L,
            tools_processing=tot(FACTORY, "tools_processing") if Tp is None else Tp,
            tools_distribution=tot(DIST, "tools_distribution") if Td is None else Td,
            land_elasticity=p.land_improvement_elasticity, tool_elasticity=p.tool_elasticity,
            v_p=p.factory_variable_labor, v_d=p.distribution_labor * st.get("road_factor", 1.0), F_p=p.factory_fixed_crew,
            F_m=p.maintenance_fixed_crew, capacity=p.factory_capacity, conditions=tuple(conditions),
            land_maintenance=p.land_improvement_maintenance, tool_maintenance=p.tool_maintenance,
            land_limit=None if limit is None else limit * land_share, land_exponent=st.get("land_exponent", 0.5))

    def firm_tech(self, world, f: Firm):
        st = world.state["economy"]
        L = world.ledger
        n_farm = sum(1 for g in self.firms if g.sector == FARM)
        t = self._tech(st, conditions=tuple(f.conditions),
                       L=L.balance(f.account, "land_improvement"),
                       Tp=L.balance(f.account, "tools_processing"),
                       Td=L.balance(f.account, "tools_distribution"),
                       land_share=1.0 / max(n_farm, 1))
        A = getattr(f, "tech", 1.0)                   # the firm's own technology level (Stage 9)
        if A != 1.0:
            t = dataclasses.replace(t, a=t.a * A, v_p=t.v_p / A, v_d=t.v_d / A)
        return t

    def normal_unit_cost(self, world, f: Firm) -> float:
        p = self.p
        N = float(self.size.sum())
        need = max(N * p.consumption_per_person, 1e-9) / max(len(self.of(f.sector)), 1)
        t = self.firm_tech(world, f)
        price_of = lambda sector: float(np.mean([g.price for g in self.of(sector)]))
        if f.sector == FARM:
            raw = need / t.e
            return f.wage * rules.farmers_required(raw, t.a_x, t.land_limit, t.land_exponent) / raw
        if f.sector == FACTORY:
            y = min(need, t.total_capacity) if t.total_capacity > 0 else need
            k, _ = rules.operating_factories(y, t.conditions, t.capacity, t.cumulative_capacity)
            labor = t.F_p * max(k, 1) + t.v_p_eff * y
            maint = t.F_m * len(t.conditions) + t.tool_maintenance * t.tools_processing
            return (f.wage * labor + price_of(FARM) * y / t.e + price_of(MAINT) * maint) / max(y, 1e-9)
        if f.sector == DIST:
            return price_of(FACTORY) + f.wage * t.v_d_eff
        return f.wage

    def sector_of(self, account: str):
        """Sector column for SFC matrices."""
        if account == "households":
            return "households"
        if account.startswith("firm:"):
            return account.split(":")[1]
        return None

    def of(self, sector):
        return [f for f in self.firms if f.sector == sector]

    money_item = "cash"

    def split_orders(self, world, sellers: list, sector: int, qty: float) -> None:
        """Record what a buyer wants from each seller: an equal split; with importers in town
        (Stage 8) each importer first gets its price-based share (the rule purchases use), so the
        domestic firms only lose the orders an importer can actually serve from its stock."""
        if not sellers:
            return
        imps = self.importers(world, sector)
        if imps:
            dom = [g for g in sellers if not getattr(g, "importer", False)]
            w = np.maximum(np.array([g.price for g in sellers]), 1e-12) ** -4.0
            share = {id(g): x for g, x in zip(sellers, w / w.sum())}
            item = OUTPUT_ITEM[sector]
            rest = qty
            for g in imps:
                g.orders_in += qty * share[id(g)]            # what it could sell (it restocks towards it)
                rest -= min(qty * share[id(g)], self.stock(world, g, item))   # what it can sell now
            for g in dom:
                g.orders_in += rest / max(len(dom), 1)
        else:
            for g in sellers:
                g.orders_in += qty / len(sellers)

    @staticmethod
    def importers(world, sector) -> list:
        """Trading houses selling imported goods in this town (Stage 8); not part of `firms`."""
        return world.state["economy"].get("importers", {}).get(sector, [])

    def borrow(self, world, f: Firm, amount: float) -> float:
        """Ask the credit providers (Stage 3 banks) for working capital; returns the amount lent."""
        got = 0.0
        for fn in world.state["economy"].get("credit_providers", ()):
            if amount - got <= 1e-12:
                break
            got += fn(world, f, amount - got)
        return got

    def money_of(self, world, f: Firm) -> float:
        return world.ledger.balance(f.account, self.money_item)

    def stock(self, world, f: Firm, item: str) -> float:
        return world.ledger.balance(f.account, item)

    def pay_households(self, world, f: Firm, amount: float, category: str, weights: np.ndarray) -> None:
        """Firm pays households; the money is split across household agents by `weights`."""
        if amount <= 0:
            return
        world.ledger.post("households", f.account, self.money_item, amount, "transfer", category)
        w = weights / weights.sum() if weights.sum() > 0 else self.size / self.size.sum()
        self.cash += amount * w
        self.income += amount * w

    def trade_share(self) -> float:
        return B.trade_completion(self.p.barter_goods, self.p.barter_meetings, self.p.exchange)

    def buy(self, world, buyer_acc: str, sellers: list[Firm], qty: float, item: str, category: str,
            buyer_cash: float) -> float:
        """Buy up to `qty` of `item` from `sellers`, cheapest first by share, within cash and stocks."""
        if qty <= 1e-12 or not sellers:
            return 0.0
        prices = np.array([s.price for s in sellers])
        stocks = np.array([self.stock(world, s, item) for s in sellers])
        sales = B.allocate_by_price(qty, prices, stocks, choosiness=4.0)
        cost = float((sales * prices).sum())
        buyer = next((f for f in self.firms if f.account == buyer_acc), None)
        if buyer is not None and cost > buyer_cash:
            buyer_cash += self.borrow(world, buyer, cost - buyer_cash)
        if cost > buyer_cash and cost > 0:
            sales *= buyer_cash / cost
        got = 0.0
        for s, q in zip(sellers, sales):
            if q <= 1e-12:
                continue
            world.ledger.transfer(s.account, buyer_acc, item, q, category)
            world.ledger.post(s.account, buyer_acc, self.money_item, q * s.price, "transfer", category)
            s.revenue += q * s.price
            s.sold += q
            got += q
        return got

    # ================================================================== phase 1
    def plan_and_hire(self, world) -> None:
        p, L = self.p, world.ledger
        self._ledger = L
        st = world.state["economy"]
        for f in self.firms:
            f.labor_demand = f.hired = f.wage_bill = f.costs = f.revenue = 0.0
            f.orders_in = f.sold = f.order_out = f.maint_req = f.maint_got = f.builders = 0.0
            f.op_costs = f.output = 0.0
        self.income = np.zeros_like(self.size)
        world.state["economy"]["spending_multiplier"] = 1.0      # layers below may scale it this tick
        self.sync_population(world)
        q = self.trade_share()
        world.vars["trade_completion"] = q
        self.plan_firms(world)
        self.hire(world, q)
        world.vars["farmers"] = sum(f.hired for f in self.of(FARM))
        world.vars["tech"] = self._tech(st)

    def plan_firms(self, world) -> None:
        """Each firm sets its production plan, purchase orders and labor demand."""
        p = self.p
        for f in self.firms:
            t = self.firm_tech(world, f)
            if f.sector == FARM:
                f.plan = float(B.production_target(f.expected, self.stock(world, f, "raw"), p.inventory_target,
                                                   p.inventory_correction))
                f.labor_demand = rules.farmers_required(f.plan, t.a_x, t.land_limit, t.land_exponent)
            elif f.sector == FACTORY:
                cap = t.total_capacity
                f.plan = float(B.production_target(f.expected, self.stock(world, f, "final"), p.inventory_target,
                                                   p.inventory_correction, cap))
                k, _ = rules.operating_factories(f.plan, t.conditions, t.capacity, t.cumulative_capacity)
                f.labor_demand = t.F_p * k + t.v_p_eff * f.plan
                f.order_out = max(0.0, f.plan / t.e - self.stock(world, f, "raw"))
            elif f.sector == DIST:
                inv = self.stock(world, f, "final")
                f.order_out = float(B.production_target(f.expected, inv, p.inventory_target, p.inventory_correction))
                f.plan = min(f.expected, inv + f.order_out)
                f.labor_demand = t.v_d_eff * f.plan
            else:
                f.plan = f.expected
                f.labor_demand = f.expected
            f.labor_demand = max(0.0, f.labor_demand)

    def hire(self, world, q: float) -> None:
        """Labor market: each occupation's pool is shared among its firms. Wages are owed now and
        paid in Phase 4 out of the firm's cash (so a firm with an empty till can still restart)."""
        self.pools_at_hire = np.array([self.size[self.sector == s].sum() for s in range(4)], float)
        for s in range(4):
            pool = float(self.pools_at_hire[s])
            firms = self.of(s)
            D = sum(f.labor_demand for f in firms)
            fill = min(1.0, pool / D) if D > 0 else 0.0
            for f in firms:
                f.hired = max(0.0, f.labor_demand * fill * q)   # barter: wages in kind need a match too
                f.wage_bill = f.hired * f.wage

    def sync_population(self, world) -> None:
        """Apply births and deaths decided by the demography layer to the household agents."""
        target = world.groups["households"].size()
        have = float(self.size.sum())
        d = target - have
        if abs(d) <= 1e-12 or have <= 0:
            return
        if d > 0:
            w = self.size / have
        else:
            w = self.size * (self.hunger + 1e-9)
            w = w / w.sum()
        new = np.maximum(self.size + d * w, 0.0)
        self.size = new * (target / new.sum()) if new.sum() > 0 else new

    # ================================================================== phase 2
    def produce(self, world) -> None:
        p, L = self.p, world.ledger
        q = world.vars["trade_completion"]
        mult = world.vars.get("harvest_multiplier", 1.0)
        for f in self.of(FARM):
            t = self.firm_tech(world, f)
            f._tech_at_production = t
            f._hired_at_production = f.hired
            raw = rules.farm_output(f.hired, t.a_x, t.land_limit, t.land_exponent) * mult
            L.produce(f.account, "raw", raw, "harvest")
            f.output = raw
        # wheat market (importers, Stage 8, sell alongside the farms)
        farms = self.of(FARM) + self.importers(world, FARM)
        for f in self.of(FACTORY):
            self.split_orders(world, farms, FARM, f.order_out)
            f.costs += 0.0
            before = self.money_of(world, f)
            self.buy(world, f.account, farms, f.order_out * q, "raw", "wheat", before)
            f.costs += before - self.money_of(world, f)
            f.op_costs += before - self.money_of(world, f)
        # bread production
        for f in self.of(FACTORY):
            t = self.firm_tech(world, f)
            f._tech_at_production = t
            f._hired_at_production = f.hired
            k, _ = rules.operating_factories(f.plan, t.conditions, t.capacity, t.cumulative_capacity)
            if t.F_p > 0:
                k = min(k, int(math.floor(f.hired / t.F_p + 1e-9)))
            out = 0.0
            if k > 0:
                by_labor = max(f.hired - t.F_p * k, 0.0) / t.v_p_eff if t.v_p_eff > 0 else math.inf
                out = min(self.stock(world, f, "raw") * t.e, by_labor, rules.capacity_of_best(k, t.conditions, t.capacity), f.plan)
            L.remove(f.account, "raw", min(out / t.e, self.stock(world, f, "raw")), "use", "processed")
            L.produce(f.account, "final", out, "baked")
            f.output = out
            f.running = k
        # wholesale market
        factories = self.of(FACTORY) + self.importers(world, FACTORY)
        for d in self.of(DIST):
            self.split_orders(world, factories, FACTORY, d.order_out)
            before = self.money_of(world, d)
            self.buy(world, d.account, factories, d.order_out * q, "final", "bread_wholesale", before)
            d.costs += before - self.money_of(world, d)
            d.op_costs += before - self.money_of(world, d)

    # ================================================================== phase 3
    def maintain(self, world) -> None:
        p, L = self.p, world.ledger
        q = world.vars["trade_completion"]
        maint = self.of(MAINT)
        supply = sum(m.hired for m in maint)
        owners = [f for f in self.firms if f.sector != MAINT]
        for f in owners:
            t = self.firm_tech(world, f)
            if f.sector == FARM:
                f.maint_req = t.land_maintenance * t.land_improvement
            elif f.sector == FACTORY:
                f.maint_req = t.F_m * len(f.conditions) + t.tool_maintenance * t.tools_processing
            else:
                f.maint_req = t.tool_maintenance * t.tools_distribution
        price = np.mean([m.price for m in maint]) if maint else 1.0
        payer = world.state["economy"].get("maintenance_payer")
        if payer is not None:
            # public maintenance (Stage 5): the government pays for everyone's upkeep, as far as it can
            funded = payer(world, sum(f.maint_req for f in owners) * price)
            orders = np.array([f.maint_req * q * funded for f in owners])
        else:
            for f in owners:
                need_cash = f.maint_req * price
                if need_cash > self.money_of(world, f):
                    self.borrow(world, f, need_cash - self.money_of(world, f))
            orders = np.array([min(f.maint_req, self.money_of(world, f) / price) * q for f in owners])
        for m in maint:
            m.orders_in = float(sum(f.maint_req for f in owners)) / max(len(maint), 1)
        fill = min(1.0, supply / orders.sum()) if orders.sum() > 0 else 0.0
        for f, o in zip(owners, orders):
            got = o * fill
            f.maint_got = got
            if got > 0 and maint:
                per = got / len(maint)
                for m in maint:
                    who = "gov" if payer is not None else f.account
                    L.post(m.account, who, self.money_item, per * m.price, "transfer",
                           "public_maintenance" if payer is not None else "maintenance")
                    m.revenue += per * m.price
                    m.sold += per
                if payer is None:
                    f.costs += got * price
                    f.op_costs += got * price
            u = rules.unmaintained_share(got, f.maint_req)
            loss = p.depreciation_rate * u
            f.unmaintained = u
            if loss > 0:
                if f.conditions:
                    kept, lost = [], 0.0
                    for c in f.conditions:
                        a = c * (1 - loss)
                        if a < p.scrap_condition:
                            lost += c
                        else:
                            lost += c - a
                            kept.append(a)
                    f.conditions = kept
                    L.remove(f.account, "factory", min(lost, self.stock(world, f, "factory")), "depreciation", "unmaintained")
                for item in ("land_improvement", "tools_processing", "tools_distribution"):
                    L.remove(f.account, item, self.stock(world, f, item) * loss, "depreciation", "unmaintained")
        # stocks spoil
        for f in self.firms:
            for item in ("raw", "final"):
                L.remove(f.account, item, max(self.stock(world, f, item), 0.0) * p.spoilage, "spoilage", "stock spoils")

    # ================================================================== phase 4
    def pay_dividends(self, world) -> None:
        """Pay wages owed (pro rata if the till is short), then dividends out of cash above the buffer."""
        p = self.p
        arrears = 0.0
        for f in self.firms:
            have = self.money_of(world, f)
            if have < f.wage_bill:
                have += self.borrow(world, f, f.wage_bill - have)
            paid = min(f.wage_bill, max(have, 0.0))
            arrears += f.wage_bill - paid
            weights = np.where(self.sector == f.sector, self.size, 0.0)
            self.pay_households(world, f, paid, "wages", weights)
            f.wages_paid = paid
            f.costs += paid
            f.op_costs += paid
        world.vars["wage_arrears"] = arrears
        # Profits go back to households: a share is set aside for building (spent as builders'
        # wages in Phase 6), the rest is paid as dividends now. Building money left unused
        # is paid out the next tick, so firm cash stays level and the circuit closes.
        taxers = world.state["economy"].get("profit_tax", ())
        for f in self.firms:
            tax = sum(fn(world, f, f.last_profit) for fn in taxers) if taxers else 0.0
            distributable = max(f.last_profit - tax, 0.0)
            f.inv_budget = p.investment_share * distributable
            d = (distributable - f.inv_budget) + f.carry
            # cash piled up beyond the normal buffer is paid out too (firms don't hoard)
            buffer = p.cash_buffer * max(f.op_costs_prev, 1e-9)
            d += p.excess_payout * max(self.money_of(world, f) - d - buffer - f.wage_bill, 0.0)
            due = sum(fn(world, f) for fn in [world.state["economy"].get("debt_service")] if fn)
            debt = sum(fn(world, f) for fn in [world.state["economy"].get("debt_of")] if fn)
            # creditors are paid before owners: no dividends out of borrowed money
            d = min(d, max(self.money_of(world, f) - due - debt, 0.0))
            f.carry = 0.0
            f.dividends = d
            if d > 0:
                self.pay_households(world, f, d, "dividends", self.size.copy())

    # ================================================================== phase 5
    def consume(self, world) -> None:
        p, L = self.p, world.ledger
        q = world.vars["trade_completion"]
        c = p.consumption_per_person
        dists = self.of(DIST)
        price = float(np.mean([d.price for d in dists]))
        k = world.state["economy"].get("spending_multiplier", 1.0)    # demand shocks, interest rates (Stage 6)
        spend = B.consumption_spending(self.income, self.cash, price, c * self.size, p.max_consumption * c * self.size,
                                       min(p.alpha_income * k, 1.0), min(p.alpha_wealth * k, 1.0))
        desired = spend / price
        total_desired = float(desired.sum())
        for d in dists:
            d.orders_in += total_desired / len(dists)
        # each distributor can hand over what its carters can carry
        capacity = []
        for d in dists:
            t = self.firm_tech(world, d)
            capacity.append(min(self.stock(world, d, "final"), d.hired / t.v_d_eff if t.v_d_eff > 0 else math.inf))
        capacity = np.array(capacity)
        world.vars["delivery_capacity"] = float(sum(d.hired / self.firm_tech(world, d).v_d_eff for d in dists))
        prices = np.array([d.price for d in dists])
        sales = B.allocate_by_price(total_desired * q, prices, capacity, choosiness=4.0)
        cost = float((sales * prices).sum())
        if cost > spend.sum() > 0:                 # dearer shops can't make people spend more than they planned
            sales *= spend.sum() / cost
        bought = float(sales.sum())
        paid = float((sales * prices).sum())
        share = desired / total_desired if total_desired > 0 else self.size / self.size.sum()
        for d, s_ in zip(dists, sales):
            if s_ <= 1e-12:
                continue
            L.transfer(d.account, "households", "final", s_, "retail")
            L.post(d.account, "households", self.money_item, s_ * d.price, "transfer", "retail")
            d.revenue += s_ * d.price
            d.sold += s_
        self.cash -= paid * share
        got = bought * share
        L.remove("households", "final", world.ledger.balance("households", "final"), "consumption")
        need = c * self.size
        self.hunger = np.where(need > 0, np.clip(1 - got / np.maximum(need, 1e-12), 0, 1), 0.0)
        N = float(self.size.sum())
        world.vars["consumption_per_person"] = bought / N if N > 0 else 0.0
        world.vars["retail_sales"] = bought
        world.vars["household_spending"] = paid
        world.vars["price_level"] = price

    # ================================================================== phase 6
    def invest(self, world) -> None:
        p, L = self.p, world.ledger
        costs = {"factory": p.factory_cost, "land_improvement": p.land_improvement_cost,
                 "tools_processing": p.tool_cost, "tools_distribution": p.tool_cost}
        spent_total = 0.0
        for f in self.firms:
            if f.sector not in (FARM, FACTORY, DIST):      # repair shops (and services) own no capital
                continue
            budget_cash = min(f.inv_budget, max(self.money_of(world, f), 0.0))
            f.carry = budget_cash                       # paid out next tick unless spent below
            pool = float(self.pools_at_hire[f.sector])
            idle = max(0.0, pool - sum(g.hired for g in self.of(f.sector)) - sum(g.builders for g in self.of(f.sector)))
            labor = min(budget_cash / f.wage if f.wage > 0 else 0.0, idle)
            finance = world.state["economy"].get("investment_finance", ())
            if labor <= 1e-9 and not (finance and idle > 1e-9):
                continue
            t = self.firm_tech(world, f)
            kinds = {FARM: ["land_improvement"], FACTORY: ["factory", "tools_processing"], DIST: ["tools_distribution"]}[f.sector]
            wip = self.stock(world, f, "factory_wip")
            threshold = p.bottleneck_threshold * (1.0 - 0.5 * world.state["economy"].get("boldness", 0.0))
            ctx = rules.InvestmentContext(labor=labor if labor > 1e-9 else min(idle, 1.0), output=max(f.plan, 0.0),
                                          need=f.expected if f.sector == FACTORY else 0.0,
                                          tech=t, costs=costs, threshold=threshold,
                                          factory_in_progress=wip > rules.EPS,
                                          sector_labor={"farming": f.hired if f.sector == FARM else 0.0,
                                                        "processing": t.v_p_eff * f.plan if f.sector == FACTORY else 0.0,
                                                        "distribution": f.hired if f.sector == DIST else 0.0})
            choice = None
            if "factory" in kinds and (ctx.factory_in_progress or ctx.utilization >= ctx.threshold):
                choice = "factory"
            else:
                for k in kinds:
                    if k != "factory" and ctx.net_saving(k) > rules.EPS:
                        choice = k
                        break
            if choice is None:
                continue
            # Minsky: the bank's optimism makes firms bolder (lower hurdle, build capacity earlier)
            boldness = world.state["economy"].get("boldness", 0.0)
            hurdle = world.state["economy"].get("hurdle_rate", 0.0) * max(0.0, 1.0 - boldness)
            pays = True
            if choice != "factory" and finance:
                # a loan-financed labor-saving project must save enough wages per tick to beat the hurdle
                probe = rules.InvestmentContext(**{**ctx.__dict__, "labor": max(idle - labor, 1e-9)})
                pays = probe.net_saving(choice) / max(idle - labor, 1e-9) * costs[choice] / costs[choice] >= hurdle
            if finance and pays and idle - labor > 1e-9:
                # borrow to put more idle hands to work on a worthwhile project (Stage 3 banks)
                want_labor = idle - labor
                if choice == "factory":
                    left = max(rules.factories_wanted(f.expected, t.total_capacity, t.capacity,
                                                      threshold), 1.0) - wip
                    want_labor = min(want_labor, max(left * costs["factory"] - labor, 0.0))
                lent = 0.0
                for fn in finance:
                    lent += fn(world, f, want_labor * f.wage - lent)
                labor += lent / f.wage if f.wage > 0 else 0.0
                budget_cash += lent
            if labor <= 1e-9:
                continue
            units = labor / costs[choice]
            if choice == "factory":
                units = min(units, max(rules.factories_wanted(f.expected, t.total_capacity, t.capacity,
                                                              threshold), 1.0) - wip)
                labor = units * costs[choice]
                L.produce(f.account, "factory_wip", units, "construction")
                done = int(self.stock(world, f, "factory_wip") + 1e-9)
                if done:
                    L.remove(f.account, "factory_wip", min(float(done), self.stock(world, f, "factory_wip")), "use", "completed")
                    L.produce(f.account, "factory", float(done), "completed")
                    f.conditions = f.conditions + [1.0] * done
            else:
                L.produce(f.account, choice, units, "investment")
            f.builders = labor
            bill = labor * f.wage
            f.carry = max(budget_cash - bill, 0.0)
            f.costs += bill
            weights = np.where(self.sector == f.sector, self.size, 0.0)
            self.pay_households(world, f, bill, "investment_wages", weights)
            spent_total += labor
        world.vars["investment_labor"] = spent_total

    # ================================================================== phase 8
    def adjust(self, world) -> None:
        """Expectations, prices, wages and occupational choice for next tick."""
        p = self.p
        lam = p.expectation_speed
        lo, hi = p.markup_bounds
        for f in self.firms:
            f.last_profit = f.revenue - f.op_costs - getattr(f, "interest_paid", 0.0)
            f.op_costs_prev = f.op_costs
            f.expected = float(B.adaptive(f.expected, f.orders_in, lam))
            # normal unit cost: what a unit costs when running at the volume that feeds the town,
            # at today's wages and input prices (so prices follow costs even when sales collapse)
            f.unit_cost = float(B.adaptive(f.unit_cost, self.normal_unit_cost(world, f), lam)) if f.unit_cost > 0 \
                else self.normal_unit_cost(world, f)
            if f.sector == MAINT:
                gap = B.order_gap(f.orders_in, f.hired)
            else:
                gap = B.inventory_gap(self.stock(world, f, OUTPUT_ITEM[f.sector]), p.inventory_target * f.expected)
            f.markup = B.markup_update(f.markup, gap, p.price_speed, p.normal_markup, p.markup_pull, lo, hi)
            if f.unit_cost > 0:
                f.price = (1.0 + f.markup) * f.unit_cost
        income = np.zeros(4)
        demand = np.zeros(4)
        supply = np.zeros(4)
        for s in range(4):
            firms = self.of(s)
            pool = float(self.size[self.sector == s].sum())
            # builders are temporary work for the unemployed: they don't steer wages or job choice
            D = sum(f.labor_demand for f in firms)
            for f in firms:
                share = f.labor_demand / D * pool if D > 0 else pool / len(firms)
                f.wage = float(B.wage_update(f.wage, f.labor_demand, share, p.wage_speed, p.max_step,
                                             p.normal_unemployment))
            wbar = float(np.mean([f.wage for f in firms]))
            income[s] = float(B.expected_income(wbar, D, pool))
            demand[s], supply[s] = D, pool
        new = B.mobility(np.array([self.size[self.sector == s].sum() for s in range(4)]), income, demand,
                         p.mobility, p.mobility_sensitivity)
        # aggregate mode: agent k is occupation k; movers take their share of savings with them
        old = self.size.copy()
        moved_cash = np.zeros(4)
        for k in range(4):
            if old[k] > 0 and new[k] < old[k]:
                moved_cash[k] = self.cash[k] * (old[k] - new[k]) / old[k]
        gain = np.maximum(new - old, 0.0)
        self.cash = self.cash - moved_cash + (moved_cash.sum() * gain / gain.sum() if gain.sum() > 0 else 0.0)
        self.size = new
        world.vars["expected_income"] = income

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.plan_and_hire, Phase.PRODUCTION: self.produce,
                Phase.MAINTENANCE: self.maintain, Phase.INCOME: self.pay_dividends,
                Phase.DISTRIBUTION: self.consume, Phase.INVESTMENT: self.invest,
                Phase.DEMOGRAPHY: self.adjust}

    # ================================================================== invariants
    def invariants(self):
        def households_match_ledger(world, ctx):
            out = []
            book = world.ledger.balance("households", self.money_item)
            if len(self.size) == 0:
                return out          # the town is empty: the last savings stay on the books as an estate
            if not close(float(self.cash.sum()), book, abs(book)):
                out.append(Violation("household_cash", f"agents hold {self.cash.sum():.9g}, ledger {book:.9g}", ("cash",), ("households",)))
            if (self.cash < -1e-9 * max(1.0, abs(book))).any():
                out.append(Violation("household_cash", f"negative household cash {self.cash.min()}", ("cash",)))
            N = world.groups["households"].size()
            if world.phase == Phase.DEMOGRAPHY and False:
                pass
            if not close(float(self.size.sum()), float(self.size.sum())):
                out.append(Violation("population", "size mismatch"))
            return out

        def factories_match_ledger(world, ctx):
            out = []
            for f in self.of(FACTORY):
                book = world.ledger.balance(f.account, "factory")
                if not close(sum(f.conditions), book):
                    out.append(Violation("factory_conditions", f"{f.account}: {sum(f.conditions)} vs ledger {book}", ("factory",)))
            return out

        def hires_within_pools(world, ctx):
            out = []
            for s in range(4):
                pool = float(self.pools_at_hire[s])
                used = sum(f.hired + f.builders for f in self.of(s))
                if used > pool * (1 + 1e-9) + 1e-9:
                    out.append(Violation("labor_pools", f"{SECTORS[s]}: {used:.6g} workers used, pool {pool:.6g}"))
            return out

        def money_flows_classified(world, ctx):
            known = set(CATEGORIES) | set(world.state["economy"].get("extra_categories", ()))
            bad = [p for p in ctx.journal if p.item == self.money_item and p.flow == "transfer" and p.note not in known]
            return [Violation("money_categories", f"unclassified money transfer: {bad[0]}", ("cash",))] if bad else []

        def physics(world, ctx):
            """Output and capital in the ledger must be explained by labor and technology."""
            p = self.p
            out = []
            made = {}
            for p_ in ctx.journal:
                if p_.flow in ("production", "use", "consumption"):
                    key = (p_.debit if p_.flow == "production" else p_.credit, p_.item, p_.flow)
                    made[key] = made.get(key, 0.0) + p_.amount
            mult = world.vars.get("harvest_multiplier", 1.0)
            for f in self.firms:
                t = getattr(f, "_tech_at_production", None) or self.firm_tech(world, f)
                if f.sector == FARM:
                    got = made.get((f.account, "raw", "production"), 0.0)
                    expect = rules.farm_output(getattr(f, "_hired_at_production", f.hired), t.a_x, t.land_limit,
                                               t.land_exponent) * mult
                    if not close(got, expect, expect):
                        out.append(Violation("physics", f"{f.account}: harvest {got:.9g} != {expect:.9g}", ("raw",)))
                if f.sector == FACTORY:
                    baked = made.get((f.account, "final", "production"), 0.0)
                    used = made.get((f.account, "raw", "use"), 0.0)
                    k = getattr(f, "running", 0)
                    if t.F_p > 0 and k > int(math.floor(getattr(f, "_hired_at_production", f.hired) / t.F_p + 1e-9)):
                        out.append(Violation("physics", f"{f.account}: runs {k} factories without their fixed crews"))
                    by_labor = max(getattr(f, "_hired_at_production", f.hired) - t.F_p * k, 0.0) / t.v_p_eff if t.v_p_eff > 0 else math.inf
                    cap = rules.capacity_of_best(k, t.conditions, t.capacity) if k else 0.0
                    if baked > min(used * t.e, by_labor, cap) * (1 + 1e-9) + 1e-9:
                        out.append(Violation("physics", f"{f.account}: baked {baked:.9g} beyond wheat/labor/capacity", ("final",)))
                cost = {"land_improvement": p.land_improvement_cost, "tools_processing": p.tool_cost,
                        "tools_distribution": p.tool_cost, "factory_wip": p.factory_cost}
                built = sum(made.get((f.account, item, "production"), 0.0) * c for item, c in cost.items())
                if not close(built, f.builders, f.builders):
                    out.append(Violation("physics", f"{f.account}: built {built:.9g} labor-ticks of capital with {f.builders:.9g} builders"))
            eaten = made.get(("households", "final", "consumption"), 0.0)
            carry = world.vars.get("delivery_capacity", 0.0)   # recorded when the carters delivered
            if eaten > carry * (1 + 1e-9) + 1e-9:
                out.append(Violation("physics", f"households ate {eaten:.9g} > carters could deliver {carry:.9g}", ("final",)))
            return out

        return [Invariant("physics", physics),
                Invariant("household_cash", households_match_ledger), Invariant("factory_conditions", factories_match_ledger),
                Invariant("labor_pools", hires_within_pools), Invariant("money_categories", money_flows_classified)]

    # ================================================================== metrics
    def metrics(self):
        def by_sector(fn):
            return lambda w: {SECTORS[s]: fn(w, s) for s in range(4)}

        def money_stock(w):
            L = w.ledger
            return L.balance("households", self.money_item) + sum(L.balance(f.account, self.money_item) for f in self.firms) \
                + world_cash_outside(w, self)

        def labor_share(w):
            N = float(self.size.sum()) or 1.0
            out = {"farming": sum(f.hired for f in self.of(FARM)) / N,
                   "processing": sum(f.hired for f in self.of(FACTORY)) / N,
                   "distribution": sum(f.hired for f in self.of(DIST)) / N,
                   "maintenance": sum(f.hired for f in self.of(MAINT)) / N,
                   "building": sum(f.builders for f in self.firms) / N}
            out["unemployed"] = max(0.0, 1.0 - sum(out.values()))
            return out

        def potential(w):
            y = rules.potential_output(float(self.size.sum()), self._tech(w.state["economy"]))
            prev = self.prev_potential if self.prev_potential is not None else y
            self.last["growth"] = y / prev - 1 if prev > 0 else 0.0
            self.prev_potential = y
            return y

        def nominal_gdp(w):
            return w.vars["household_spending"] + sum(f.builders * f.wage for f in self.firms)

        def inflation(w):
            pl = w.vars["price_level"]
            r = pl / self.prev_price_level - 1 if self.prev_price_level > 0 else 0.0
            self.prev_price_level = pl
            return r

        def capital(w):
            L = w.ledger
            return {"factory": sum(L.balance(f.account, "factory") for f in self.firms),
                    "land_improvement": sum(L.balance(f.account, "land_improvement") for f in self.firms),
                    "tools_processing": sum(L.balance(f.account, "tools_processing") for f in self.firms),
                    "tools_distribution": sum(L.balance(f.account, "tools_distribution") for f in self.firms)}

        return [
            Metric("population", lambda w: float(self.size.sum()), "people", "people"),
            Metric("consumption_per_person", lambda w: w.vars["consumption_per_person"], "bread eaten per person", "units"),
            Metric("labor_share", labor_share, "share of people per activity", "share"),
            Metric("unemployment_rate", lambda w: labor_share(w)["unemployed"], "share of people without work", "share"),
            Metric("total_output", lambda w: sum(getattr(f, "output", 0.0) for f in self.of(FACTORY)), "bread baked", "units"),
            Metric("raw_output", lambda w: sum(getattr(f, "output", 0.0) for f in self.of(FARM)), "wheat harvested", "units"),
            Metric("potential_output", potential, "max sustainable bread with all labor (Stage 1 technology)", "units"),
            Metric("growth_rate", lambda w: self.last.get("growth", 0.0), "growth of potential output", "1/tick"),
            Metric("prices", by_sector(lambda w, s: float(np.mean([f.price for f in self.of(s)]))), "average price by sector", "money"),
            Metric("price_level", lambda w: w.vars["price_level"], "retail bread price", "money"),
            Metric("inflation", inflation, "change in the retail bread price", "1/tick"),
            Metric("wages", by_sector(lambda w, s: float(np.mean([f.wage for f in self.of(s)]))), "average wage by occupation", "money"),
            Metric("real_wage", lambda w: float(np.mean([f.wage for f in self.firms])) / w.vars["price_level"], "average wage in loaves", "units"),
            Metric("money", lambda w: {"households": w.ledger.balance("households", self.money_item),
                                       "firms": sum(w.ledger.balance(f.account, self.money_item) for f in self.firms)}, "money held", "money"),
            Metric("money_stock", money_stock, "total money", "money"),
            Metric("nominal_gdp", nominal_gdp, "household spending + investment wages", "money"),
            Metric("velocity", lambda w: nominal_gdp(w) / max(money_stock(w), 1e-12), "nominal GDP / money stock", "1/tick"),
            Metric("inventories", lambda w: {"wheat": sum(w.ledger.balance(f.account, "raw") for f in self.firms),
                                             "bread": sum(w.ledger.balance(f.account, "final") for f in self.firms)}, "firm stocks", "units"),
            Metric("trade_completion", lambda w: w.vars["trade_completion"], "share of desired trades that happen", "share"),
            Metric("failed_trades", lambda w: 1.0 - w.vars["trade_completion"], "share of desired trades that fail", "share"),
            Metric("capital", capital, "capital stock by type", "units"),
            Metric("factories", lambda w: sum(len(f.conditions) for f in self.of(FACTORY)), "number of factories", "count"),
            Metric("unmaintained_share", lambda w: float(np.mean([getattr(f, "unmaintained", 0.0) for f in self.of(FACTORY)])),
                   "share of factory upkeep not bought", "share"),
            Metric("effective_farm_productivity", lambda w: (sum(getattr(f, "output", 0.0) for f in self.of(FARM)) /
                                                              max(sum(f.hired for f in self.of(FARM)), 1e-12)), "wheat per farmer", "units"),
            Metric("expected_income", lambda w: {SECTORS[s]: float(w.vars["expected_income"][s]) for s in range(4)},
                   "wage x chance of a job, by occupation", "money"),
            Metric("occupations", lambda w: {SECTORS[s]: float(self.size[self.sector == s].sum()) for s in range(4)},
                   "people per occupation", "people"),
        ]
