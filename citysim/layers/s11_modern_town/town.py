"""Stage 11: a modern town before the internet — one clean, compact, stock-flow consistent economy.

One tick is a quarter (tick 1 = 1980 Q1). Ten thousand people; every dollar and every unit of
output goes through the double-entry ledger, so the engine's conservation checks cover it all.

Organs (supply chains and institutions):
* Food chain: farms grow food -> shops (retail) sell it to households.
* Goods chain: factories make consumer goods -> shops -> households; factories also make
  machines that every business installs as capital.
* Services: care, education, leisure, advertising — labor-only, bought directly.
* Telecom and hardware: phone service for households, and IT equipment for businesses.
* Finance: the bank's and insurer's fee services.
* State: city hall employs public workers and funds a research university; taxes on wages,
  profits and realized capital gains; unemployment benefits; bonds when in deficit.
* Bank: deposits are the town's money; it lends to businesses and holds city bonds.
* Central bank: sets the policy rate with a Taylor rule; reacts to market stress (see Params).
* Insurer: collects premiums, pays claims, invests its reserves in shares.
* Stock market: every business sector is a listed company; prices are valuations of expected
  earnings, discounted at the policy rate plus a risk premium, with growth expectations that
  extrapolate recent growth and swing with market sentiment.
* Households: workers (90% of people) and owners (10%: managers, professionals, investors),
  who own most of the shares. Spending follows income and wealth (shares at market value).

Every business: hires toward the labor its expected sales need, sells what customers order up
to its capacity (labor x productivity, and installed capital), prices at a markup over unit
cost, invests to keep enough capital, borrows when its cash runs short, pays taxes and
dividends. Wages follow a Phillips curve. Productivity grows with the university's research.

Other layers extend the town through hooks in `world.state["economy"]`: extra spending
categories (`consumer_markets`), extra employers (`extra_employment`), productivity
multipliers (`productivity_boost`), extra listed equity (`equity_markets`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field

from ...core.invariants import Invariant, Violation
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

MARKET = ["farm", "factory", "retail", "services", "telecom", "finance"]
ITEM = {"farm": "food", "factory": "goods", "services": "services", "telecom": "telecom", "finance": "finance"}
HH = ["hh:workers", "hh:owners"]
BANK, CB, GOV, INS = "bank", "cb", "gov", "insurer"
ROW = "row"                     # the rest of the world: sells the town what its factories cannot make in time
MONEY = "deposits"


START_YEAR = [1980.0]


def year_of(tick: int) -> float:
    return START_YEAR[0] + (tick - 1) / 4


@dataclass
class Sector:
    name: str
    A: float                      # output per worker per quarter
    markup: float
    premium: float = 1.0          # wage relative to the town's base wage
    L: float = 0.0                # workers
    price: float = 1.0
    exp_demand: float = 0.0       # expected units sold next quarter
    sold: float = 0.0
    demand: float = 0.0
    revenue: float = 0.0
    costs: float = 0.0
    interest: float = 0.0
    profit: float = 0.0
    earnings: float = 0.0         # smoothed after-tax profit
    growth_exp: float = 0.0       # expected revenue growth per quarter (valuation)
    revenue_prev: float = 0.0
    capex: float = 0.0
    capacity: float = 0.0
    shares: float = 1000.0
    share_price: float = 1.0
    hist: dict = field(default_factory=dict)


class TownParams(LayerParams):
    people: float = Field(10000.0, gt=0)
    owners_share: float = Field(0.1, gt=0, lt=1, description="share of people who are owners")
    owners_wage_share: float = Field(0.25, ge=0, le=1, description="owners' share of the wage bill (managers, professionals)")
    base_wage: float = Field(10.0, gt=0, description="dollars (thousands) per worker per quarter at the start")
    budget: dict[str, float] = Field(default_factory=lambda: {"food": 0.11, "goods": 0.28, "services": 0.48,
                                                              "telecom": 0.045, "finance": 0.085})
    mpc_income: float = Field(0.89, ge=0, le=1)
    mpc_wealth: float = Field(0.1, ge=0, description="spending out of deposits per quarter (target ~ (1-mpc)/this)")
    mpc_equity: float = Field(0.008, ge=0, description="spending out of share wealth per quarter (the wealth effect)")
    rate_sensitivity: float = Field(0.5, ge=0, description="spending falls by this x (real rate - neutral) [annual]")
    markups: dict[str, float] = Field(default_factory=lambda: {"farm": 0.30, "factory": 0.35, "retail": 0.25, "services": 0.28,
                                                               "telecom": 0.45, "finance": 0.45})
    premiums: dict[str, float] = Field(default_factory=lambda: {"farm": 0.8, "factory": 1.0, "retail": 0.85, "services": 0.95,
                                                                "telecom": 1.3, "finance": 1.4})
    productivity: dict[str, float] = Field(default_factory=lambda: {"farm": 7.0, "factory": 1.4, "retail": 2.6, "services": 1.0,
                                                                    "telecom": 1.4, "finance": 1.0})
    public_share: float = Field(0.14, ge=0, lt=1, description="people employed by city hall")
    research_share: float = Field(0.015, ge=0, lt=1, description="people working at the university")
    research_growth: float = Field(0.25, ge=0, description="trend productivity growth per quarter per unit research share")
    capital_output: float = Field(4.0, gt=0, description="capital worth this many quarters of revenue")
    depreciation: float = Field(0.02, ge=0, le=1, description="per quarter")
    machine_cost: float = Field(1.0, gt=0, description="factory output units per machine")
    imports: bool = Field(True, description="when factories are sold out, buy machines and goods from abroad")
    export_rate: float = Field(0.08, ge=0, le=1, description="share of the town's money held abroad spent on its exports each quarter")
    markup_speed: float = Field(0.02, ge=0, description="markup rise per quarter per unit of excess demand")
    catch_up: float = Field(0.2, ge=0, description="extra investment speed per unit of capital shortage (turning customers away)")
    invest_speed: float = Field(0.03, ge=0, le=1, description="share of the capital gap closed per quarter")
    income_speed: float = Field(0.3, gt=0, le=1, description="weight of this quarter's income in spending plans")
    hire_speed: float = Field(0.5, gt=0, le=1)
    fire_speed: float = Field(0.25, gt=0, le=1)
    expectation_speed: float = Field(0.4, gt=0, le=1)
    phillips: float = Field(0.05, ge=0, description="wage growth per quarter per unit of (u* - u)")
    debt_target: float = Field(1.4, ge=0, description="fiscal rule: city bonds in quarters of GDP")
    tax_adjust: float = Field(0.002, ge=0, description="fiscal rule: income tax change per quarter per unit of debt gap")
    natural_unemployment: float = Field(0.055, ge=0, lt=1)
    frictional_unemployment: float = Field(0.035, ge=0, lt=1, description="people always between jobs (nobody can hire them)")
    start_year: float = Field(1980.0, description="calendar year of tick 1")
    income_tax: float = Field(0.22, ge=0, lt=1)
    profit_tax: float = Field(0.30, ge=0, lt=1)
    capital_gains_tax: float = Field(0.20, ge=0, lt=1)
    realization: float = Field(0.04, ge=0, le=1, description="share of households' unrealized gains realized per quarter")
    benefit: float = Field(0.45, ge=0, description="unemployment benefit, share of the base wage")
    payout: float = Field(0.6, ge=0, le=1, description="share of after-tax profit paid as dividends")
    cash_buffer: float = Field(1.0, ge=0, description="quarters of costs a business keeps; more is paid out")
    loan_spread: float = Field(0.025, ge=0, description="annual loan rate above the policy rate")
    deposit_spread: float = Field(0.01, ge=0, description="annual deposit rate below the policy rate")
    bond_spread: float = Field(0.005, ge=0, description="annual city bond rate above the policy rate")
    amortization: float = Field(0.04, ge=0, le=1, description="share of loans repaid per quarter")
    bank_capital: float = Field(0.10, ge=0)
    capital_requirement: float = Field(0.08, ge=0)
    premium_rate: float = Field(0.02, ge=0, description="insurance premium, share of household income")
    claims_ratio: float = Field(0.75, ge=0, description="expected claims / premiums")
    insurer_equity_share: float = Field(0.4, ge=0, le=1, description="share of the insurer's assets held in shares")
    insurer_reserve: float = Field(6.0, ge=0, description="cash reserve, quarters of premiums; the rest is paid out")
    equity_premium: float = Field(0.05, ge=0, description="annual return over the policy rate shareholders want")
    growth_memory: float = Field(0.85, ge=0, lt=1, description="weight of past expectations in expected growth")
    sentiment_momentum: float = Field(0.15, ge=0, description="sentiment rise per unit of the market's quarterly return")
    sentiment_decay: float = Field(0.12, ge=0, le=1, description="share of sentiment that fades per quarter")
    sentiment_growth: float = Field(0.005, ge=0, description="extra expected growth per quarter per unit of sentiment")
    min_yield: float = Field(0.03, gt=0, description="smallest earnings yield over growth (P/E ceiling ~ 1/this)")
    extrapolation: float = Field(0.15, ge=0, le=1, description="how much recent growth sways expected long-run growth")
    pe_floor: float = Field(6.0, gt=0, description="lowest price/earnings ratio")
    # central bank
    neutral_rate: float = Field(0.025, description="annual real neutral rate")
    inflation_target: float = Field(0.02)
    taylor_inflation: float = Field(0.5, ge=0)
    taylor_gap: float = Field(0.5, ge=0)
    okun: float = Field(2.0, ge=0, description="output gap per point of unemployment above normal")
    rate_floor: float = Field(0.0025)
    rate_smoothing: float = Field(0.8, ge=0, lt=1)
    long_rate_speed: float = Field(0.05, gt=0, le=1, description="weight of today's real rate in the long rate")
    crunch: float = Field(0.5, ge=0, description="loan spread added per unit of bank capital shortfall")
    stress_cut: float = Field(0.0075, ge=0, description="extra cut when shares fall sharply (the Fed's 1998/2001 response)")
    stress_threshold: float = Field(0.15, ge=0, description="fall from the recent peak that counts as market stress")
    policy: str = Field("taylor", pattern="^(taylor|lean|fixed)$", description="lean: also raise rates as shares get dear")
    lean_strength: float = Field(0.02, ge=0, description="lean: annual rate added per unit of log overvaluation")
    initial_rate: float = Field(0.06)
    seed_shift: int = 0


class ModernTown(Layer):
    name = "modern_town"
    provides = ["economy", "town"]
    Params = TownParams
    p: TownParams

    # ================================================================== setup
    def setup(self, world) -> None:
        p, L = self.p, world.ledger
        self.world = world
        START_YEAR[0] = p.start_year
        self.rng = np.random.default_rng([world.seed, 1101 + p.seed_shift])
        for item in ("food", "goods", "services", "telecom", "finance", "machines", "it_equipment", "public_services"):
            L.register_item(item, "good")
        L.register_item("capital", "capital")
        L.register_item(MONEY, "claim")
        L.register_item("loans", "claim")
        L.register_item("bonds", "claim")
        L.register_item("reserves", "money")
        for acc in [*HH, BANK, CB, GOV, INS, ROW] + [f"firm:{s}" for s in MARKET]:
            L.open_account(acc)
        self.sec = {s: Sector(s, p.productivity[s], p.markups[s], p.premiums[s]) for s in MARKET}
        for s in MARKET:
            L.register_item(f"shares:{s}", "claim")
        st = world.state.setdefault("economy", {})
        st.update(town=self, consumer_markets=[], extra_employment=[], productivity_boost={}, equity_markets=[],
                  extra_wealth=[], extra_categories=[], hooks_capex=[])
        self.N = p.people
        self.w = p.base_wage
        self.rate = p.initial_rate                      # annual policy rate
        self.cpi = 1.0
        self.cpi_prev = 1.0
        self.infl = p.inflation_target                  # annual, smoothed
        self.trend = 1.0                                # research-driven productivity level
        self.public = p.public_share * self.N
        self.research = p.research_share * self.N
        self.sentiment = 0.0
        self.premium_shock = 0.0        # extra risk premium (scripted episodes)
        self.spread_shock = 0.0         # extra loan spread (scripted episodes; negative: liquidity)
        self.forced_cut = 0.0
        self.cut_now = 0.0
        self.stressed = False
        self.rate_cap = None            # a promise to hold rates low (scripted episodes)
        self.extra_spread = {}          # annual risk premium on particular borrowers (e.g. junk-rated carriers)
        self.u_hist = []
        self.long_real = p.neutral_rate                 # long-term real interest rate (slow average)
        self.tax_rate = p.income_tax
        self.peak_index = None
        self.stress = 0.0
        self.t = self.zero()
        self.income_prev = {h: 0.0 for h in HH}
        self.history = []
        # ---- initial state: solve for the labor each sector needs at starting prices, so the town
        # starts close to its steady state (consumption scaled so employment is at its normal level)
        for sec in self.sec.values():
            sec.price = (1 + sec.markup) * self.w * sec.premium / sec.A
        r, f, fac = self.sec["retail"], self.sec["farm"], self.sec["factory"]
        rp = {"food": (1 + r.markup) * (f.price + r.premium * self.w / r.A),
              "goods": (1 + r.markup) * (fac.price + r.premium * self.w / r.A)}
        target_L = self.N * (1 - p.natural_unemployment) - self.public - self.research

        def units_for(C):
            u = {}
            u["farm"] = p.budget["food"] * C / rp["food"]
            u["factory"] = p.budget["goods"] * C / rp["goods"]
            u["retail"] = u["farm"] + u["factory"]
            for s_ in ("services", "telecom", "finance"):
                u[s_] = p.budget[ITEM[s_]] * C / self.sec[s_].price
            # machines to replace wear: capital is `capital_output` quarters of revenue
            base = dict(u)
            for _ in range(20):                         # machines for everyone, including the factories
                rev = {s_: u[s_] * (self.sec[s_].price if s_ != "retail" else 0.35 * rp["goods"]) for s_ in u}
                K = {s_: p.capital_output * rev[s_] / fac.price for s_ in u}
                u["factory"] = base["factory"] + p.depreciation * sum(K.values()) * (1 - 0.08)
                u["telecom"] = base["telecom"] + p.depreciation * sum(K.values()) * 0.08 * fac.price / self.sec["telecom"].price
            return u, K
        C = 1.0
        u, K = units_for(C)
        C = target_L / sum(u[s_] / self.sec[s_].A for s_ in u)
        u, K = units_for(C)
        self.K0 = K
        for s_, sec in self.sec.items():
            sec.exp_demand = u[s_]
            sec.L = u[s_] / sec.A
            sec.demand = u[s_]
        L.begin_tick(0)
        bonds0 = C * 2.0                                # city debt: half a year of consumption
        for s, sec in self.sec.items():
            K = self.K0[s]
            L.produce(f"firm:{s}", "capital", K, "initial endowment")
            loan = 0.4 * K * fac.price
            cash = 1.0 * sec.L * self.w * sec.premium
            self._loan(f"firm:{s}", loan + cash, "initial loan")
        L.post(BANK, GOV, "bonds", bonds0, "transfer", "initial bonds")
        L.post(GOV, BANK, MONEY, bonds0, "transfer", "initial bonds")
        hh0 = bonds0 * 0.9                              # the city's past deficits are households' savings
        self.pay(GOV, "hh:workers", hh0 * 0.4, "initial savings")
        self.pay(GOV, "hh:owners", hh0 * 0.6, "initial savings")
        L.post(BANK, "ext:creation", "reserves", p.bank_capital * (bonds0 + 0.4 * C), "creation", "bank capital")
        # shares: owners hold 75%, workers (pensions) 15%, the insurer 10%
        for s, sec in self.sec.items():
            for acc, w_ in (("hh:owners", 0.62), ("hh:workers", 0.28), (INS, 0.10)):
                L.post(acc, f"firm:{s}", f"shares:{s}", sec.shares * w_, "transfer", "listing")
        L.post(INS, "ext:creation", "reserves", 0.02 * C, "creation", "insurer capital")
        for s, sec in self.sec.items():
            sec.earnings = 0.08 * sec.L * self.w * sec.premium
            sec.share_price = self.valuation(sec)
        for h in HH:
            self.income_prev[h] = C / 0.9 * (0.72 if h == "hh:workers" else 0.28)

    @staticmethod
    def zero():
        return dict(gdp=0.0, consumption=0.0, investment=0.0, gov_wages=0.0, benefits=0.0, taxes=0.0, cg_tax=0.0,
                    defaults=0.0, new_loans=0.0, dividends=0.0, claims=0.0, premiums=0.0, interest_paid=0.0,
                    ads=0.0, it_investment=0.0, real_consumption=0.0, unmet=0.0, imports=0.0, exports=0.0)

    # ================================================================== money helpers
    def bal(self, acc, item=MONEY):
        return self.world.ledger.balance(acc, item)

    def pay(self, frm, to, amount, note, borrow=True) -> float:
        """Move money; a business that runs short borrows from the bank (a credit line)."""
        if amount <= 1e-12:
            return 0.0
        have = self.bal(frm)
        if have < amount and borrow and (frm.startswith("firm:") or frm == GOV):     # startups get no credit line
            need = amount - max(have, 0.0)
            if frm == GOV:
                self.world.ledger.post(BANK, GOV, "bonds", need, "transfer", "city bonds")
                self.world.ledger.post(GOV, BANK, MONEY, need, "transfer", "city bonds")
            else:
                self._loan(frm, need, "credit line")
            have = self.bal(frm)
        x = min(amount, max(have, 0.0))
        if x > 1e-12:
            self.world.ledger.post(to, frm, MONEY, x, "transfer", note)
        return x

    def _loan(self, acc, amount, note):
        if amount <= 1e-12:
            return
        L = self.world.ledger
        L.post(BANK, acc, "loans", amount, "transfer", note)
        L.post(acc, BANK, MONEY, amount, "transfer", note)
        self.t["new_loans"] += amount

    def debt(self, acc):
        return -self.bal(acc, "loans")

    def repay(self, acc, amount, note="repayment"):
        amount = min(amount, self.debt(acc), max(self.bal(acc), 0.0))
        if amount <= 1e-12:
            return 0.0
        L = self.world.ledger
        L.post(BANK, acc, MONEY, amount, "transfer", note)
        L.post(acc, BANK, "loans", amount, "transfer", note)
        return amount

    def sell(self, seller, buyer, item, units, price, note, consume=False, produce=True) -> float:
        """seller makes `units` (if produce) and sells to buyer; returns units actually paid for."""
        if units <= 1e-12:
            return 0.0
        L = self.world.ledger
        cost = units * price
        have = self.bal(buyer)
        if buyer.startswith("hh:") or buyer == INS:
            if have < cost:
                units = max(have, 0.0) / price
                cost = units * price
        if units <= 1e-12:
            return 0.0
        if produce:
            L.produce(seller, item, units, note)
        L.transfer(seller, buyer, item, units, note)
        self.pay(buyer, seller, cost, note)
        if consume:
            L.remove(buyer, item, units, "consumption", note)
        return units

    # ================================================================== measures
    def employed(self) -> float:
        extra = sum(fn() for fn in self.world.state["economy"]["extra_employment"])
        return sum(s.L for s in self.sec.values()) + self.public + self.research + extra

    def hireable(self) -> float:
        """People who can be hired this quarter: everyone out of work beyond those between jobs."""
        return max(self.N * (1 - self.p.frictional_unemployment) - self.employed(), 0.0)

    def unemployment(self) -> float:
        return max(0.0, 1.0 - self.employed() / self.N)

    def disc(self) -> float:
        """Quarterly discount rate for valuations."""
        return (self.rate + self.p.equity_premium) / 4

    def valuation(self, sec: Sector) -> float:
        """Price per share: next year's earnings over (discount rate - expected growth), the gap kept at
        least `min_yield` a year (a price/earnings ceiling of 1/min_yield in calm times)."""
        d = self.long_real + self.p.equity_premium + self.premium_shock    # real long-term discount rate
        trend = (1 + self.p.research_growth * self.research / self.N) ** 4 - 1
        g = trend + self.p.extrapolation * ((1 + sec.growth_exp) ** 4 - 1 - trend) \
            + self.sentiment * self.p.sentiment_growth * 4                           # real growth expected
        g = min(g, d - self.p.min_yield)
        e = max(sec.earnings, 0.02 * sec.revenue_prev, 1e-6) * 4
        pe = min(max((1 + g) / (d - g), self.p.pe_floor), 1 / self.p.min_yield)
        return e * pe / sec.shares

    def equity_value(self, acc) -> float:
        L = self.world.ledger
        v = sum(L.balance(acc, f"shares:{s}") * sec.share_price for s, sec in self.sec.items())
        for fn in self.world.state["economy"]["equity_markets"]:
            v += fn(acc)
        return v

    def wealth(self, acc) -> float:
        return max(self.bal(acc), 0.0) + self.equity_value(acc)

    def index(self) -> float:
        return sum(sec.share_price * sec.shares for sec in self.sec.values())

    # ================================================================== phase 1: policy, plans, hiring
    def plan(self, world) -> None:
        p = self.p
        self.t = self.zero()
        # central bank (on last quarter's data)
        u = self.unemployment()
        self.u_hist.append(u)
        rising = len(self.u_hist) > 4 and u - self.u_hist[-5] > 0.003
        gap = -p.okun * (u - p.natural_unemployment)
        target = p.neutral_rate + self.infl + p.taylor_inflation * (self.infl - p.inflation_target) + p.taylor_gap * gap
        if p.policy == "lean":
            target += p.lean_strength * max(self.overvaluation(), 0.0)
        if p.policy == "fixed":
            target = p.initial_rate
        new = p.rate_smoothing * self.rate + (1 - p.rate_smoothing) * target
        self.cut_now = 0.0
        if self.stress > p.stress_threshold and p.policy != "fixed" and (not self.stressed or rising):
            new -= p.stress_cut          # the central bank steadies falling markets (again each quarter jobs are being lost)
            self.cut_now += p.stress_cut
            self.stressed = True
        if self.stress < p.stress_threshold / 2:
            self.stressed = False
        if self.forced_cut and p.policy != "fixed":      # an emergency cut (scripted episodes)
            new -= self.forced_cut
            self.cut_now += self.forced_cut
        self.forced_cut = 0.0
        if self.rate_cap is not None and p.policy != "fixed":
            new = min(new, self.rate_cap)
        self.rate = max(new, p.rate_floor)
        real = self.rate - (0.7 * p.inflation_target + 0.3 * self.infl)
        self.long_real = (1 - p.long_rate_speed) * self.long_real + p.long_rate_speed * real
        # productivity: research trend x adoption boosts (other layers)
        self.trend *= 1 + p.research_growth * self.research / self.N
        boost = world.state["economy"]["productivity_boost"]
        for s, sec in self.sec.items():
            sec.A = p.productivity[s] * self.trend * boost.get(s, 1.0)
        # hiring towards expected sales
        avail = self.hireable()
        wants = {}
        for s, sec in self.sec.items():
            need = min(sec.exp_demand, self.capital_capacity(s)) / max(sec.A, 1e-9) * 1.03   # no staff without machines
            wants[s] = need - sec.L
        hires = sum(max(d, 0.0) for d in wants.values())
        fill = min(1.0, 0.8 * avail / hires) if hires > 1e-9 else 1.0
        for s, sec in self.sec.items():
            d = wants[s]
            sec.L = max(sec.L + (p.hire_speed * d * fill if d > 0 else p.fire_speed * d), 0.0)
            sec.capacity = sec.L * sec.A
            sec.capacity = min(sec.capacity, self.capital_capacity(s))
            sec.sold = sec.demand = sec.revenue = sec.costs = sec.interest = 0.0

    # ================================================================== phase 5: markets
    def markets(self, world) -> None:
        p, L = self.p, world.ledger
        st = world.state["economy"]
        real_rate = self.rate - self.infl
        mood = max(0.6, 1 - p.rate_sensitivity * (real_rate - p.neutral_rate))
        budget = dict(p.budget)
        extra = [m for m in st["consumer_markets"]]
        for m in extra:                                 # e.g. online services take a share of the budget
            for k, v in m.shift().items():
                budget[k] = budget.get(k, 0.0) + v
        r, f = self.sec["retail"], self.sec["farm"]
        fac = self.sec["factory"]
        retail_unit = r.premium * self.w / r.A
        price_food = (1 + r.markup) * (f.price + retail_unit)
        price_goods = (1 + r.markup) * (fac.price + retail_unit)
        self.retail_prices = {"food": price_food, "goods": price_goods}
        retail_cap = r.capacity
        # each household group spends out of income and wealth
        for h in HH:
            plan = mood * (p.mpc_income * self.income_prev[h] + p.mpc_wealth * max(self.bal(h), 0.0)
                           + p.mpc_equity * self.equity_value(h))
            plan = min(plan, max(self.bal(h), 0.0))
            spent = 0.0
            for k, share in budget.items():
                if share <= 0:
                    continue
                money = plan * share
                if k in ("food", "goods"):
                    src = f if k == "food" else fac
                    price = price_food if k == "food" else price_goods
                    want = money / price
                    src.demand += want
                    r.demand += want
                    units = min(want, retail_cap, max(src.capacity - src.sold, 0.0))
                    more = min(want, retail_cap) - units if k == "goods" else 0.0      # the shops import the rest
                    retail_cap -= units
                    got = 0.0
                    if units > 1e-12:
                        got = self.sell(f"firm:{src.name}", "firm:retail", ITEM[src.name], units, src.price, f"{k} wholesale")
                        src.sold += got
                        src.revenue += got * src.price
                        r.costs += got * src.price
                    if more > 1e-12 and p.imports:
                        imp = self.import_goods("firm:retail", "goods", more, src.price, "imported goods")
                        retail_cap -= imp
                        r.costs += imp * src.price
                        got += imp
                    if got > 1e-12:
                        got = self.sell("firm:retail", h, ITEM[src.name], got, price, f"{k} retail", consume=True, produce=False)
                        r.sold += got
                        r.revenue += got * price
                        spent += got * price
                        self.t["real_consumption"] += got * (1.0 if k == "food" else 1.0)
                elif k in ITEM.values():
                    sec = next(x for x in self.sec.values() if ITEM.get(x.name) == k)
                    want = money / sec.price
                    sec.demand += want
                    units = min(want, max(sec.capacity - sec.sold, 0.0))
                    got = self.sell(f"firm:{sec.name}", h, k, units, sec.price, k, consume=True)
                    sec.sold += got
                    sec.revenue += got * sec.price
                    spent += got * sec.price
                    self.t["real_consumption"] += got
                else:
                    for m in extra:
                        if k in m.categories:
                            got = m.buy(world, h, k, money)
                            spent += got
            self.t["consumption"] += spent
        # government services are produced by public employees and consumed by everyone
        L.produce(GOV, "public_services", self.public, "public services")
        L.remove(GOV, "public_services", self.public, "consumption", "public services")
        # investment: businesses keep enough capital for expected sales (machines from factories,
        # a share as IT equipment from telecom)
        for fn in st.get("hooks_build", []):                        # e.g. a network build-out that outbids everyone
            fn(world)
        for s in sorted(self.sec, key=lambda s: s != "factory"):    # machine makers equip themselves first
            sec, acc = self.sec[s], f"firm:{s}"
            K = self.bal(acc, "capital")
            need = self.capital_needed(s, sec.exp_demand) * 1.05
            short = min(max(need / max(K, 1e-9) - 1.1, 0.0), 0.5)       # turning customers away: invest faster
            buy = max(p.depreciation * K + (p.invest_speed + p.catch_up * short) * (need - K), 0.0)
            boom = st.get("it_boom", 0.0) * p.depreciation * K        # extra IT bought in a tech boom
            if boom < 0 and K > need:                                   # ...and the hangover while the surplus lasts
                buy = max(buy + boom, 0.0)
            self.invest(acc, buy, sec)
            if boom > 1e-12:
                self.invest(acc, boom, sec, it_share=1.0)
        for fn in st["hooks_capex"]:
            fn(world)
        # the rest of the world spends some of the town's money it holds on the town's goods (from spare capacity)
        units = min(p.export_rate * max(self.bal(ROW), 0.0) / fac.price, max(fac.capacity - fac.sold, 0.0))
        got = self.sell("firm:factory", ROW, "goods", units, fac.price, "exports")
        if got > 1e-12:
            L.remove(ROW, "goods", got, "export", "exports")
            fac.sold += got
            fac.revenue += got * fac.price
            self.t["exports"] += got * fac.price

    def import_goods(self, acc, item, units, price, note) -> float:
        """Buy from abroad: the money goes to the rest of the world (its deposits here), the goods arrive."""
        paid = self.pay(acc, ROW, units * price, note)
        got = paid / price
        if got > 1e-12:
            self.world.ledger.post(acc, "ext:import", item, got, "import", note)
        self.t["imports"] += paid
        return got

    def invest(self, acc, units, sec=None, it_share=None) -> float:
        """Buy capital: machines from factories; IT equipment from an IT maker if there is one, else telecom."""
        if units <= 1e-12:
            return 0.0
        fac, tel = self.sec["factory"], self.sec["telecom"]
        st = self.world.state["economy"]
        its = st.get("it_share", 0.08) if it_share is None else it_share
        got = 0.0
        vendor = st.get("it_vendor")                    # e.g. a computer/router maker (other layers)
        if vendor is not None and units * its > 1e-12:
            spent = vendor(acc, units * its * fac.price)
            got = spent / fac.price                     # capital is counted at factory prices
            if sec is not None:
                sec.capex += spent
            self.t["investment"] += spent
            self.t["it_investment"] += spent
        i_units = min(units * its - got, max(tel.capacity - tel.sold, 0.0))
        m_units = min(units - got - i_units, max(fac.capacity - fac.sold, 0.0))     # IT sold out: plain machines instead
        fac.demand += units * (1 - its)
        tel.demand += units * its - got
        L = self.world.ledger
        short = units - got - i_units - m_units
        if short > 1e-12 and self.p.imports:                         # factories sold out: machines from abroad
            imp = self.import_goods(acc, "machines", short, fac.price, f"imported machines for {acc}")
            L.remove(acc, "machines", imp, "use", "installed")
            L.produce(acc, "capital", imp, "installed")
            if sec is not None:
                sec.capex += imp * fac.price
            self.t["investment"] += imp * fac.price
            got += imp
        for src, item, n in ((fac, "machines", m_units), (tel, "it_equipment", i_units)):
            if n <= 1e-12:
                continue
            g = self.sell(f"firm:{src.name}", acc, item, n, src.price, f"{item} for {acc}")
            src.sold += g
            src.revenue += g * src.price
            L.remove(acc, item, g, "use", "installed")
            L.produce(acc, "capital", g, "installed")
            if sec is not None:
                sec.capex += g * src.price
            self.t["investment"] += g * src.price
            if item == "it_equipment":
                self.t["it_investment"] += g * src.price
            got += g
        return got

    # ================================================================== phase 4: incomes
    def incomes(self, world) -> None:
        p = self.p
        income = {h: 0.0 for h in HH}
        ow = p.owners_wage_share

        def wages(frm, amount, note):
            paid = self.pay(frm, "hh:workers", amount * (1 - ow), note)
            paid += self.pay(frm, "hh:owners", amount * ow, note)
            income["hh:workers"] += amount * (1 - ow)
            income["hh:owners"] += amount * ow
            return paid
        for s, sec in self.sec.items():
            bill = sec.L * self.w * sec.premium
            wages(f"firm:{s}", bill, "wages")
            sec.costs += bill
        gw = (self.public + self.research) * self.w
        wages(GOV, gw, "public wages")
        self.t["gov_wages"] = gw
        for fn in world.state["economy"].get("payrolls", []):
            for h, x in fn(world).items():
                income[h] += x
        unemployed = max(self.N - self.employed(), 0.0)
        ben = unemployed * p.benefit * self.w
        self.pay(GOV, "hh:workers", ben, "benefits")
        income["hh:workers"] += ben
        self.t["benefits"] = ben
        # interest
        rd, rl, rb = max(self.rate - p.deposit_spread, 0.0) / 4, (self.rate + self.loan_spread()) / 4, (self.rate + p.bond_spread) / 4
        for acc in self.borrowers():
            due = (rl + self.extra_spread.get(acc, 0.0) / 4) * self.debt(acc)
            x = self.pay(acc, BANK, due, "loan interest")
            self.t["interest_paid"] += x
            s = acc.split(":", 1)[1]
            if s in self.sec:
                self.sec[s].interest += x
        self.pay(GOV, BANK, rb * max(self.bal(BANK, "bonds"), 0.0), "bond interest")
        for acc in [*HH, INS]:
            x = rd * max(self.bal(acc), 0.0)               # the bank credits interest: new deposits
            if x > 1e-12:
                self.world.ledger.post(acc, BANK, MONEY, x, "transfer", "deposit interest")
            if acc in income:
                income[acc] += x
        # insurance: premiums in, random claims out (storms, fires)
        for h in HH:
            prem = p.premium_rate * self.income_prev[h]
            x = self.pay(h, INS, prem, "premiums", borrow=False)
            self.t["premiums"] += x
        expected = p.claims_ratio * self.t["premiums"]
        claims = expected * float(self.rng.lognormal(-0.18, 0.6))
        claims = min(claims, max(self.bal(INS), 0.0))
        for h, w_ in (("hh:workers", 0.8), ("hh:owners", 0.2)):
            self.pay(INS, h, claims * w_, "claims", borrow=False)
            income[h] += claims * w_
        self.t["claims"] = claims
        self.income_now = income

    def unit_value(self, s) -> float:
        """Revenue per unit of output (retail: its margin per unit handled)."""
        if s == "retail":
            return 0.35 * self.retail_prices["goods"] if hasattr(self, "retail_prices") else 1.0
        return self.sec[s].price

    def capital_needed(self, s, units) -> float:
        return self.p.capital_output * units * self.unit_value(s) / self.sec["factory"].price

    def capital_capacity(self, s) -> float:
        K = self.bal(f"firm:{s}", "capital")
        per = self.capital_needed(s, 1.0)
        return K / per * 1.15 if per > 1e-12 else math.inf

    def loan_spread(self) -> float:
        """Annual spread over the policy rate: dearer when the bank is short of capital (credit crunch)."""
        p = self.p
        assets = max(self.bal(BANK, "loans") + self.bal(BANK, "bonds"), 1e-9)
        short = max(0.0, 1.5 * p.capital_requirement - self.bank_equity() / assets)
        return max(p.loan_spread + p.crunch * short + self.spread_shock, 0.0)

    def borrowers(self):
        return [a for a in self.world.ledger.accounts if (a.startswith("firm:") or a.startswith("net:")) and self.debt(a) > 1e-12]

    # ================================================================== phase 7: profits, taxes, dividends, finance
    def finance(self, world) -> None:
        p, L = self.p, world.ledger
        income = self.income_now
        # businesses
        for s, sec in self.sec.items():
            acc = f"firm:{s}"
            K = self.bal(acc, "capital")
            dep = p.depreciation * K
            L.remove(acc, "capital", dep, "depreciation", "wear")
            sec.profit = sec.revenue - sec.costs - sec.interest - dep * self.sec["factory"].price
            tax = p.profit_tax * max(sec.profit, 0.0)
            self.pay(acc, GOV, tax, "profit tax")
            self.t["taxes"] += tax
            after = sec.profit - tax
            sec.earnings = 0.9 * sec.earnings + 0.1 * after
            div = p.payout * max(after, 0.0)
            cash = self.bal(acc)
            buffer = p.cash_buffer * (sec.costs + sec.capex)
            div += 0.25 * max(cash - div - buffer - self.debt(acc), 0.0)
            div = min(div, max(cash, 0.0))
            self.dividends(acc, f"shares:{s}", div, income)
            # amortization and early repayment from spare cash
            self.repay(acc, p.amortization * self.debt(acc))
            self.repay(acc, max(self.bal(acc) - buffer, 0.0))
            sec.capex = 0.0
        # bank: profits to owners while well capitalised
        eq = self.bank_equity()
        assets = max(self.bal(BANK, "loans") + self.bal(BANK, "bonds"), 1e-9)
        if eq / assets > p.capital_requirement * 1.5:
            pay = 0.5 * (eq - p.capital_requirement * 1.5 * assets)
            pay = min(pay, 0.02 * assets)
            if pay > 0:
                L.post("hh:owners", BANK, MONEY, pay, "transfer", "bank dividends")
                income["hh:owners"] += pay
        # insurer: invests reserves in shares (buys from owners) and pays out what it doesn't need
        self.insurer_invest()
        reserve = p.insurer_reserve * max(self.t["premiums"], 1e-9)
        surplus = self.bal(INS) - reserve
        if surplus > 1e-9:
            x = self.pay(INS, "hh:owners", 0.5 * surplus, "insurer dividends", borrow=False)
            income["hh:owners"] += x
        # household income tax, capital gains tax
        gdp = self.N * self.w * 1.25                     # trend nominal income (not this quarter's GDP)
        gap = self.bal(BANK, "bonds") / gdp - p.debt_target
        self.tax_rate = float(np.clip(self.tax_rate + p.tax_adjust * gap / max(p.debt_target, 1e-9), 0.05, 0.45))
        for h in HH:
            tax = self.tax_rate * income[h]
            x = self.pay(h, GOV, tax, "income tax", borrow=False)
            self.t["taxes"] += x
            income[h] -= x
        for fn in world.state["economy"].get("finance_hooks", []):
            fn(world, income)
        # stock market: expectations, sentiment and prices
        idx0 = self.index() + sum(m.index() for m in world.state["economy"].get("indices", []))
        for s, sec in self.sec.items():
            g = sec.revenue / sec.revenue_prev / (self.cpi / self.cpi_prev) - 1 if sec.revenue_prev > 1e-9 else 0.0
            g = float(np.clip(g, -0.1, 0.1))
            sec.growth_exp = p.growth_memory * sec.growth_exp + (1 - p.growth_memory) * g
            sec.revenue_prev = sec.revenue
        for s, sec in self.sec.items():
            sec.share_price = self.valuation(sec)
        idx1 = self.index() + sum(m.index() for m in world.state["economy"].get("indices", []))
        ret = idx1 / idx0 - 1 if idx0 > 1e-9 else 0.0
        self.sentiment = (1 - p.sentiment_decay) * self.sentiment + p.sentiment_momentum * ret
        self.peak_index = idx1 if self.peak_index is None else max(self.peak_index * 0.99, idx1)
        self.stress = 1 - idx1 / self.peak_index if self.peak_index else 0.0
        gains = max(idx1 - idx0, 0.0)
        cg = p.capital_gains_tax * p.realization * gains * 0.85
        x = self.pay("hh:owners", GOV, cg, "capital gains tax", borrow=False)
        self.t["cg_tax"] = x
        self.t["taxes"] += x
        income["hh:owners"] -= x
        # city: surpluses repay bonds; deficits were financed by bonds as they arose
        spare = self.bal(GOV) - 0.5 * (self.t["gov_wages"] + self.t["benefits"])
        if spare > 1e-9:
            r = min(spare, self.bal(BANK, "bonds"))
            if r > 0:
                L.post(BANK, GOV, MONEY, r, "transfer", "bond repayment")
                L.post(GOV, BANK, "bonds", r, "transfer", "bond repayment")
        for h in HH:
            self.income_prev[h] = (1 - p.income_speed) * self.income_prev[h] + p.income_speed * max(income[h], 0.0)

    def dividends(self, issuer, item, amount, income):
        if amount <= 1e-12:
            return
        L = self.world.ledger
        holders = {a: L.balance(a, item) for a in [*HH, INS]}
        tot = sum(v for v in holders.values() if v > 0)
        for a, n in holders.items():
            if n > 0:
                x = amount * n / tot
                self.pay(issuer, a, x, "dividends", borrow=False)
                if a in income:
                    income[a] += x
        self.t["dividends"] += amount

    def insurer_invest(self):
        L = self.world.ledger
        cash = self.bal(INS)
        eq = self.equity_value(INS)
        target = self.p.insurer_equity_share * (cash + eq)
        buy = 0.05 * (target - eq)
        if abs(buy) < 1e-9:
            return
        # trade the index pro rata with owners at market prices
        for s, sec in self.sec.items():
            w_ = sec.share_price * sec.shares / max(self.index(), 1e-9)
            n = buy * w_ / max(sec.share_price, 1e-12)
            if n > 0:
                n = min(n, L.balance("hh:owners", f"shares:{s}") * 0.5)
                cost = n * sec.share_price
                if cost > cash:
                    continue
                L.post(INS, "hh:owners", f"shares:{s}", n, "transfer", "insurer buys")
                L.post("hh:owners", INS, MONEY, cost, "transfer", "insurer buys")
                cash -= cost
            elif n < 0:
                n = min(-n, L.balance(INS, f"shares:{s}"))
                cost = n * sec.share_price
                if cost > max(self.bal("hh:owners"), 0.0):
                    continue
                L.post("hh:owners", INS, f"shares:{s}", n, "transfer", "insurer sells")
                L.post(INS, "hh:owners", MONEY, cost, "transfer", "insurer sells")

    def bank_equity(self) -> float:
        L = self.world.ledger
        return L.balance(BANK, "loans") + L.balance(BANK, "bonds") + L.balance(BANK, MONEY) + L.balance(BANK, "reserves")

    def overvaluation(self) -> float:
        """log(market value / value at normal growth expectations)."""
        tot = norm = 0.0
        for sec in self.sec.values():
            tot += sec.share_price * sec.shares
            d = self.disc()
            norm += max(sec.earnings, 0.02 * sec.revenue_prev, 1e-6) * 1.0 / d
        for m in self.world.state["economy"].get("indices", []):
            tot += m.index()
            norm += m.fair_value()
        return math.log(max(tot, 1e-9) / max(norm, 1e-9))

    # ================================================================== phase 8: wages, prices, expectations
    def adjust(self, world) -> None:
        p = self.p
        lam = p.expectation_speed
        for s, sec in self.sec.items():
            sec.exp_demand = (1 - lam) * sec.exp_demand + lam * sec.demand
        u = self.unemployment()
        g_trend = p.research_growth * self.research / self.N
        expected = 0.7 * p.inflation_target + 0.3 * self.infl        # expectations mostly anchored
        gap = p.natural_unemployment - u
        tight = max(0.0, p.natural_unemployment - u) / max(p.natural_unemployment - p.frictional_unemployment, 1e-9)
        dlnw = expected / 4 + g_trend + p.phillips * gap * (1 + 2 * tight ** 2)      # convex: wages race when labor is scarce
        self.w *= math.exp(max(dlnw, -0.005))                         # wages are sticky downward
        for s, sec in self.sec.items():
            unit = self.w * sec.premium / sec.A
            short = sec.demand / max(sec.capacity, 1e-9) - 1
            sec.markup = float(np.clip(sec.markup + p.markup_speed * np.clip(short, -0.3, 0.3)
                                       - 0.15 * (sec.markup - p.markups[s]), 0.5 * p.markups[s], 2 * p.markups[s]))
            if s == "retail":
                sec.price = 1.0
            else:
                sec.price = (1 + sec.markup) * unit
        # consumer prices
        w_food, w_goods = p.budget["food"], p.budget["goods"]
        rp = self.retail_prices
        basket = (w_food * rp["food"] + w_goods * rp["goods"] + p.budget["services"] * self.sec["services"].price
                  + p.budget["telecom"] * self.sec["telecom"].price + p.budget["finance"] * self.sec["finance"].price)
        if not hasattr(self, "basket0"):
            self.basket0 = basket
        self.cpi_prev, self.cpi = self.cpi, basket / self.basket0
        q = self.cpi / self.cpi_prev - 1
        self.infl = 0.8 * self.infl + 0.2 * ((1 + q) ** 4 - 1)
        self.t["gdp"] = self.t["consumption"] + self.t["investment"] + self.t["gov_wages"] + self.t["exports"] - self.t["imports"]
        self.real_gdp = self.t["gdp"] / self.cpi

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.plan, Phase.DISTRIBUTION: self.markets, Phase.INCOME: self.incomes,
                Phase.FINANCE: self.finance, Phase.DEMOGRAPHY: self.adjust}

    # ================================================================== invariants
    def invariants(self):
        def labor(world, ctx):
            out = []
            if self.employed() > self.N * (1 + 1e-9):
                out.append(Violation("labor", f"{self.employed():.1f} employed > {self.N} people"))
            if any(s.L < -1e-9 for s in self.sec.values()):
                out.append(Violation("labor", "negative employment"))
            return out

        def solvent_households(world, ctx):
            return [Violation("money", f"{h} overdrawn") for h in [*HH, INS] if self.bal(h) < -1e-6]

        def capacity(world, ctx):
            out = []
            for s, sec in self.sec.items():
                if sec.sold > sec.capacity * (1 + 1e-9) + 1e-9:
                    out.append(Violation("physics", f"{s} sold {sec.sold:.3f} beyond capacity {sec.capacity:.3f}"))
            return out

        def claims(world, ctx):
            L = world.ledger
            out = []
            for item in ["loans", "bonds"]:
                if L.balance(BANK, item) < -1e-6:
                    out.append(Violation("claims", f"bank owes {item}"))
            return out

        return [Invariant("labor", labor), Invariant("households_solvent", solvent_households),
                Invariant("capacity", capacity), Invariant("claims", claims)]

    # ================================================================== metrics
    def metrics(self):
        def shares(w):
            out = {s: sec.L / self.N for s, sec in self.sec.items()}
            out["public"] = self.public / self.N
            out["research"] = self.research / self.N
            for fn in w.state["economy"].get("extra_employment_named", []):
                out.update({k: v / self.N for k, v in fn().items()})
            out["unemployed"] = self.unemployment()
            return out

        def top_wealth(w):
            a, b = self.wealth("hh:owners"), self.wealth("hh:workers")
            return a / (a + b) if a + b > 0 else 0.0

        return [
            Metric("year", lambda w: year_of(w.tick), "calendar year", "year"),
            Metric("gdp", lambda w: self.t["gdp"], "consumption + investment + public wages", "$"),
            Metric("real_gdp", lambda w: self.real_gdp, "GDP at constant prices", "$"),
            Metric("consumption", lambda w: self.t["consumption"], "household spending", "$"),
            Metric("investment", lambda w: self.t["investment"], "business investment", "$"),
            Metric("it_investment", lambda w: self.t["it_investment"], "IT equipment bought", "$"),
            Metric("imports", lambda w: self.t["imports"], "goods and machines bought from abroad", "$"),
            Metric("exports", lambda w: self.t["exports"], "goods sold abroad", "$"),
            Metric("foreign_deposits", lambda w: self.bal(ROW), "the town's money held by the rest of the world", "$"),
            Metric("unemployment", lambda w: self.unemployment(), "share of people without work", "share"),
            Metric("inflation", lambda w: self.infl, "smoothed annual consumer price inflation", "1/yr"),
            Metric("cpi", lambda w: self.cpi, "consumer prices (start = 1)", "x"),
            Metric("policy_rate", lambda w: self.rate, "central bank rate (annual)", "1/yr"),
            Metric("wage", lambda w: self.w, "base wage per quarter", "$"),
            Metric("real_wage", lambda w: self.w / self.cpi, "base wage in start-year prices", "$"),
            Metric("employment", shares, "share of people by employer", "share"),
            Metric("productivity", lambda w: {s: sec.A / self.p.productivity[s] for s, sec in self.sec.items()},
                   "output per worker relative to 1980", "x"),
            Metric("old_economy_index", lambda w: self.index(), "market value of the established businesses", "$"),
            Metric("sentiment", lambda w: self.sentiment, "market mood (momentum)", "x"),
            Metric("overvaluation", lambda w: self.overvaluation(), "log market value over value at normal growth", "log"),
            Metric("income_tax_rate", lambda w: self.tax_rate, "income tax rate (fiscal rule)", "share"),
            Metric("gov_balance", lambda w: self.t["taxes"] - self.t["gov_wages"] - self.t["benefits"], "city budget balance", "$"),
            Metric("public_debt", lambda w: self.bal(BANK, "bonds"), "city bonds", "$"),
            Metric("bank_loans", lambda w: self.bal(BANK, "loans"), "loans outstanding", "$"),
            Metric("loan_rate", lambda w: self.rate + self.loan_spread(), "rate businesses pay", "1/yr"),
            Metric("emergency_cut", lambda w: self.cut_now, "extra cut this quarter (market stress)", "1/yr"),
            Metric("bank_capital_ratio", lambda w: self.bank_equity() / max(self.bal(BANK, "loans") + self.bal(BANK, "bonds"), 1e-9),
                   "bank equity / assets", "ratio"),
            Metric("defaults", lambda w: self.t["defaults"], "loans written off", "$"),
            Metric("money", lambda w: sum(max(self.bal(a), 0.0) for a in w.ledger.accounts
                                          if not w.ledger.accounts[a].external and a != BANK), "deposits", "$"),
            Metric("owners_wealth_share", top_wealth, "owners' share of household wealth", "share"),
            Metric("household_wealth", lambda w: self.wealth("hh:owners") + self.wealth("hh:workers"), "deposits + shares", "$"),
            Metric("insurer_equity", lambda w: self.bal(INS) + self.equity_value(INS), "insurer assets", "$"),
            Metric("real_consumption", lambda w: self.t["real_consumption"], "units consumed", "units"),
        ]
