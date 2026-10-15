"""Stage 5: government and public goods.

The government ("gov" account) collects an income tax from households and a profit tax
from firms (before dividends), pays unemployment benefits (a share of the average wage)
and optionally a basic income, can pay for everyone's maintenance (public maintenance),
builds and maintains roads (a public good that cuts the labor needed to deliver bread for
every distributor at once), and borrows by selling bonds to the bank when spending exceeds
revenue (without a bank it must balance its budget). A share of public payroll can be
wasted (misallocated spending).

Spending priority when money is short: bond interest, benefits, public maintenance,
road upkeep, road building. Above the debt ceiling the government stops borrowing.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.invariants import Invariant, Violation, close
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

GOV = "gov"
CATS = ("income_tax", "profit_tax", "benefits", "basic_income", "public_wages", "public_maintenance", "rebate")


class GovernmentParams(LayerParams):
    income_tax: float = Field(0.1, ge=0, le=1, description="share of household income taxed")
    profit_tax: float = Field(0.2, ge=0, le=1, description="share of firm profit taxed before dividends")
    unemployment_benefit: float = Field(0.4, ge=0, description="benefit per unemployed person, as a share of the average wage")
    basic_income: float = Field(0.0, ge=0, description="paid to everyone, as a share of the average wage")
    public_maintenance: bool = Field(False, description="the government pays for all private capital upkeep")
    road_target: float = Field(0.0, ge=0, description="road units the government aims to build")
    road_cost: float = Field(300.0, gt=0, description="labor-ticks per road unit")
    road_elasticity: float = Field(0.25, ge=0, description="distribution labor x (1 + roads)^-elasticity")
    road_upkeep: float = Field(1.0, ge=0, description="workers per road unit to keep it from decaying")
    road_depreciation: float = Field(0.05, ge=0, le=1, description="share of unmaintained roads lost per tick")
    max_public_workers: float = Field(0.1, ge=0, le=1, description="most of the population the government employs")
    waste: float = Field(0.0, ge=0, le=1, description="share of public payroll that produces nothing")
    borrow: bool = Field(True, description="sell bonds to the bank to cover deficits (needs a bank)")
    bond_rate: float = Field(0.005, ge=0, description="interest per tick on bonds")
    debt_ceiling: float = Field(30.0, ge=0, description="most debt, in ticks of average tax revenue")
    cash_buffer: float = Field(2.0, ge=0, description="ticks of spending the government keeps; beyond that it "
                                                     "repays debt, then rebates the rest per head")
    repay_speed: float = Field(1.0, ge=0, le=1, description="share of spare money used to repay debt per tick "
                                                         "(repayment destroys money; Stage 10)")
    rebate_speed: float = Field(0.5, ge=0, le=1, description="share of excess money rebated per tick")


class Government(Layer):
    name = "government"
    requires = ["market"]
    provides = ["government"]
    Params = GovernmentParams
    p: GovernmentParams

    def setup(self, world) -> None:
        L = world.ledger
        st = world.state["economy"]
        self.m = st["market_layer"]
        L.open_account(GOV)
        L.register_item("roads", "capital")
        st["extra_categories"] = list(st.get("extra_categories", [])) + list(CATS)
        st.setdefault("money_holders", []).append(GOV)
        st["profit_tax"] = list(st.get("profit_tax", [])) + [self.tax_profit]
        if self.p.public_maintenance:
            st["maintenance_payer"] = self.pay_maintenance
        st["gov_layer"] = self
        self.revenue_avg = 0.0
        self.t = self.zero()

    @staticmethod
    def zero():
        return dict(income_tax=0.0, profit_tax=0.0, benefits=0.0, basic_income=0.0, public_wages=0.0,
                    public_maintenance=0.0, bond_interest=0.0, borrowed=0.0, repaid=0.0, public_workers=0.0, rebate=0.0,
                    road_labor=0.0, roads_built=0.0)

    # ------------------------------------------------------------------ money
    @property
    def mi(self) -> str:
        return self.m.money_item

    def money(self, world) -> float:
        return world.ledger.balance(GOV, self.mi)

    def debt(self, world) -> float:
        return -world.ledger.balance(GOV, "bonds")

    def fund(self, world, amount: float) -> float:
        """Make `amount` available, borrowing from the bank if allowed; returns what is available."""
        have = max(self.money(world), 0.0)
        if have >= amount:
            return amount
        buyer = world.state["economy"].get("bond_buyer")
        if self.p.borrow and buyer is not None:
            room = self.p.debt_ceiling * max(self.revenue_avg, 1e-9) - self.debt(world)
            want = min(amount - have, max(room, 0.0))
            got = buyer(world, GOV, want) if want > 1e-9 else 0.0
            self.t["borrowed"] += got
            have += got
        return min(amount, have)

    def avg_wage(self) -> float:
        return float(np.mean([f.wage for f in self.m.firms])) if self.m.firms else 1.0

    def unemployed_weights(self) -> np.ndarray:
        m = self.m
        if hasattr(m, "employer"):
            return np.where(m.employer < 0, m.size, 0.0)
        hired = np.array([sum(f.hired + f.builders for f in m.of(s)) for s in range(4)])
        return np.maximum(m.pools_at_hire - hired, 0.0)

    def pay_households(self, world, amount: float, weights: np.ndarray, category: str) -> None:
        m = self.m
        if amount <= 1e-12 or weights.sum() <= 0:
            return
        world.ledger.post("households", GOV, self.mi, amount, "transfer", category)
        w = weights / weights.sum()
        m.cash += amount * w
        m.income += amount * w

    # ------------------------------------------------------------------ hooks for the market
    def tax_profit(self, world, f, profit: float) -> float:
        tax = min(self.p.profit_tax * max(profit, 0.0), max(world.ledger.balance(f.account, self.mi), 0.0))
        if tax > 1e-12:
            world.ledger.post(GOV, f.account, self.mi, tax, "transfer", "profit_tax")
            self.t["profit_tax"] += tax
        return tax

    def pay_maintenance(self, world, cost: float) -> float:
        """Share of all private upkeep the government can fund this tick (0..1)."""
        if cost <= 1e-12:
            return 1.0
        avail = self.fund(world, cost)
        self.t["public_maintenance"] += avail
        return avail / cost

    # ------------------------------------------------------------------ phase 4: taxes and transfers
    def taxes_and_transfers(self, world) -> None:
        """Public works first (people hired for roads are not unemployed), then income tax, then benefits."""
        p, m = self.p, self.m
        public = self.public_works(world)
        tax = np.clip(p.income_tax * m.income, 0.0, np.maximum(m.cash, 0.0))
        total = float(tax.sum())
        if total > 1e-12:
            world.ledger.post(GOV, "households", self.mi, total, "transfer", "income_tax")
            m.cash -= tax
            m.income -= tax
            self.t["income_tax"] += total
        wage = self.avg_wage()
        u = np.maximum(self.unemployed_weights() - public, 0.0)
        want = p.unemployment_benefit * wage * float(u.sum())
        paid = self.fund(world, want) if want > 0 else 0.0
        self.pay_households(world, paid, u, "benefits")
        self.t["benefits"] += paid
        if p.basic_income > 0:
            want = p.basic_income * wage * float(m.size.sum())
            paid = self.fund(world, want)
            self.pay_households(world, paid, m.size.copy(), "basic_income")
            self.t["basic_income"] += paid

    # ------------------------------------------------------------------ phase 6: roads
    def public_works(self, world) -> np.ndarray:
        """Hire idle people to keep up and build roads; returns who was hired (people per agent)."""
        p, m, L = self.p, self.m, world.ledger
        st = world.state["economy"]
        roads = L.balance(GOV, "roads")
        need_upkeep = p.road_upkeep * roads
        need_build = max(p.road_target - roads, 0.0) * p.road_cost
        idle = self.idle_weights()
        cap = min(p.max_public_workers * float(m.size.sum()), float(idle.sum()))
        upkeep_w = min(need_upkeep, cap)
        build_w = min(need_build, cap - upkeep_w)
        wage = self.avg_wage()
        afford = self.fund(world, (upkeep_w + build_w) * wage)
        scale = afford / ((upkeep_w + build_w) * wage) if (upkeep_w + build_w) > 0 else 0.0
        upkeep_w *= scale
        build_w *= scale
        workers = upkeep_w + build_w
        hired = self.pick(idle, workers)
        if workers > 1e-12:                       # whole people: scale the jobs to who was actually hired
            k = float(hired.sum()) / workers
            upkeep_w, build_w, workers = upkeep_w * k, build_w * k, float(hired.sum())
        self.pay_households(world, workers * wage, hired, "public_wages")
        self.t["public_wages"] += workers * wage
        self.t["public_workers"] += workers
        useful = 1.0 - p.waste
        built = build_w * useful / p.road_cost
        if built > 1e-12:
            L.produce(GOV, "roads", built, "road building")
            self.t["roads_built"] += built
            self.t["road_labor"] += build_w
        u = 1.0 - (upkeep_w * useful / need_upkeep) if need_upkeep > 1e-12 else 0.0
        loss = p.road_depreciation * max(u, 0.0) * L.balance(GOV, "roads")
        L.remove(GOV, "roads", loss, "depreciation", "unmaintained roads")
        st["road_factor"] = (1.0 + L.balance(GOV, "roads")) ** (-p.road_elasticity)
        world.vars["public_workers"] = workers
        return hired

    def pick(self, idle: np.ndarray, workers: float) -> np.ndarray:
        """Choose who takes the public jobs: whole agents at random in agent mode, a share otherwise."""
        m = self.m
        if workers <= 1e-12 or idle.sum() <= 0:
            return np.zeros_like(idle)
        if not hasattr(m, "employer"):
            return idle * min(workers / idle.sum(), 1.0)
        idx = np.where(idle > 0)[0]
        m.rng.shuffle(idx)
        take = idx[np.cumsum(m.size[idx]) <= workers + 1e-9]
        out = np.zeros_like(idle)
        out[take] = m.size[take]
        return out

    def idle_weights(self) -> np.ndarray:
        m = self.m
        if hasattr(m, "employer"):
            return np.where((m.employer < 0) & (m.retrain == 0), m.size, 0.0)
        return self.unemployed_weights()

    # ------------------------------------------------------------------ phase 7: debt service
    def debt_service(self, world) -> None:
        p, L = self.p, world.ledger
        debt = self.debt(world)
        bank = world.state["economy"].get("bank_layer")
        if debt > 1e-12 and bank is not None:
            pr = world.state["economy"].get("policy_rate")      # Stage 6: bonds pay the policy rate
            interest = (p.bond_rate if pr is None else pr) * debt
            pay = min(interest, max(self.money(world), 0.0))
            if pay < interest:
                pay = self.fund(world, interest)
            if pay > 1e-12:
                L.post("bank:0", GOV, self.mi, pay, "transfer", "bond_interest")
                self.t["bond_interest"] += pay
            # surplus repays debt (keep a tick of spending in hand)
            spare = self.money(world) - self.p.cash_buffer * self.spending_last()
            if spare > 1e-9:
                r = min(p.repay_speed * spare, self.debt(world))
                L.post("bank:0", GOV, self.mi, r, "transfer", "bond_redemption")
                L.post(GOV, "bank:0", "bonds", r, "transfer", "bond_redemption")
                self.t["repaid"] += r
        rev = self.t["income_tax"] + self.t["profit_tax"]
        self.revenue_avg = 0.9 * self.revenue_avg + 0.1 * rev if self.revenue_avg > 0 else rev
        # fiscal rule: a government that hoards money drains the economy, so excess money goes back
        excess = self.money(world) - self.p.cash_buffer * max(self.spending_last(), 1e-9)
        if excess > 1e-9 and self.debt(world) <= 1e-9:
            r = self.p.rebate_speed * excess
            self.pay_households(world, r, self.m.size.copy(), "rebate")
            self.t["rebate"] = r

    def spending_last(self) -> float:
        t = self.t
        return t["benefits"] + t["basic_income"] + t["public_wages"] + t["public_maintenance"]

    def end_of_tick(self, world) -> None:
        world.vars["gov"] = self.t
        self.t = self.zero()

    def hooks(self):
        return {Phase.INCOME: self.taxes_and_transfers, Phase.FINANCE: self.debt_service,
                Phase.DEMOGRAPHY: self.end_of_tick}

    # ------------------------------------------------------------------ invariants and metrics
    def invariants(self):
        def solvent(world, ctx):
            v = []
            if world.ledger.balance(GOV, self.mi) < -1e-9:
                v.append(Violation("government", "the government spent money it did not have", (self.mi,), (GOV,)))
            if world.ledger.balance(GOV, "bonds") > 1e-9:
                v.append(Violation("government", "the government holds its own bonds as an asset", ("bonds",), (GOV,)))
            return v

        def roads_paid(world, ctx):
            t = world.vars.get("gov", {})
            built = sum(p_.amount for p_ in ctx.journal if p_.item == "roads" and p_.flow == "production")
            labor = t.get("road_labor", 0.0) * (1 - self.p.waste)
            if built * self.p.road_cost > labor * (1 + 1e-9) + 1e-9:
                return [Violation("physics", f"{built:.6g} road units need more labor than {labor:.6g}", ("roads",))]
            return []

        return [Invariant("government", solvent), Invariant("roads_paid", roads_paid)]

    def metrics(self):
        def g(k):
            return lambda w: w.vars.get("gov", {}).get(k, 0.0)

        def revenue(w):
            t = w.vars.get("gov", {})
            return t.get("income_tax", 0.0) + t.get("profit_tax", 0.0)

        def spending(w):
            t = w.vars.get("gov", {})
            return sum(t.get(k, 0.0) for k in ("benefits", "basic_income", "public_wages", "public_maintenance",
                                                "bond_interest", "rebate"))

        return [
            Metric("tax_revenue", revenue, "income + profit taxes", "money"),
            Metric("taxes", lambda w: {"income": g("income_tax")(w), "profit": g("profit_tax")(w)}, "taxes by type", "money"),
            Metric("public_spending", lambda w: {k: g(k)(w) for k in ("benefits", "basic_income", "public_wages",
                                                                      "public_maintenance", "bond_interest", "rebate")},
                   "public spending by type", "money"),
            Metric("deficit", lambda w: spending(w) - revenue(w), "spending - taxes", "money"),
            Metric("public_debt", lambda w: self.debt(w), "bonds outstanding", "money"),
            Metric("debt_to_revenue", lambda w: self.debt(w) / max(self.revenue_avg, 1e-9), "debt in ticks of tax revenue", "ticks"),
            Metric("gov_money", lambda w: self.money(w), "government's money", "money"),
            Metric("roads", lambda w: w.ledger.balance(GOV, "roads"), "road units", "units"),
            Metric("public_workers", g("public_workers"), "people employed by the government", "people"),
        ]
