"""Stage 8: trade, money and migration between cities.

Each city gets, per traded good, an importing house ("import:raw", "import:final"): a seller
in the local wheat or wholesale bread market, competing with local farms and bakeries
on price. After the cities' retail markets close (Phase 5), every importer restocks: it
estimates its sales, picks suppliers among the other cities by landed cost (the
supplier's price, converted, divided by 1 - transport cost), sends money to its agent
("port") in the supplier city, which buys from that city's producers like any customer.
The goods are shipped (an iceberg cost: a share `transport_cost` is lost on the way) and
arrive for the next tick. Unspent money is sent back.

Money between cities:
* currency union: one money. Payments move bank reserves: the payer's deposits are
  destroyed at its bank, which pays reserves out ("export" of cash); the receiving bank
  takes the reserves in ("import") and credits deposits. A city with a trade deficit loses
  reserves.
* floating: each city keeps its own money. A currency dealer holds money in every city;
  it buys one and sells the other at the exchange rate, and moves the rate towards
  whichever currency it is running short of.

Migration: each tick a share of people moves from the city with the lower expected real
income (average wage x employment rate / bread price) to a higher one, the unemployed
first, taking their money. Capital flight: households in a city with higher unemployment
than the best city move some of their money there (held by an account "savers:<home>"),
and bring it back once things calm down. Fiscal union (currency union only): a union budget
moves a share of the gap in money per head from richer towns' households to poorer ones.

Merchants only import what beats the local price (landed cost x (1 + import markup) below
the average local price), so towns without a cost advantage do not ship goods back and forth.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...core.metrics import Metric
from ...core.phases import Phase
from ..s2_money_prices import behavior as B
from ..s2_money_prices.market import Firm

FARM, FACTORY = 0, 1
SECTOR_OF = {"raw": FARM, "final": FACTORY}
BANK = "bank:0"
CATS = ("export", "fx", "trade_capital", "import_dividends", "migration", "capital_flight", "remit_out", "remit_in",
        "fiscal_transfer")


class Embargo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=1)
    duration: int = Field(1, ge=1)
    between: list[str]


class TradeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(True, description="false: the cities live in autarky")
    currency: Literal["union", "floating"] = "union"
    goods: list[Literal["raw", "final"]] = Field(default_factory=lambda: ["raw", "final"])
    transport_cost: float = Field(0.1, ge=0, lt=1, description="share of a shipment lost on the way")
    transport_costs: dict[str, float] = Field(default_factory=dict, description="per pair 'a-b' overrides")
    import_markup: float = Field(0.02, ge=0, description="importers' normal markup over landed cost")
    inventory_target: float = Field(1.0, ge=0, description="importers' stock target, ticks of expected sales")
    inventory_correction: float = Field(2.0, gt=0)
    expectation_speed: float = Field(0.3, gt=0, le=1)
    price_speed: float = Field(0.05, ge=0)
    importer_capital: float = Field(50.0, ge=0, description="money each importer starts with (from households)")
    payout: float = Field(0.5, ge=0, le=1, description="share of importers' excess money paid out per tick")
    capital_ticks: float = Field(2.0, ge=0, description="importers' working capital, ticks of expected purchases")
    capital_speed: float = Field(0.2, ge=0, le=1, description="share of a working-capital shortfall owners put in per tick")
    fx_capital: float = Field(0.1, ge=0, description="floating: dealer's money in each city, share of households' money")
    fx_float_use: float = Field(0.2, ge=0, le=1, description="floating: share of the dealer's float it may use "
                                                              "per tick to fill unmatched orders")
    fx_spread: float = Field(0.005, ge=0, lt=1, description="floating: share the dealer keeps on each conversion")
    fx_speed: float = Field(0.1, ge=0, description="floating: how fast the rate moves with excess demand for a currency")
    migration: float = Field(0.0, ge=0, description="share of the town that moves per unit of relative real-income gap")
    migration_cost: float = Field(0.1, ge=0, description="income gap (relative) people need before moving")
    capital_flight: float = Field(0.0, ge=0, description="share of money moved per tick per unit of unemployment gap")
    flight_return: float = Field(0.05, ge=0, le=1, description="share of fled money brought home per calm tick")
    transfers: float = Field(0.0, ge=0, le=1, description="fiscal union: share of the gap in money per head "
                                                         "moved from richer to poorer towns' households per tick")
    embargoes: list[Embargo] = Field(default_factory=list)


class TradeNetwork:
    def __init__(self, region, params: dict):
        self.p = TradeParams.model_validate(params or {})
        self.region = region
        self.names = region.names
        self.rate = {c: 1.0 for c in self.names}         # value of each currency in a common unit
        self.stats = self.zero()
        self.fx_target = {}
        if not self.p.enabled:
            return
        for k, c in enumerate(self.names):
            self.setup_city(c, k)

    @staticmethod
    def zero():
        return dict(shipped={}, exports={}, imports={}, migrants={}, flight={}, transfers={}, remitted=0.0)

    # ------------------------------------------------------------------ helpers
    def w(self, c):
        return self.region.world(c)

    def m(self, c):
        return self.w(c).state["economy"]["market_layer"]

    def bank(self, c):
        return self.w(c).state["economy"].get("bank_layer")

    def mi(self, c):
        return self.m(c).money_item

    def money(self, c, acc):
        return self.w(c).ledger.balance(acc, self.mi(c))

    def setup_city(self, c: str, k: int) -> None:
        w, m, p = self.w(c), self.m(c), self.p
        L, st = w.ledger, w.state["economy"]
        st["extra_categories"] = list(st.get("extra_categories", [])) + list(CATS)
        holders = st.setdefault("money_holders", [])
        imps = {}
        for j, g in enumerate(p.goods):
            sec = SECTOR_OF[g]
            acc = f"import:{g}"
            L.open_account(acc)
            holders.append(acc)
            dom = m.of(sec)
            price = float(np.mean([f.price for f in dom])) if dom else 1.0
            f = Firm(-(1000 + j), sec, acc, price, 0.0, 0.0)
            f.unit_cost = price / (1 + p.import_markup)
            f.markup = p.import_markup
            f.importer = True
            imps.setdefault(sec, []).append(f)
            self.household_pay(c, acc, p.importer_capital, "trade_capital")
        st["importers"] = imps
        for acc in ("port", *(f"savers:{o}" for o in self.names if o != c)):
            L.open_account(acc)
            holders.append(acc)
        if p.currency == "floating":
            L.open_account("fx")
            holders.append("fx")
            self.household_pay(c, "fx", p.fx_capital * float(m.cash.sum()), "fx")
            self.fx_target[c] = self.money(c, "fx")    # starting holdings: their value shares are the target

    def household_pay(self, c, to_acc, amount, category) -> float:
        """Households pay `amount` (in proportion to their money) to an account in their city."""
        m, L = self.m(c), self.w(c).ledger
        have = float(m.cash.sum())
        x = min(amount, max(have, 0.0))
        if x <= 1e-12:
            return 0.0
        L.post(to_acc, "households", self.mi(c), x, "transfer", category)
        m.cash -= x * m.cash / have
        return x

    def to_households(self, c, from_acc, amount, category, weights=None) -> None:
        m, L = self.m(c), self.w(c).ledger
        if amount <= 1e-12 or len(m.size) == 0:
            return
        L.post("households", from_acc, self.mi(c), amount, "transfer", category)
        wts = m.size if weights is None else weights
        m.cash += amount * wts / wts.sum()

    def fx(self, a, b) -> float:
        """Units of b's money per unit of a's money."""
        return self.rate[a] / self.rate[b] if self.p.currency == "floating" else 1.0

    def cost(self, a, b) -> float:
        key, rev = f"{a}-{b}", f"{b}-{a}"
        return self.p.transport_costs.get(key, self.p.transport_costs.get(rev, self.p.transport_cost))

    def open(self, a, b, tick) -> bool:
        for e in self.p.embargoes:
            if set(e.between) >= {a, b} and e.start <= tick < e.start + e.duration:
                return False
        return True

    # ------------------------------------------------------------------ moving money
    def send(self, a, acc_a, b, acc_b, amount_a: float, category: str) -> float:
        """Move `amount_a` of a's money from acc_a (in a) to acc_b (in b). Returns what arrives (b's money)."""
        if amount_a <= 1e-12:
            return 0.0
        La, Lb = self.w(a).ledger, self.w(b).ledger
        amount_a = min(amount_a, max(self.money(a, acc_a), 0.0))
        if self.p.currency == "floating":
            out = amount_a * self.fx(a, b) * (1 - self.p.fx_spread)       # the dealer keeps a small spread
            have = max(self.money(b, "fx"), 0.0)
            if out > have:                           # the dealer is short of b's money
                out = have
                amount_a = out / (self.fx(a, b) * (1 - self.p.fx_spread))
            if amount_a <= 1e-12:
                return 0.0
            La.post("fx", acc_a, self.mi(a), amount_a, "transfer", "fx")
            Lb.post(acc_b, "fx", self.mi(b), out, "transfer", "fx")
            return out
        # union: deposits leave a's bank with reserves, and arrive at b's bank
        ba, bb = self.bank(a), self.bank(b)
        if ba is not None:
            amount_a = min(amount_a, max(La.balance(BANK, "cash"), 0.0))
            if amount_a <= 1e-12:
                return 0.0
            La.post(BANK, acc_a, "deposits", amount_a, "transfer", "remit_out")
            La.post("ext:export", BANK, "cash", amount_a, "export", "remit_out")
        else:
            La.post("ext:export", acc_a, "cash", amount_a, "export", "remit")
        if bb is not None:
            Lb.post(BANK, "ext:import", "cash", amount_a, "import", "remit_in")
            Lb.post(acc_b, BANK, "deposits", amount_a, "transfer", "remit_in")
        else:
            Lb.post(acc_b, "ext:import", "cash", amount_a, "import", "remit")
        self.stats["remitted"] += amount_a
        return amount_a

    # ------------------------------------------------------------------ phase 1: reset importers
    def reset(self, region) -> None:
        self.stats = self.zero()
        for c in self.names:
            for imps in self.w(c).state["economy"].get("importers", {}).values():
                for f in imps:
                    f.revenue = f.sold = f.orders_in = f.op_costs = 0.0

    # ------------------------------------------------------------------ phase 5: restock
    def restock(self, region) -> None:
        """Importers restock in three steps: orders, foreign-exchange clearing, purchases."""
        p, tick = self.p, region.tick
        books = []                     # one entry per importer: its orders and what it paid
        # 1. orders: each importer picks suppliers by landed cost and sets a budget in its own money
        for b in self.names:
            wb = self.w(b)
            Lb = wb.ledger
            for sec, imps in wb.state["economy"].get("importers", {}).items():
                item = "raw" if sec == FARM else "final"
                dom_b = self.m(b).of(sec)
                for f in imps:
                    # expected sales: its price-based share of the town's orders (the rule buyers use to
                    # split their orders, see MarketEconomy.split_orders), so a cheaper importer grows
                    f.expected = float(B.adaptive(f.expected, max(f.sold, f.orders_in), p.expectation_speed))
                    stock = Lb.balance(f.account, item)
                    want = float(B.production_target(f.expected, stock, p.inventory_target, p.inventory_correction))
                    srcs, landed, avail = [], [], []
                    for a in self.names:
                        if a == b or not self.open(a, b, tick):
                            continue
                        dom = self.m(a).of(sec)
                        if not dom:
                            continue
                        pa = float(np.mean([g.price for g in dom]))
                        srcs.append((a, dom, pa))
                        landed.append(pa * self.fx(a, b) / (1 - self.cost(a, b)))
                        avail.append(sum(self.w(a).ledger.balance(g.account, item) for g in dom))
                    book = dict(b=b, f=f, item=item, orders=[], est=float(min(landed)) if landed else None)
                    books.append(book)
                    if not srcs:
                        continue
                    # merchants only import what beats the local price (no cross-hauling at a loss)
                    local = float(np.mean([g.price for g in dom_b])) if dom_b else np.inf
                    if book["est"] * (1 + p.import_markup) > local or want <= 1e-9:
                        continue
                    alloc = B.allocate_by_price(want, np.array(landed), np.array(avail) * np.array(
                        [1 - self.cost(s_[0], b) for s_ in srcs]), choosiness=4.0)
                    cash = max(self.money(b, f.account), 0.0)
                    for (a, dom, pa), q in zip(srcs, alloc):
                        if q <= 1e-9:
                            continue
                        tau = self.cost(a, b)
                        buy_q = q / (1 - tau)                       # ship enough that q arrives
                        budget = min(buy_q * pa * self.fx(a, b) * 1.05, cash)
                        cash -= budget
                        if budget > 1e-12:
                            book["orders"].append(dict(a=a, dom=dom, pa=pa, tau=tau, buy_q=buy_q, budget=budget))
        # 2. floating: the foreign-exchange market clears. Orders to buy a currency are filled only as far
        # as orders to sell it (plus a little of the dealer's float) allow, and the currency in excess
        # demand appreciates. Trade must (nearly) balance: there are no capital flows to finance it.
        fill = {c: 1.0 for c in self.names}
        if p.currency == "floating":
            demand = {c: 0.0 for c in self.names}
            supply = {c: 0.0 for c in self.names}
            for book in books:
                for o in book["orders"]:
                    v = o["budget"] * self.rate[book["b"]]           # value in the common unit
                    demand[o["a"]] += v
                    supply[book["b"]] += v
            for c in self.names:
                buffer = p.fx_float_use * max(self.money(c, "fx"), 0.0) * self.rate[c]
                if demand[c] > 1e-12:
                    fill[c] = min(1.0, (supply[c] + buffer) / demand[c])
            for c in self.names:
                tot = demand[c] + supply[c]
                if tot > 1e-12:
                    self.rate[c] *= float(np.exp(p.fx_speed * (demand[c] - supply[c]) / tot))
            g = float(np.exp(np.mean([np.log(v) for v in self.rate.values()])))
            for c in self.names:
                self.rate[c] /= g
            self.stats["fx_fill"] = fill
        # 3. purchases
        for book in books:
            b, f, item = book["b"], book["f"], book["item"]
            Lb = self.w(b).ledger
            paid = arrived = 0.0
            for o in book["orders"]:
                a, dom, pa, tau, buy_q = o["a"], o["dom"], o["pa"], o["tau"], o["buy_q"]
                k = fill[a]
                budget_b = o["budget"] * k
                for g in dom:
                    g.orders_in += buy_q / len(dom)
                sent = self.send(b, f.account, a, "port", budget_b, "fx")
                if sent <= 1e-12:
                    continue
                ma, wa = self.m(a), self.w(a)
                got = ma.buy(wa, "port", dom, buy_q * k, item, "export", self.money(a, "port"))
                if got > 1e-12:
                    wa.ledger.post("ext:export", "port", item, got, "export", "shipment")
                    Lb.post(f.account, "ext:import", item, got, "import", "shipment")
                    Lb.post("ext:waste", f.account, item, got * tau, "waste", "transport")
                    self.stats["exports"][a] = self.stats["exports"].get(a, 0.0) + got * pa
                    key = (a, b, item)
                    self.stats["shipped"][key] = self.stats["shipped"].get(key, 0.0) + got
                back = self.send(a, "port", b, f.account, self.money(a, "port"), "fx")
                left = self.money(a, "port")          # rarely, the dealer cannot convert the change back:
                if left > 1e-12:                      # the supplier town keeps it (counted as a payment)
                    self.to_households(a, "port", left, "export")
                cost_b = budget_b - back
                paid += cost_b
                arrived += got * (1 - tau)
                self.stats["imports"][b] = self.stats["imports"].get(b, 0.0) + cost_b
            f.op_costs = paid
            # pricing: landed cost (what it paid, or the cheapest quote) plus a markup that
            # rises when its shelves are short and falls when they overflow
            unit = paid / arrived if arrived > 1e-9 else book["est"]
            if unit is not None:
                f.unit_cost = float(B.adaptive(f.unit_cost, unit, p.expectation_speed)) if f.unit_cost > 0 else unit
            gap = B.inventory_gap(Lb.balance(f.account, item), p.inventory_target * f.expected)
            f.markup = B.markup_update(f.markup, gap, p.price_speed, p.import_markup, 0.1, 0.0, 1.0)
            f.price = (1 + f.markup) * f.unit_cost       # never below landed cost
            # working capital: owners (the town's households) keep about two ticks of purchases in
            # the business, topping it up when short and taking out what is beyond it
            target = p.capital_ticks * f.expected * f.unit_cost + p.importer_capital
            have = self.money(b, f.account)
            if have < target:
                self.household_pay(b, f.account, p.capital_speed * (target - have), "trade_capital")
            elif have > 2 * target:
                self.to_households(b, f.account, p.payout * (have - 2 * target), "import_dividends")

    # ------------------------------------------------------------------ phase 8: people and capital move
    def real_income(self, c) -> float:
        m, w = self.m(c), self.w(c)
        N = float(m.size.sum())
        if N <= 0:
            return 0.0
        working = sum(f.hired + f.builders for f in m.firms) + w.vars.get("public_workers", 0.0)
        employment = min(working / N, 1.0)
        wage = float(np.mean([f.wage for f in m.firms])) if m.firms else 0.0
        price = w.vars.get("price_level", 1.0) or 1.0
        return wage * employment / price

    def unemployment(self, c) -> float:
        m, w = self.m(c), self.w(c)
        N = float(m.size.sum())
        working = sum(f.hired + f.builders for f in m.firms) + w.vars.get("public_workers", 0.0)
        return float(np.clip(1 - working / N, 0, 1)) if N > 0 else 0.0

    def move_people(self, region) -> None:
        p = self.p
        if p.migration <= 0:
            return
        inc = {c: self.real_income(c) for c in self.names}
        for a in self.names:
            best = max((c for c in self.names if c != a), key=lambda c: inc[c], default=None)
            if best is None or inc[a] <= 0:
                continue
            gap = inc[best] / inc[a] - 1 - p.migration_cost
            if gap <= 0:
                continue
            ma, mb = self.m(a), self.m(best)
            n = int(p.migration * gap * len(ma.size))
            if n <= 0 or len(ma.size) <= n + 10:
                continue
            rng = ma.rng
            jobless = np.where(ma.employer < 0)[0]
            pick = rng.permutation(jobless)[:n] if len(jobless) >= n else \
                np.concatenate([jobless, rng.permutation(np.where(ma.employer >= 0)[0])[: n - len(jobless)]])
            sectors = ma.sector[pick].copy()
            cash = ma.cash[pick].copy()
            money = ma.remove_agents(self.w(a), pick)
            # their money leaves a's households account and arrives on b's
            La = self.w(a).ledger
            money = max(money, 0.0)
            arrived = 0.0
            if money > 1e-12:
                La.post("port", "households", self.mi(a), money, "transfer", "migration")
                arrived = self.send(a, "port", best, "port", money, "migration")
                if arrived > 1e-12:
                    self.w(best).ledger.post("households", "port", self.mi(best), arrived, "transfer", "migration")
            cash = np.maximum(cash, 0.0)
            share = cash / cash.sum() if cash.sum() > 0 else np.full(len(cash), 1 / len(cash))
            mb.add_agents(self.w(best), sectors, arrived * share)
            if self.money(a, "port") > 1e-12:          # what the dealer could not convert stays at home
                self.to_households(a, "port", self.money(a, "port"), "migration")
            self.stats["migrants"][(a, best)] = n * ma.p.agent_size

    def move_capital(self, region) -> None:
        p = self.p
        if p.capital_flight <= 0:
            return
        u = {c: self.unemployment(c) for c in self.names}
        for a in self.names:
            best = min((c for c in self.names if c != a), key=lambda c: u[c], default=None)
            if best is None:
                continue
            gap = u[a] - u[best]
            if gap > 0.02:
                x = p.capital_flight * gap * float(self.m(a).cash.sum())
                moved = self.household_pay(a, "port", x, "capital_flight")
                self.send(a, "port", best, f"savers:{a}", moved, "capital_flight")
                left = self.money(a, "port")                 # what could not be sent stays at home
                if left > 1e-12:
                    self.to_households(a, "port", left, "capital_flight")
                self.stats["flight"][a] = self.stats["flight"].get(a, 0.0) + moved - max(left, 0.0)
            else:
                for c in self.names:                     # bring it home
                    if c == a:
                        continue
                    held = self.money(c, f"savers:{a}")
                    if held > 1e-9:
                        back = self.send(c, f"savers:{a}", a, "port", p.flight_return * held, "capital_flight")
                        if back > 1e-12:
                            self.to_households(a, "port", back, "capital_flight")
                            self.stats["flight"][a] = self.stats["flight"].get(a, 0.0) - back

    def fiscal_transfers(self, region) -> None:
        """A union budget evens out money per head: richer towns' households pay, poorer ones receive."""
        p = self.p
        if p.transfers <= 0 or p.currency != "union":
            return
        pop = {c: float(self.m(c).size.sum()) for c in self.names}
        money = {c: float(self.m(c).cash.sum()) for c in self.names}
        mean = sum(money.values()) / max(sum(pop.values()), 1e-9)
        gap = {c: p.transfers * (money[c] - mean * pop[c]) for c in self.names}
        payers = {c: g for c, g in gap.items() if g > 1e-9}
        takers = {c: -g for c, g in gap.items() if g < -1e-9}
        need = sum(takers.values())
        for a, amount in payers.items():
            paid = self.household_pay(a, "port", amount, "fiscal_transfer")
            for b, want in takers.items():
                got = self.send(a, "port", b, "port", paid * want / need, "fiscal_transfer")
                self.to_households(b, "port", got, "fiscal_transfer")
                self.stats["transfers"][b] = self.stats["transfers"].get(b, 0.0) + got
            left = self.money(a, "port")
            if left > 1e-12:
                self.to_households(a, "port", left, "fiscal_transfer")

    def end_of_tick(self, region) -> None:
        if not self.p.enabled:
            return
        self.move_people(region)
        self.move_capital(region)
        self.fiscal_transfers(region)

    def hooks(self):
        if not self.p.enabled:
            return {}
        return {Phase.SHOCKS_POLICY: self.reset, Phase.DISTRIBUTION: self.restock, Phase.DEMOGRAPHY: self.end_of_tick}

    # ------------------------------------------------------------------ region invariants
    def check(self, region) -> list[str]:
        out = []
        items = set()
        for c in self.names:
            items |= set(self.w(c).ledger.items)
        for item in items:
            ex = sum(p.amount for c in self.names for p in self.w(c).ledger.journal if p.item == item and p.flow == "export")
            im = sum(p.amount for c in self.names for p in self.w(c).ledger.journal if p.item == item and p.flow == "import")
            if abs(ex - im) > 1e-9 * max(1.0, ex, im):
                out.append(f"{item}: exported {ex:.9g} but imported {im:.9g}")
            if self.p.currency == "floating" and self.w(self.names[0]).ledger.items.get(item) == "money" and ex > 0:
                out.append(f"money crossed a border under floating rates: {item} {ex}")
        return out

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        def by_city(fn):
            return lambda r: {c: fn(c) for c in self.names}

        def shipped(r):
            out = {}
            for (a, b, item), q in self.stats["shipped"].items():
                out[f"{item}:{a}->{b}"] = q
            return out

        return [
            Metric("exchange_rate", by_city(lambda c: self.rate[c]), "value of each city's money (floating)", "x"),
            Metric("exports", by_city(lambda c: self.stats["exports"].get(c, 0.0)), "value of goods sold abroad (local money)", "money"),
            Metric("imports", by_city(lambda c: self.stats["imports"].get(c, 0.0)), "paid for goods bought abroad (local money)", "money"),
            Metric("shipments", shipped, "units shipped per good and route", "units"),
            Metric("migrants", lambda r: {f"{a}->{b}": n for (a, b), n in self.stats["migrants"].items()}, "people moving", "people"),
            Metric("capital_flight", by_city(lambda c: self.stats["flight"].get(c, 0.0)), "money sent abroad by households (net)", "money"),
            Metric("savings_abroad", by_city(lambda c: sum(self.money(o, f"savers:{c}") for o in self.names if o != c)
                                             if self.p.enabled else 0.0), "households' money held abroad", "money"),
            Metric("real_income", by_city(self.real_income), "average wage x employment / bread price", "loaves"),
            Metric("trade_balance", by_city(lambda c: self.stats["exports"].get(c, 0.0) - self.stats["imports"].get(c, 0.0)),
                   "exports - imports (local money)", "money"),
            Metric("union_transfers", by_city(lambda c: self.stats["transfers"].get(c, 0.0)), "fiscal-union transfers received", "money"),
        ]
