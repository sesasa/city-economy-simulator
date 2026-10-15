"""Stage 11: the internet arrives — a breakthrough, a new market, a bubble and a lasting change.

At `start` (1993 Q2, the first popular web browser), the university's networking research
becomes usable by everyone. From then on:

* Adoption: households and businesses take it up along Bass S-curves (innovators + imitators).
* New markets: households shop online (a growing share of their goods budget, delivered by
  online shops that buy from the factories and need fewer staff than high-street shops) and pay
  for internet access (telecom, which buys wholesale bandwidth from the carriers). Businesses move
  part of their advertising online, to portals and search sites.
* Startups: venture investors (the owners) put a share of their wealth into ventures, more when
  the "new economy" mood is high; part of the money seeds new ventures, the rest funds later
  rounds. Investors learn a startup's quality as it ages and stop funding the weakest (less
  strictly in a mania). A startup hires to spend each round, buys servers from the router maker,
  hosting from the carriers and marketing from the services sector, and wins customers through
  quality, size and network effects. The best go public while the IPO window is open; weak ones
  are bought by a leader or fail when their cash runs out.
* New supply chain: three fiber carriers and a router maker are new listed companies. Carriers
  build capacity for the traffic they expect — in the boom they believe traffic doubles every
  hundred days — buying routers from the router maker and construction from the factories, on
  bank credit. When traffic falls short, prices of bandwidth collapse, the network sits mostly
  dark, and over-indebted carriers go bankrupt; the bank's losses make credit dearer for all.
* Valuation: new-economy companies are valued at a multiple of revenue that swells with the
  mood, which feeds on its own momentum, on the buzz of adoption and on the belief that the
  central bank will cushion any fall; it fades with failures and dearer money.
* Spillovers: businesses that adopt raise their productivity (retail most), slowly, as they
  learn and install IT.
* History (optional, `history: true`): a warning about "irrational exuberance" (Dec 1996); a
  global financial scare in autumn 1998 (risk premiums jump; the central bank cuts three times);
  the Y2K liquidity injection (late 1999, one quarter); and, if jobs are still scarce, the 2003
  deflation scare (rates held at 1% "for a considerable period", until mid-2004).
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
from .town import BANK, HH, INS, MONEY, year_of

CATS = ("shop", "portal")       # online shops (e-commerce) and portals/search (advertising)
STATUS = {"private": 0, "public": 1, "failed": 2, "acquired": 3}


@dataclass
class Startup:
    id: int
    cat: str
    quality: float
    signal: float               # what investors will learn about its quality
    born: int
    acc: str
    item: str
    L: float = 0.0
    revenue: float = 0.0
    revenue_prev: float = 0.0
    growth: float = 0.0
    reach: float = 0.01
    raised: float = 0.0
    valuation: float = 0.0
    shares: float = 1000.0
    status: str = "private"
    died: int = -1
    public_at: int = -1
    acquired_by: int = -1
    peak_value: float = 0.0
    profit: float = 0.0
    staff_cost: float = 0.0
    plan: float = 0.0
    value_prev: float = 0.0
    trail: list = field(default_factory=list)


@dataclass
class Infra:
    """A new-economy infrastructure company: a fiber carrier or the router maker."""
    name: str
    kind: str
    acc: str
    item: str
    L: float = 0.0
    capacity: float = 0.0       # carriers: traffic it can carry
    orders: float = 0.0         # router maker: units ordered this quarter
    sold: float = 0.0
    expected: float = 0.0
    revenue: float = 0.0
    costs: float = 0.0
    profit: float = 0.0
    valuation: float = 0.0
    value_prev: float = 0.0
    shares: float = 1000.0
    bankrupt: int = -1
    distress: int = 0
    growth: float = 0.0
    revenue_prev: float = 0.0
    appetite: float = 1.0       # carriers: how boldly it builds and borrows
    legacy: float = 1.0         # carriers: weight in the old voice business
    trail: list = field(default_factory=list)


class InternetParams(LayerParams):
    start: int = Field(54, ge=1, description="breakthrough tick (the first popular web browser)")
    hh_innovation: float = Field(0.006, ge=0)
    hh_imitation: float = Field(0.13, ge=0)
    hh_ceiling: float = Field(0.8, ge=0, le=1)
    firm_innovation: float = Field(0.004, ge=0)
    firm_imitation: float = Field(0.12, ge=0)
    firm_ceiling: float = Field(0.95, ge=0, le=1)
    ecommerce_max: float = Field(0.15, ge=0, le=1, description="share of the goods budget bought online at full adoption")
    ecommerce_ramp: float = Field(0.03, ge=0, description="per quarter: how fast online shopping habits form")
    access_share: float = Field(0.012, ge=0, description="share of household budget spent on internet access")
    wholesale_share: float = Field(0.5, ge=0, le=1, description="share of access revenue telecom pays carriers for bandwidth")
    ad_rate: float = Field(0.012, ge=0, description="share of business revenue spent on advertising")
    ad_online_max: float = Field(0.5, ge=0, le=1, description="share of advertising that can move online")
    ecommerce_margin: float = Field(0.12, ge=0, description="online shops' markup over factory prices")
    gains: dict[str, float] = Field(default_factory=lambda: {"retail": 0.35, "finance": 0.30, "telecom": 0.25,
                                                             "factory": 0.15, "services": 0.10, "farm": 0.05})
    learning: float = Field(0.05, ge=0, le=1, description="per quarter: how fast productivity gains are realised")
    it_boom: float = Field(1.0, ge=0, description="extra IT investment, share of normal investment, at full mood")
    it_share_max: float = Field(0.35, ge=0, le=1, description="IT share of business investment at full adoption")
    # startups
    seed: float = Field(200.0, gt=0, description="seed round ($)")
    vc_rate: float = Field(0.0012, ge=0, description="share of owners' wealth put into ventures per quarter at mood 0")
    vc_mood: float = Field(1.0, ge=0, description="venture money x exp(this x mood); the mood counts down to -1 only (VC never stops entirely)")
    seed_share: float = Field(0.35, ge=0, le=1, description="share of venture money kept for new ventures")
    max_new: int = Field(14, ge=0, description="most startups founded in a quarter")
    quality_sd: float = Field(0.8, ge=0)
    signal_noise: float = Field(0.5, ge=0, description="noise in what investors learn about quality")
    triage: float = Field(0.2, ge=0, le=1, description="share of older private startups investors stop funding (calm market)")
    triage_age: int = Field(8, ge=0, description="quarters before investors judge a startup")
    triage_mood: float = Field(0.5, gt=0, description="mood at which investors stop being choosy")
    network: dict[str, float] = Field(default_factory=lambda: {"shop": 0.35, "portal": 0.6},
                                      description="network-effect exponent per market (higher: winner takes more)")
    staff_productivity: dict[str, float] = Field(default_factory=lambda: {"shop": 45.0, "portal": 30.0},
                                                 description="revenue a startup worker can handle per quarter ($)")
    tech_premium: float = Field(1.6, ge=1, description="startup wages relative to the base wage")
    overhead: float = Field(0.55, ge=0, description="marketing, servers and hosting per dollar of wages")
    runway: float = Field(6.0, gt=0, description="quarters of spending a startup plans its hiring around")
    hire_speed: float = Field(0.5, gt=0, le=1)
    round_quarters: float = Field(5.0, gt=0, description="a funding round covers this many quarters of spending")
    ipo_age: int = Field(6, ge=0, description="quarters before a startup may go public")
    ipo_mood: float = Field(0.7, description="the IPO window opens when the mood is above this")
    ipo_top: float = Field(0.15, gt=0, le=1, description="only the top share of private startups by revenue can list")
    ipo_slots: int = Field(2, ge=0, description="most IPOs per quarter")
    ipo_fraction: float = Field(0.2, gt=0, le=1, description="share of the company sold at the IPO")
    lean: float = Field(1.2, gt=0, description="in survival mode a startup keeps the staff its revenue needs / this")
    plug: float = Field(0.0, ge=0, le=1, description="chance per quarter, in a sour market, that investors close a startup far from profit")
    acquisition_chance: float = Field(0.6, ge=0, le=1, description="chance a dying startup is bought by its market leader")
    multiple: float = Field(3.0, gt=0, description="value / annual revenue at mood 0")
    mood_multiple: float = Field(1.1, ge=0, description="log multiple per unit of mood")
    max_multiple: float = Field(150.0, gt=0)
    mood_momentum: float = Field(0.8, ge=0, description="mood rise per unit of the new-economy index's quarterly return")
    mood_buzz: float = Field(8.0, ge=0, description="mood rise per unit of quarterly adoption growth")
    novelty: float = Field(2.0, ge=0, description="buzz is full until half of the eventual users are online, then fades")
    mood_failures: float = Field(4.0, ge=0, description="mood fall per unit of the startup failure rate")
    mood_rates: float = Field(4.0, ge=0, description="mood fall per unit of rate rises (annual) over the past year")
    mood_drawdown: float = Field(0.8, ge=0, description="mood fall per quarter per unit of the index's fall from its peak")
    peak_memory: float = Field(0.8, ge=0, le=1, description="share of the memory of the market's peak kept each quarter")
    mood_put: float = Field(0.15, ge=0, description="mood rise when the central bank cuts to steady markets (the 'put')")
    crowd: float = Field(12.0, gt=0, description="listed internet companies before prices sway the mood fully")
    mood_decay: float = Field(0.12, ge=0, le=1)
    mood_max: float = Field(3.0)
    # the network: carriers and the router maker
    voice_traffic: float = Field(3000.0, ge=0, description="traffic the carriers already carry (phone calls)")
    legacy_share: float = Field(0.25, ge=0, le=1, description="share of telecom's revenue paid to carriers for long-distance capacity")
    carrier_leverage: float = Field(3.0, gt=0, description="bank lends carriers up to this x annual revenue (more in a mania)")
    carriers: list[str] = Field(default_factory=lambda: ["Longwire", "TransMeridian", "Beacon Fiber"])
    appetite: list[float] = Field(default_factory=lambda: [0.25, 1.0, 1.6], description="each carrier's boldness in building and borrowing")
    legacy_weight: list[float] = Field(default_factory=lambda: [2.0, 1.0, 1.0], description="each carrier's weight in the old voice business (the incumbent first)")
    router_maker: str = Field("Corelink")
    infra_equity: float = Field(1500.0, gt=0, description="money each infrastructure company raises when it lists")
    usage_growth: float = Field(0.12, ge=0, description="traffic per user growth per quarter")
    hype_growth: float = Field(0.9, ge=0, description="traffic growth per quarter the carriers believe in at full mood")
    target_utilization: float = Field(0.75, gt=0, le=1, description="share of capacity carriers plan to have in use")
    build_horizon: float = Field(8.0, gt=0, description="quarters ahead the carriers build for")
    build_speed: float = Field(0.15, gt=0, le=1)
    capacity_cost: float = Field(0.15, gt=0, description="equipment units per unit of traffic capacity (falls as usage grows)")
    glut: float = Field(0.35, gt=0, le=1, description="utilization below which carriers start a price war")
    price_war: float = Field(0.15, ge=0, description="bandwidth prices fall each quarter by (utilization / glut)^this")
    legacy_exposure: float = Field(0.3, ge=0, description="how much the price war reaches the carriers' old voice business (0: not at all)")
    price_recovery: float = Field(0.02, ge=0, le=1, description="share of the price gap recovered per quarter once capacity is in use")
    hosting_share: float = Field(0.25, ge=0, le=1, description="share of startups' overhead paid to carriers for hosting")
    router_productivity: float = Field(20.0, gt=0, description="equipment units per router-maker worker per quarter")
    router_markup: float = Field(0.45, ge=0)
    components: float = Field(0.3, ge=0, le=1, description="share of router revenue spent on factory components")
    carrier_spread: float = Field(0.02, ge=0, description="carriers' extra annual loan rate in calm markets")
    junk_spread: float = Field(0.08, ge=0, description="further extra rate when the mood is at its worst (the junk-bond market shuts)")
    amortization: float = Field(0.025, ge=0, le=1, description="share of a carrier's loans repaid per quarter")
    max_infra_multiple: float = Field(25.0, gt=0, description="highest value / annual revenue for carriers and the router maker")
    bankruptcy_quarters: int = Field(2, ge=1, description="quarters a carrier can miss its interest before bankruptcy")
    reorganise: int = Field(4, ge=1, description="quarters until a bankrupt carrier emerges and builds again")
    write_off: float = Field(0.6, ge=0, le=1, description="share of a bankrupt carrier's loans the bank loses")
    history: bool = Field(True, description="script the 1996 warning, the 1998 scare and Y2K")


def tick_of(year: float) -> int:
    from .town import START_YEAR
    return int(round((year - START_YEAR[0]) * 4)) + 1


class Internet(Layer):
    name = "internet"
    requires = ["town"]
    Params = InternetParams
    p: InternetParams

    def setup(self, world) -> None:
        st = world.state["economy"]
        self.town = st["town"]
        self.world = world
        self.rng = np.random.default_rng([world.seed, 1102])
        self.hh = self.firms = self.realised = self.ecom = self.mood = 0.0
        self.hh_hist = []
        self.startups: list[Startup] = []
        self.infra: list[Infra] = []
        self.next_id = 0
        self.traffic = self.capacity = 0.0
        self.usage = 1.0
        self.belief = 0.0
        self.bw_price = 1.0             # bandwidth price relative to before the internet
        self.rates = []
        self.t = self.zero()
        self.cum = dict(founded=0, failed=0, ipos=0, acquired=0, vc=0.0, ipo_money=0.0, defaults=0.0, bankruptcies=0)
        self.episode = ""
        self.peak = 0.0
        self.cutting = False
        st["consumer_markets"].append(self)
        st["extra_employment"].append(lambda: sum(s.L for s in self.alive()) + sum(f.L for f in self.infra))
        st.setdefault("extra_employment_named", []).append(lambda: {"internet": sum(s.L for s in self.alive()),
                                                                     "network": sum(f.L for f in self.infra)})
        st["equity_markets"].append(self.holdings_value)
        st.setdefault("indices", []).append(self)
        st.setdefault("payrolls", []).append(self.payroll)
        st.setdefault("finance_hooks", []).append(self.finance)
        st["hooks_capex"].append(self.capex)
        st.setdefault("hooks_build", []).append(self.build)
        # businesses buy their computers and network gear from the router maker once it exists
        st["it_vendor"] = lambda acc, money: (self.router_sale(world, acc, money / self.router_price(), "computers")
                                              if self.router() is not None else 0.0)
        self.categories = ("online_shop", "access")

    @staticmethod
    def zero():
        return dict(online=0.0, access=0.0, ads=0.0, vc=0.0, seed=0.0, ipo=0.0, founded=0, failed=0, ipos=0, acquired=0,
                    servers=0.0, routers=0.0, marketing=0.0, hosting=0.0, wholesale=0.0, fiber=0.0, defaults=0.0, bankruptcies=0)

    def alive(self):
        return [s for s in self.startups if s.status in ("private", "public")]

    def carriers(self):
        return [f for f in self.infra if f.kind == "carrier"]

    def router(self):
        return next((f for f in self.infra if f.kind == "router"), None)

    def active(self, world) -> bool:
        return world.tick >= self.p.start

    # ------------------------------------------------------------------ budget shift (town.markets)
    def shift(self) -> dict:
        p, tp = self.p, self.town.p
        online = self.ecom * p.ecommerce_max * tp.budget["goods"]
        access = self.hh * p.access_share
        return {"goods": -online, "online_shop": online, "services": -access, "access": access}

    def buy(self, world, h, k, money) -> float:
        T = self.town
        if k == "access":
            tel = T.sec["telecom"]
            units = min(money / tel.price, max(tel.capacity - tel.sold, 0.0))
            got = T.sell("firm:telecom", h, "telecom", units, tel.price, "internet access", consume=True)
            tel.sold += got
            tel.revenue += got * tel.price
            tel.demand += money / tel.price
            self.t["access"] += got * tel.price
            # telecom buys wholesale bandwidth from the carriers; a glut drives its price down
            pay = self.p.wholesale_share * got * tel.price * self.bw_price
            self.pay_carriers("firm:telecom", pay, "wholesale bandwidth")
            self.t["wholesale"] += pay
            return got * tel.price
        shops = [s for s in self.alive() if s.cat == "shop"]
        if not shops or money <= 1e-9:
            return 0.0
        fac = T.sec["factory"]
        price = fac.price * (1 + self.p.ecommerce_margin)
        spent = 0.0
        for s, share in zip(shops, self.shares(shops)):
            cap = s.L * self.p.staff_productivity["shop"] * T.w / T.p.base_wage - s.revenue
            m = min(money * share, max(cap, 0.0))
            units = min(m / price, max(fac.capacity - fac.sold, 0.0))
            if units <= 1e-9:
                continue
            got = T.sell("firm:factory", s.acc, "goods", units, fac.price, "goods for online shops")
            fac.sold += got
            fac.revenue += got * fac.price
            fac.demand += got
            got = T.sell(s.acc, h, "goods", got, price, "online shopping", consume=True, produce=False)
            s.revenue += got * price
            spent += got * price
            T.t["real_consumption"] += got
        self.t["online"] += spent
        return spent

    def pay_carriers(self, frm, amount, note, even=False):
        """New traffic goes to carriers in proportion to their capacity; the old voice business (even=True)
        stays split as it always was."""
        cs = [c for c in self.carriers() if c.capacity > 0]
        tot = sum(c.legacy if even else c.capacity for c in cs)
        for c in cs:
            x = self.town.pay(frm, c.acc, amount * (c.legacy if even else c.capacity) / tot, note, borrow=frm.startswith("firm:"))
            c.revenue += x

    def shares(self, group) -> np.ndarray:
        if not group:
            return np.array([])
        nu = self.p.network[group[0].cat]
        a = np.array([s.quality * (1 + s.L) ** 0.5 * (s.reach + 1e-3) ** nu for s in group])
        return a / a.sum()

    # ------------------------------------------------------------------ the infrastructure companies
    def list_infra(self, world):
        p, T = self.p, self.town
        L = world.ledger
        for name, kind in [(n, "carrier") for n in p.carriers] + [(p.router_maker, "router")]:
            slug = name.lower().replace(" ", "_")
            f = Infra(name, kind, f"firm:{slug}", f"shares:{slug}")
            L.open_account(f.acc)
            L.register_item(f.item, "claim")
            for acc, w_ in (("hh:owners", 0.65), ("hh:workers", 0.25), (INS, 0.10)):
                x = min(p.infra_equity * w_, 0.3 * max(T.bal(acc), 0.0))
                L.post(acc, f.acc, f.item, f.shares * w_, "transfer", "listing")
                T.pay(acc, f.acc, x, "listing", borrow=False)
            f.valuation = p.infra_equity
            f.L = 2.0
            if kind == "carrier":
                f.appetite = p.appetite[p.carriers.index(name) % len(p.appetite)]
                f.legacy = p.legacy_weight[p.carriers.index(name) % len(p.legacy_weight)]
                f.capacity = 1.3 * p.voice_traffic / len(p.carriers)
                f.revenue_prev = p.legacy_share * T.sec["telecom"].revenue / len(p.carriers)
            self.infra.append(f)
        self.capacity = sum(c.capacity for c in self.carriers())

    def router_price(self) -> float:
        T = self.town
        return (1 + self.p.router_markup) * T.w * 1.3 / (self.p.router_productivity * T.trend)

    def router_sale(self, world, buyer, units, note) -> float:
        """The router maker sells equipment (it_equipment), installed as the buyer's capital."""
        r, T = self.router(), self.town
        if r is None or units <= 1e-12:
            return 0.0
        r.orders += units
        cap = r.L * self.p.router_productivity * T.trend - r.sold
        units = min(units, max(cap, 0.0))
        if units <= 1e-12:
            return 0.0
        price = self.router_price()
        got = T.sell(r.acc, buyer, "it_equipment", units, price, note)
        L = world.ledger
        fac = T.sec["factory"]
        L.remove(buyer, "it_equipment", got, "use", "installed")
        L.produce(buyer, "capital", got * price / fac.price, "installed")     # capital is counted at factory prices
        r.sold += got
        r.revenue += got * price
        # components from the factories
        comp = min(self.p.components * got * price / fac.price, max(fac.capacity - fac.sold, 0.0))
        c = T.sell("firm:factory", r.acc, "goods", comp, fac.price, "router components")
        L.remove(r.acc, "goods", c, "use", "components")
        fac.sold += c
        fac.revenue += c * fac.price
        r.costs += c * fac.price
        return got * price

    # ------------------------------------------------------------------ phase 1: adoption, funding, hiring
    def plan(self, world) -> None:
        p, T = self.p, self.town
        self.t = self.zero()
        if not self.active(world):
            return
        if not self.infra:
            self.list_infra(world)
        self.episodes(world)
        self.hh += (p.hh_innovation + p.hh_imitation * self.hh / p.hh_ceiling) * (p.hh_ceiling - self.hh)
        pull = 1 + 2 * self.hh
        self.firms += (p.firm_innovation + p.firm_imitation * pull * self.firms / p.firm_ceiling) * (p.firm_ceiling - self.firms)
        self.ecom += p.ecommerce_ramp * (self.hh - self.ecom)
        self.realised += p.learning * (self.firms - self.realised)
        st = world.state["economy"]
        st["productivity_boost"] = {s: 1 + g * self.realised for s, g in p.gains.items()}
        st["it_share"] = 0.08 + (p.it_share_max - 0.08) * self.firms
        # the "new economy" capex boom: businesses buy extra IT when the mood is high, and cut back when it breaks
        st["it_boom"] = p.it_boom * max(self.mood, -1.0) / p.mood_max * self.firms
        # venture money: a share of owners' wealth, more in a hot market; part kept for new ventures
        pool = p.vc_rate * T.wealth("hh:owners") * math.exp(p.vc_mood * max(self.mood, -1.0)) * min(1.0, self.hh * 20)
        pool = min(pool, 0.5 * max(T.bal("hh:owners"), 0.0))
        seed_pool, follow = p.seed_share * pool, (1 - p.seed_share) * pool
        # investors learn quality as startups age and stop funding the weakest (less strictly in a mania)
        score = lambda s: math.log(s.quality) + s.signal + 2 * s.growth
        older = [s for s in self.alive() if s.status == "private" and world.tick - s.born >= p.triage_age]
        bar = -1e9
        if older:
            cut = p.triage * max(0.0, 1 - max(self.mood, 0.0) / p.triage_mood)     # a mania funds almost anything
            scores = sorted(score(s) for s in older)
            bar = scores[int(cut * (len(scores) - 1))] if cut > 0 else -1e9
        for s in sorted([s for s in self.alive() if s.status == "private"], key=lambda s: -score(s)):
            burn = self.burn(s)
            if T.bal(s.acc) > 2 * burn or follow <= 1e-9:
                continue
            if world.tick - s.born >= p.triage_age and score(s) < bar:
                continue                                  # investors have seen enough
            amount = min(p.round_quarters * max(burn, p.seed / 4), follow)
            self.invest(s, amount, "venture round", ("hh:owners",))
            follow -= amount
            self.t["vc"] += amount
        seed_pool += follow
        n = int(min(p.max_new, seed_pool // p.seed))
        for _ in range(n):
            s = self.found(world)
            self.invest(s, p.seed, "seed round", ("hh:owners",))
            self.t["vc"] += p.seed
            self.t["seed"] += p.seed
        self.cum["vc"] += self.t["vc"]
        # IPOs: the best private startups, a few a quarter, while the window is open
        if self.mood > p.ipo_mood:
            revs = sorted((s.revenue_prev for s in self.alive() if s.status == "private"), reverse=True)
            if revs:
                bar_rev = revs[min(len(revs) - 1, int(p.ipo_top * len(revs)))]
                ready = sorted([s for s in self.alive() if s.status == "private" and world.tick - s.born >= p.ipo_age
                                and s.revenue_prev > 0 and s.growth > 0 and s.revenue_prev >= bar_rev],
                               key=lambda s: -s.revenue_prev)
                for s in ready[:p.ipo_slots]:
                    self.ipo(world, s)
        # hiring
        avail = T.hireable()
        wants = {}
        for s in self.alive():
            work = s.revenue_prev / max(p.staff_productivity[s.cat] * T.w / T.p.base_wage, 1e-9) * 1.2
            runway = T.bal(s.acc) / max(self.burn(s), 1e-9) if s.L > 0 else 99
            # running out of money: shrink to what customers pay for (survival mode)
            wants[("s", s.id)] = min(work / p.lean, s.L) - s.L if (runway < 3 and s.profit <= 0) else max(s.plan, work) - s.L
        r = self.router()
        if r is not None:
            r.sold = 0.0
            wants[("r", 0)] = r.expected / max(p.router_productivity * T.trend, 1e-9) * 1.1 - r.L
        for c in self.carriers():
            wants[("c", c.name)] = 0.002 * c.capacity / max(self.usage, 1.0) + 1.0 - c.L
        tot = sum(max(d, 0.0) for d in wants.values())
        fill = min(1.0, 0.6 * avail / tot) if tot > 1e-9 else 1.0
        for s in self.alive():
            d = wants[("s", s.id)]
            s.L = max(s.L + (p.hire_speed * d * fill if d > 0 else 0.7 * d), 0.0)
            s.revenue_prev, s.revenue = s.revenue, 0.0
        if r is not None:
            d = wants[("r", 0)]
            r.L = max(r.L + (0.4 * d * fill if d > 0 else 0.3 * d), 0.5)
        for c in self.carriers():
            d = wants[("c", c.name)]
            c.L = max(c.L + (0.4 * d * fill if d > 0 else 0.3 * d), 0.5)
        for f in self.infra:
            f.revenue_prev, f.revenue, f.costs = f.revenue, 0.0, 0.0
            f.orders = 0.0

    def episodes(self, world) -> None:
        """Scripted moments from history (optional)."""
        T, p = self.town, self.p
        self.episode = ""
        T.premium_shock = 0.0
        T.spread_shock = 0.0
        if not p.history:
            return
        t = world.tick
        if t == tick_of(1996.75):                          # "irrational exuberance": a brief wobble
            self.mood -= 0.12
            self.episode = "warning"
        if tick_of(1998.5) <= t < tick_of(1999.0):         # the autumn 1998 scare: three quarter-point cuts (Sep-Nov)
            T.premium_shock = 0.015
            if t == tick_of(1998.5):
                self.mood -= 0.2
            T.forced_cut = 0.0025 if t == tick_of(1998.5) else 0.005
            self.episode = "scare"
        T.rate_cap = None
        if tick_of(2003.0) <= t < tick_of(2004.5) and (T.unemployment() > T.p.natural_unemployment
                                                         or T.infl < T.p.inflation_target):
            T.rate_cap = max(0.01, T.rate - 0.005)         # the 2003 deflation scare: down to 1%, held "for a considerable period"
            self.episode = "hold"
        if t == tick_of(1999.75):                          # Y2K: a burst of liquidity
            T.spread_shock = -0.005
            self.mood += 0.3
            self.episode = "y2k"

    def burn(self, s) -> float:
        return s.L * self.town.w * self.p.tech_premium * (1 + self.p.overhead)

    def found(self, world) -> Startup:
        p = self.p
        cat = "shop" if self.rng.random() < 0.45 else "portal"
        s = Startup(self.next_id, cat, float(self.rng.lognormal(0.0, p.quality_sd)), float(self.rng.normal(0, p.signal_noise)),
                    world.tick, f"net:{self.next_id}", f"shares:net{self.next_id}")
        self.next_id += 1
        L = world.ledger
        L.open_account(s.acc)
        L.register_item(s.item, "claim")
        self.startups.append(s)
        self.cum["founded"] += 1
        self.t["founded"] += 1
        return s

    def invest(self, s, amount, note, investors):
        T, L = self.town, self.world.ledger
        pre = max(s.valuation, s.raised * 1.5, amount)
        price = pre / s.shares
        paid = 0.0
        for acc in investors:
            x = min(amount, max(T.bal(acc), 0.0))
            if x <= 1e-9:
                continue
            L.post(acc, s.acc, s.item, x / price, "transfer", note)
            T.pay(acc, s.acc, x, note, borrow=False)
            s.shares += x / price
            paid += x
            amount -= x
        s.raised += paid
        s.valuation = pre + paid
        cost = T.w * self.p.tech_premium * (1 + self.p.overhead)
        runway = self.p.runway * (1 - 0.4 * min(max(self.mood, 0.0) / self.p.mood_max, 1.0))   # "get big fast"
        s.plan = max(s.plan, max(T.bal(s.acc), 0.0) / (runway * cost))
        return paid

    def ipo(self, world, s):
        p, T, L = self.p, self.town, world.ledger
        value = max(s.valuation, self.value_of(s))
        amount = p.ipo_fraction * value
        price = value / s.shares
        got = 0.0
        for acc, w_ in {"hh:owners": 0.65, "hh:workers": 0.25, INS: 0.10}.items():
            x = min(amount * w_, 0.5 * max(T.bal(acc), 0.0))
            if x <= 1e-9:
                continue
            L.post(acc, s.acc, s.item, x / price, "transfer", "IPO")
            T.pay(acc, s.acc, x, "IPO", borrow=False)
            s.shares += x / price
            got += x
        s.raised += got
        cost = T.w * p.tech_premium * (1 + p.overhead)
        s.plan = max(s.plan, max(T.bal(s.acc), 0.0) / (p.runway * cost))
        s.status = "public"
        s.public_at = world.tick
        self.cum["ipos"] += 1
        self.cum["ipo_money"] += got
        self.t["ipos"] += 1
        self.t["ipo"] += got

    # ------------------------------------------------------------------ phase 4: payroll
    def payroll(self, world) -> dict:
        p, T = self.p, self.town
        out = {h: 0.0 for h in HH}
        ow = T.p.owners_wage_share
        for s in self.alive():
            bill = s.L * T.w * p.tech_premium
            have = max(T.bal(s.acc), 0.0)
            if bill > have:
                s.L *= have / bill if bill > 0 else 0.0
                bill = have
            T.pay(s.acc, "hh:workers", bill * (1 - ow), "startup wages", borrow=False)
            T.pay(s.acc, "hh:owners", bill * ow, "startup wages", borrow=False)
            out["hh:workers"] += bill * (1 - ow)
            out["hh:owners"] += bill * ow
            s.staff_cost = bill
        for f in self.infra:                               # established credit: they can borrow for payroll
            bill = f.L * T.w * 1.3
            paid = T.pay(f.acc, "hh:workers", bill * (1 - ow), "network wages")
            paid += T.pay(f.acc, "hh:owners", bill * ow, "network wages")
            out["hh:workers"] += bill * (1 - ow)
            out["hh:owners"] += bill * ow
            f.costs += paid
        return out

    # ------------------------------------------------------------------ phase 5/6: spending, building the network
    def capex(self, world) -> None:
        p, T = self.p, self.town
        if not self.active(world):
            return
        sv = T.sec["services"]
        rp = self.router_price()
        for s in self.alive():
            spend = min(s.L * T.w * p.tech_premium * p.overhead, max(T.bal(s.acc), 0.0) * 0.5)
            mk = 0.5 * spend
            units = min(mk / sv.price, max(sv.capacity - sv.sold, 0.0))
            got = T.sell("firm:services", s.acc, "services", units, sv.price, "marketing", consume=True)
            sv.sold += got
            sv.revenue += got * sv.price
            sv.demand += mk / sv.price
            self.t["marketing"] += got * sv.price
            before = T.bal(s.acc)
            self.pay_carriers(s.acc, p.hosting_share * spend, "hosting")
            self.t["hosting"] += before - T.bal(s.acc)
            servers = max(1 - 0.5 - p.hosting_share, 0.0) * spend
            bought = self.router_sale(world, s.acc, servers / rp, "servers")
            self.t["servers"] += bought
            T.t["investment"] += bought                                 # servers are investment too
            T.t["it_investment"] += bought
        portals = [s for s in self.alive() if s.cat == "portal"]
        if portals:
            shares = self.shares(portals)
            for sname, sec in T.sec.items():
                budget = p.ad_rate * p.ad_online_max * self.firms * self.hh * sec.revenue
                if budget <= 1e-9:
                    continue
                for s, sh in zip(portals, shares):
                    cap = s.L * p.staff_productivity["portal"] * T.w / T.p.base_wage - s.revenue
                    x = T.pay(f"firm:{sname}", s.acc, min(budget * sh, max(cap, 0.0)), "online advertising")
                    s.revenue += x
                    sec.costs += x
                    self.t["ads"] += x

    def build(self, world) -> None:
        """Carriers build for the traffic they expect, on credit (before other businesses invest: in a boom
        they outbid everyone for routers and construction)."""
        p, T = self.p, self.town
        if not self.active(world):
            return
        rp = self.router_price()
        for c in self.carriers():                         # carriers borrow at junk rates, which soar when the mood sours
            T.extra_spread[c.acc] = p.carrier_spread + p.junk_spread * min(max(-self.mood, 0.0), 2.0) / 2
        self.usage *= 1 + p.usage_growth
        self.traffic = p.voice_traffic + self.hh * T.N * self.usage
        # carriers' established business: telecom pays them for long-distance capacity (prices fall in a glut)
        util = min(self.traffic / self.capacity, 1.0) if self.capacity > 1e-9 else 1.0
        if util < p.glut:                                  # a price war: idle capacity undercuts, prices stay down
            self.bw_price *= (util / p.glut) ** p.price_war
        else:
            self.bw_price += p.price_recovery * (1 - self.bw_price)
        legacy = p.legacy_share * T.sec["telecom"].revenue * self.bw_price ** p.legacy_exposure
        self.pay_carriers("firm:telecom", legacy, "long-distance capacity", even=True)
        self.t["wholesale"] += legacy
        believed = max(p.usage_growth, p.hype_growth * max(self.mood, 0.0) / p.mood_max)
        self.belief = believed
        target = self.traffic * (1 + believed) ** p.build_horizon / p.target_utilization
        build = p.build_speed * max(target - self.capacity, 0.0)
        # the bank lends for fiber up to a multiple of a carrier's revenue — a multiple that loosens in a mania
        limit = p.carrier_leverage * (1 + max(self.mood, 0.0) / p.mood_max * 1.5) if self.mood > -0.5 else 0.5
        cs = self.carriers()
        avg = sum(c.revenue_prev for c in cs) / max(len(cs), 1)            # the bank judges the industry's prospects
        for c in cs:                                       # a year after bankruptcy, a carrier emerges under its creditors
            if c.bankrupt >= 0 and world.tick == c.bankrupt + p.reorganise:
                c.appetite = min(c.appetite, 0.5)
        up = [c for c in cs if c.bankrupt < 0 or world.tick >= c.bankrupt + p.reorganise]
        live = up                                          # credit is rationed below, cash of their own is not
        fac = T.sec["factory"]
        bold = sum(c.appetite for c in up)
        for c in self.rng.permutation(live):
            units = build * c.appetite / max(bold, 1e-9) * p.capacity_cost / self.usage
            if units <= 1e-12:
                continue
            own = max(T.bal(c.acc) - 2 * c.costs, 0.0) * min(1.0, c.appetite)  # cash of its own (the cautious keep a reserve), then credit
            room = max(limit * 4 * avg * c.appetite - T.debt(c.acc), 0.0) + own
            units = min(units, room / (0.5 * rp + 0.5 * fac.price))
            spent = self.router_sale(world, c.acc, 0.5 * units, "routers")
            m = min(0.5 * units, max(fac.capacity - fac.sold, 0.0))
            g = T.sell("firm:factory", c.acc, "machines", m, fac.price, "fiber construction")
            fac.sold += g
            fac.revenue += g * fac.price
            fac.demand += 0.5 * units
            if 0.5 * units - g > 1e-12 and T.p.imports:              # cable and equipment from abroad
                g += T.import_goods(c.acc, "machines", 0.5 * units - g, fac.price, "imported fiber equipment")
            world.ledger.remove(c.acc, "machines", g, "use", "installed")
            world.ledger.produce(c.acc, "capital", g, "installed")
            T._loan(c.acc, max(spent + g * fac.price - own, 0.0), "fiber loan")     # the rest is built on credit
            c.capacity += (spent / rp + g) / p.capacity_cost * self.usage
            self.t["fiber"] += spent + g * fac.price
            self.t["routers"] += spent
            T.t["investment"] += spent + g * fac.price                 # the network counts as investment in GDP
            T.t["it_investment"] += spent
        for c in self.carriers():
            c.capacity *= 0.99
        self.capacity = sum(c.capacity for c in self.carriers())

    # ------------------------------------------------------------------ phase 7: results, valuations, exits, mood
    def finance(self, world, income) -> None:
        p, T = self.p, self.town
        if not self.active(world):
            return
        for cat in CATS:
            group = [s for s in self.alive() if s.cat == cat]
            for s, sh in zip(group, self.shares(group)):
                s.reach = 0.7 * s.reach + 0.3 * sh
        failures, news = 0, 0.0
        for s in self.alive():
            s.growth = 0.7 * s.growth + 0.3 * ((s.revenue / s.revenue_prev - 1) if s.revenue_prev > 1e-9 else 0.5)
            s.growth = float(np.clip(s.growth, -0.5, 1.0))
            s.profit = s.revenue - s.staff_cost * (1 + p.overhead)
            s.valuation = self.value_of(s)
            s.peak_value = max(s.peak_value, s.valuation)
            spare = T.bal(s.acc) - 4 * self.burn(s)
            if s.profit > 0 and spare > 0:
                self.distribute(0.5 * spare, income, s.item, s.acc)
        for s in self.alive():
            hopeless = (self.mood < 0 and s.status == "private" and world.tick - s.born >= p.triage_age
                        and s.revenue < 0.5 * s.staff_cost * (1 + p.overhead)
                        and self.rng.random() < p.plug * min(1.0, -self.mood))         # investors pull the plug
            if ((T.bal(s.acc) < 0.5 * self.burn(s) or s.L < 0.05 or hopeless) and s.profit <= 0
                    and world.tick - s.born > 2):
                if self.acquire(world, s, income) or self.rescue(world, s):
                    continue
                self.fail(world, s, income)
                failures += 1
                news += 1.0 if s.public_at > 0 else 0.2                 # a listed company's collapse makes the news
        for s in self.startups:
            if s.status in ("private", "public") or s.died == world.tick:
                s.trail.append((world.tick, round(s.L, 2), round(s.revenue, 1), round(s.valuation, 1), STATUS[s.status]))
        self.infra_results(world, income)
        same = [x for x in self.alive() if x.status == "public" and x.value_prev > 0]
        before = sum(x.value_prev for x in same)
        ret = sum(x.valuation for x in same) / before - 1 if before > 1e-9 else 0.0
        for x in self.listed_all():
            x.value_prev = x.valuation
        self.rates.append(T.rate)
        rise = T.rate - self.rates[-5] if len(self.rates) >= 5 else 0.0
        self.hh_hist.append(self.hh)
        buzz = self.hh_hist[-1] - self.hh_hist[-2] if len(self.hh_hist) >= 2 else 0.0
        buzz *= min(1.0, max(0.0, p.novelty * (1 - self.hh / p.hh_ceiling)))      # novelty fades as the web goes mainstream
        alive = max(sum(1.0 if x.status == "public" else 0.2 for x in self.alive()) + news, 1.0)
        idx = self.dotcom_index()
        self.peak = max(self.peak * p.peak_memory, idx)                        # the memory of the peak fades slowly
        drawdown = 1 - idx / self.peak if self.peak > 1e-9 else 0.0
        put = T.cut_now > 0 and not self.cutting                        # the 'put': a cut to steady markets lifts spirits once
        self.cutting = T.cut_now > 0
        crowd = min(1.0, sum(1 for x in self.alive() if x.status == "public") / p.crowd)   # a market needs a crowd
        self.terms = dict(momentum=crowd * p.mood_momentum * float(np.clip(ret, -0.5, 0.5)), buzz=p.mood_buzz * buzz,
                          failures=-crowd * p.mood_failures * news / alive, rates=-p.mood_rates * max(rise, 0.0),
                          put=p.mood_put * put, drawdown=-crowd * p.mood_drawdown * drawdown, bankrupt=-0.3 * self.t["bankruptcies"])
        self.mood = ((1 - p.mood_decay) * self.mood + crowd * p.mood_momentum * float(np.clip(ret, -0.5, 0.5))
                     + p.mood_buzz * buzz - crowd * p.mood_failures * news / alive - p.mood_rates * max(rise, 0.0)
                     + p.mood_put * put - crowd * p.mood_drawdown * drawdown - 0.3 * self.t["bankruptcies"])
        self.mood = float(np.clip(self.mood, -2.0, p.mood_max))
        self.t["failed"] = failures
        self.cum["failed"] += failures

    def infra_results(self, world, income):
        p, T, L = self.p, self.town, world.ledger
        for f in self.infra:
            K = L.balance(f.acc, "capital")
            dep = 0.025 * K
            if K > 0:
                L.remove(f.acc, "capital", dep, "depreciation", "wear")
            interest = (T.rate + T.loan_spread() + T.extra_spread.get(f.acc, 0.0)) / 4 * T.debt(f.acc)
            f.profit = f.revenue - f.costs - interest - dep * T.sec["factory"].price
            g = (f.revenue / f.revenue_prev - 1) if f.revenue_prev > 50 else 0.0
            f.growth = 0.8 * f.growth + 0.2 * float(np.clip(g, -0.5, 0.5))
            if f.kind == "router":
                f.expected = 0.6 * f.expected + 0.4 * f.orders
            spare = T.bal(f.acc) - 2 * (f.costs + interest)
            if spare > 0:                                  # carriers pay their fiber loans off slowly
                T.repay(f.acc, spare if f.kind == "router" else min(spare, p.amortization * T.debt(f.acc)))
            if f.kind == "carrier" and f.bankrupt < 0:
                debt = T.debt(f.acc)
                cover = (f.revenue - f.costs) / interest if interest > 1e-9 else 9.0
                # while markets are open a struggling carrier can refinance; once the mood sours, lenders will not
                # roll over debts larger than the business is worth, nor lend to one that cannot pay its interest
                worth = max(self.business_value(f), 8 * 4 * max(f.revenue - f.costs, 0.0))   # to a lender
                broke = debt > worth or cover < 1.0
                f.distress = f.distress + 1 if (broke and debt > 2 * 4 * max(f.revenue, 1e-9) and self.mood < -0.5) else 0
                if f.distress >= p.bankruptcy_quarters:
                    loss = p.write_off * debt
                    L.post(f.acc, BANK, "loans", loss, "transfer", "carrier bankruptcy")
                    T.t["defaults"] += loss
                    self.t["defaults"] += loss
                    self.cum["defaults"] += loss
                    self.t["bankruptcies"] += 1
                    self.cum["bankruptcies"] += 1
                    f.bankrupt = world.tick
            f.valuation = self.value_of_infra(f)
            f.trail.append((world.tick, round(f.L, 2), round(f.revenue, 1), round(f.valuation, 1),
                            2 if f.bankrupt >= 0 else 1, round(T.debt(f.acc), 1), round(f.capacity / max(self.usage, 1e-9), 2)))

    def value_of(self, s) -> float:
        p = self.p
        if s.status not in ("private", "public"):
            return 0.0
        rev = s.revenue * 4
        mult = min(p.multiple * math.exp(p.mood_multiple * self.mood) * (1 + 4 * max(s.growth, 0.0)), p.max_multiple)
        v = max(rev * mult, max(s.profit, 0.0) * 4 * 15) + max(self.town.bal(s.acc), 0.0)
        if rev <= 1e-9:
            v = max(v, s.raised * (1.5 if self.mood > 0 else 0.8))
        return v

    def value_of_infra(self, f) -> float:
        p, T = self.p, self.town
        if f.bankrupt >= 0:
            return 0.0                                     # old shareholders wiped out
        return max(self.business_value(f) - T.debt(f.acc) + max(T.bal(f.acc), 0.0), 0.0)

    def business_value(self, f) -> float:
        """What the business is worth to a buyer (before its debts), at today's mood."""
        p = self.p
        mult = min(p.multiple * math.exp(p.mood_multiple * self.mood) * (1 + 2 * min(max(f.growth, 0.0), 0.3)), p.max_infra_multiple)
        return max(f.revenue * 4 * mult, max(f.profit, 0.0) * 4 * 15)

    def acquire(self, world, s, income) -> bool:
        """A market leader with cash may buy a dying startup cheaply (staff and customers join it)."""
        p, T = self.p, self.town
        if self.rng.random() > p.acquisition_chance:
            return False
        leaders = [x for x in self.alive() if x.cat == s.cat and x is not s and x.revenue > 3 * max(s.revenue, 1e-9)
                   and x.reach > s.reach]
        if not leaders:
            return False
        buyer = max(leaders, key=lambda x: x.revenue)
        price = max(0.4 * s.valuation, (0.3 if self.mood > 0 else 0.1) * s.raised)     # fire sales in a bust
        if T.bal(buyer.acc) < 2 * price + self.burn(buyer):
            return False
        L = world.ledger
        holders = {a: L.balance(a, s.item) for a in [*HH, INS]}
        tot = sum(v for v in holders.values() if v > 0)
        for a, n in holders.items():
            if n > 0:
                x = T.pay(buyer.acc, a, price * n / tot, "acquisition", borrow=False)
                if a in income:
                    income[a] += x
        left = max(T.bal(s.acc), 0.0)
        if left > 1e-9:
            T.pay(s.acc, buyer.acc, left, "acquisition", borrow=False)
        K = L.balance(s.acc, "capital")
        if K > 0:
            L.remove(s.acc, "capital", K, "depreciation", "merged")
            L.produce(buyer.acc, "capital", K, "merged")
        buyer.L += s.L
        buyer.plan += s.L
        buyer.reach += s.reach
        s.L = 0.0
        s.status = "acquired"
        s.acquired_by = buyer.id
        s.died = world.tick
        s.valuation = 0.0
        self.cum["acquired"] += 1
        self.t["acquired"] += 1
        return True

    def rescue(self, world, s) -> bool:
        p, T = self.p, self.town
        if self.mood < 0 or s.growth < 0 or math.log(s.quality) + s.signal < 0:
            return False
        amount = p.round_quarters * self.burn(s)
        if T.bal("hh:owners") < 3 * amount:
            return False
        self.invest(s, amount, "rescue round", ("hh:owners",))
        self.t["vc"] += amount
        return True

    def distribute(self, amount, income, item, acc):
        L = self.world.ledger
        holders = {a: L.balance(a, item) for a in [*HH, INS]}
        tot = sum(v for v in holders.values() if v > 0)
        for a, n in holders.items():
            if n > 0 and tot > 0:
                x = self.town.pay(acc, a, amount * n / tot, "dividends", borrow=False)
                if a in income:
                    income[a] += x

    def fail(self, world, s, income):
        T, L = self.town, world.ledger
        left = max(T.bal(s.acc), 0.0)
        if left > 1e-9:
            self.distribute(left, income, s.item, s.acc)
        K = L.balance(s.acc, "capital")
        if K > 0:
            L.remove(s.acc, "capital", K, "depreciation", "servers scrapped")
        s.status = "failed"
        s.died = world.tick
        s.L = 0.0
        s.valuation = 0.0

    # ------------------------------------------------------------------ measures
    def listed(self):
        return [s for s in self.alive() if s.status == "public"] + [f for f in self.infra if f.bankrupt < 0]

    def listed_all(self):
        return self.alive() + self.infra

    def holdings_value(self, acc) -> float:
        L = self.world.ledger
        v = sum(L.balance(acc, s.item) / s.shares * s.valuation for s in self.alive() if s.shares > 0)
        v += sum(L.balance(acc, f.item) / f.shares * f.valuation for f in self.infra)
        return v

    def index(self) -> float:
        """All listed new-economy companies (startups, carriers, router maker): part of the town's market."""
        return sum(x.valuation for x in self.listed())

    def dotcom_index(self) -> float:
        return sum(s.valuation for s in self.alive() if s.status == "public")

    def network_index(self) -> float:
        return sum(f.valuation for f in self.infra if f.bankrupt < 0)

    def fair_value(self) -> float:
        return sum(max(x.profit, 0.0) * 4 / 0.08 for x in self.listed())

    def served(self) -> float:
        return min(self.traffic, self.capacity)

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.plan}

    def invariants(self):
        def shares_ok(world, ctx):
            L = world.ledger
            out = []
            for s in self.startups[-60:]:
                held = sum(L.balance(a, s.item) for a in [*HH, INS])
                if abs(held + L.balance(s.acc, s.item)) > 1e-6 * max(held, 1.0):
                    out.append(Violation("internet", f"{s.acc}: shares don't add up"))
            for s in self.alive():
                if self.town.bal(s.acc) < -1e-6:
                    out.append(Violation("internet", f"{s.acc} overdrawn (startups have no credit line)"))
            if self.capacity > 0 and self.served() > self.capacity * (1 + 1e-9):
                out.append(Violation("internet", "the network carried more traffic than its capacity"))
            return out
        return [Invariant("internet", shares_ok)]

    def metrics(self):
        def stat(k):
            return lambda w: self.t.get(k, 0.0)

        def top_share(w):
            out = {}
            for cat in CATS:
                rev = sorted([s.revenue for s in self.alive() if s.cat == cat], reverse=True)
                tot = sum(rev)
                out[cat] = rev[0] / tot if tot > 0 else 0.0
            return out

        return [
            Metric("adoption", lambda w: {"households": self.hh, "businesses": self.firms, "learned": self.realised},
                   "share online / productivity gains realised", "share"),
            Metric("online_shopping", stat("online"), "household spending at online shops", "$"),
            Metric("internet_access", stat("access"), "household spending on internet access", "$"),
            Metric("online_ads", stat("ads"), "business advertising paid to portals and search", "$"),
            Metric("vc_funding", stat("vc"), "venture money invested this quarter", "$"),
            Metric("seed_funding", stat("seed"), "seed money for new ventures this quarter", "$"),
            Metric("ipo_funding", stat("ipo"), "money raised at IPOs this quarter", "$"),
            Metric("startups", lambda w: {"alive": len(self.alive()), "public": sum(1 for s in self.alive() if s.status == "public"),
                                          "founded": self.cum["founded"], "failed": self.cum["failed"], "ipos": self.cum["ipos"],
                                          "acquired": self.cum["acquired"]}, "startups", "count"),
            Metric("founded_now", stat("founded"), "startups founded this quarter", "count"),
            Metric("failed_now", stat("failed"), "startups failed this quarter", "count"),
            Metric("net_index", lambda w: self.dotcom_index(), "market value of listed internet companies", "$"),
            Metric("network_index", lambda w: self.network_index(), "market value of the carriers and the router maker", "$"),
            Metric("net_private_value", lambda w: sum(s.valuation for s in self.alive() if s.status == "private"), "value of private startups", "$"),
            Metric("price_to_sales", lambda w: self.dotcom_index() / max(4 * sum(s.revenue for s in self.alive() if s.status == "public"), 1e-9),
                   "listed internet companies' value / annual revenue", "x"),
            Metric("mood", lambda w: self.mood, "new-economy mood", "x"),
            Metric("internet_jobs", lambda w: sum(s.L for s in self.alive()), "people working at startups", "people"),
            Metric("network_jobs", lambda w: sum(f.L for f in self.infra), "people at carriers and the router maker", "people"),
            Metric("internet_revenue", lambda w: sum(s.revenue for s in self.alive()), "startup revenue", "$"),
            Metric("leader_share", top_share, "largest firm's share of its market's revenue", "share"),
            Metric("servers", stat("servers"), "servers bought by startups", "$"),
            Metric("router_sales", lambda w: self.router().revenue if self.router() else 0.0, "router maker revenue", "$"),
            Metric("startup_marketing", stat("marketing"), "marketing bought by startups", "$"),
            Metric("fiber_investment", stat("fiber"), "network capacity built this quarter", "$"),
            Metric("carrier_debt", lambda w: sum(self.town.debt(c.acc) for c in self.carriers()), "carriers' bank loans", "$"),
            Metric("carrier_revenue", lambda w: sum(c.revenue for c in self.carriers()), "carriers' bandwidth revenue", "$"),
            Metric("carrier_bankruptcies", lambda w: self.cum["bankruptcies"], "carriers bankrupt so far", "count"),
            Metric("network", lambda w: {"traffic": self.traffic, "capacity": self.capacity,
                                         "utilization": self.served() / self.capacity if self.capacity > 0 else 0.0,
                                         "believed_growth": self.belief, "price": self.bw_price},
                   "data traffic, capacity, use", "units"),
            Metric("carrier_defaults", stat("defaults"), "carrier loans written off", "$"),
            Metric("episode", lambda w: {"warning": 1.0, "scare": 2.0, "y2k": 3.0, "hold": 4.0}.get(self.episode, 0.0), "scripted episode", "code"),
        ]
