"""Stage 4: firms enter where margins are fat and exit after sustained losses.

Entry: when a sector's average profit margin (profit / revenue, smoothed) is above
`entry_margin`, each tick there is a chance a household with enough savings founds a
new firm, paying `startup_ticks` of a starting crew's wages in as equity. Factories
need a 1,000 labor-tick plant, so factory entry is off by default (a barrier to entry).
Exit: a firm that has lost money for `exit_patience` ticks in a row (and is short of cash)
closes if its sector has another firm: its debts default, and its plant, stocks, workers
and customers pass to the sector's largest firm (an acquisition at zero price); its
remaining cash goes to its owners.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase
from ..s2_money_prices.market import FACTORY
from .agent_market import SERVICES


class EntryExitParams(LayerParams):
    entry_margin: float = Field(0.2, description="profit/revenue above which firms want to enter")
    entry_chance: float = Field(0.05, ge=0, le=1, description="chance per tick that an entrant appears, when attractive")
    max_firms: int = Field(20, ge=1, description="most firms per sector")
    startup_ticks: float = Field(4.0, ge=0, description="founder's equity, in ticks of a starting crew's wages")
    exit_patience: int = Field(12, ge=1, description="consecutive loss-making ticks before a firm closes")
    exit_cash_ticks: float = Field(1.0, ge=0, description="...and only once its cash is below this many ticks of costs")
    allow_factory_entry: bool = False


class EntryExit(Layer):
    name = "entry_exit"
    requires = ["agents"]
    Params = EntryExitParams
    p: EntryExitParams

    def setup(self, world) -> None:
        self.m = world.state["economy"]["market_layer"]
        self.margin = {}
        self.log = []

    def step(self, world) -> None:
        m, p = self.m, self.p
        rng = m.rng
        for s in range(m.n_sec):
            firms = m.of(s)
            rev = sum(f.revenue for f in firms)
            prof = sum(f.last_profit for f in firms)
            now = prof / rev if rev > 0 else (1.0 if s == SERVICES and not firms else 0.0)
            self.margin[s] = 0.8 * self.margin.get(s, now) + 0.2 * now
        # exits
        public = getattr(m, "public_sectors", ())          # e.g. research (Stage 9): no entry or exit
        for f in list(m.firms):
            if f.sector in public:
                continue
            broke = world.ledger.balance(f.account, m.money_item) < p.exit_cash_ticks * max(f.op_costs_prev, 1e-9)
            if len(m.of(f.sector)) > 1 and getattr(f, "loss_streak", 0) >= p.exit_patience and broke:
                self.close(world, f)
        # entries
        for s in range(m.n_sec):
            if (s == FACTORY and not p.allow_factory_entry) or s in public:
                continue
            if len(m.of(s)) >= p.max_firms or self.margin.get(s, 0.0) <= p.entry_margin or rng.random() >= p.entry_chance:
                continue
            self.open(world, s)

    def open(self, world, s: int) -> None:
        m, p = self.m, self.p
        firms = m.of(s)
        wage = float(np.mean([f.wage for f in firms])) if firms else float(np.mean([f.wage for f in m.firms]))
        if s == SERVICES:
            sv = m.services
            crew = sv.fixed_crew + sv.labor_per_unit * 2
        else:
            crew = max(2.0, sum(f.hired for f in firms) / max(len(firms), 1) * 0.25)
        stake = p.startup_ticks * wage * crew
        rich = np.where(m.cash >= stake)[0]
        if not len(rich):
            return
        owner = int(m.rng.choice(rich))
        if s == SERVICES and not firms:
            f = world.state["economy"]["services_layer"].open_firm(world, m, owner=owner)
        else:
            price = float(np.mean([g.price for g in firms]))
            expected = sum(g.expected for g in firms) / (len(firms) + 1)    # hopes for a fair share
            f = m.new_firm(world, s, price, wage, expected, owner=owner)
        world.ledger.post(f.account, "households", m.money_item, stake, "transfer", "equity")
        m.cash[owner] -= stake
        m.entries += 1
        self.log.append((world.tick, "enter", m.sector_names[s], f.account))

    def close(self, world, f) -> None:
        m, L = self.m, world.ledger
        bank = world.state["economy"].get("bank_layer")
        if bank is not None and bank.debt(world, f) > 1e-9:
            bank.default(world, f, bank.debt(world, f))
        rivals = [g for g in m.of(f.sector) if g is not f]
        buyer = max(rivals, key=lambda g: g.revenue)
        for item in ("raw", "final", "factory", "factory_wip", "land_improvement", "tools_processing", "tools_distribution"):
            x = L.balance(f.account, item)
            if x > 1e-12:
                L.transfer(f.account, buyer.account, item, x, "acquisition")
        buyer.conditions = buyer.conditions + f.conditions
        f.conditions = []
        # the acquirer takes over the plant with its workforce and its customers
        moved = m.employer == f.id
        m.employer[moved] = buyer.id
        buyer.hired += float(m.size[moved].sum())
        buyer.expected += f.expected
        cash = L.balance(f.account, m.money_item)
        if cash > 1e-12:
            m.pay_households(world, f, cash, "liquidation", None)
        m.firms.remove(f)
        m.exits += 1
        self.log.append((world.tick, "exit", m.sector_names[f.sector], f.account))

    def hooks(self):
        return {Phase.DEMOGRAPHY: self.step}

    def metrics(self):
        return [Metric("entry_margin", lambda w: {self.m.sector_names[s]: v for s, v in self.margin.items()},
                       "smoothed profit / revenue by sector", "share")]
