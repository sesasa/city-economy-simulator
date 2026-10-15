"""Stage 7: a stock market with fundamental traders, trend followers and margin loans.

All firms' equity is represented by one index of shares (split once at the listing so that,
at fair value, shares are `base_share` of households' wealth); holders receive every
firm's dividends in proportion to their shares (replacing founder ownership). Households
(agents) are of three kinds:
* passive: hold, with a little random trading (noise);
* fundamentalists: hold more when the price is below the fundamental value
  V = (average dividends per share) / (policy rate + equity premium), less when above;
* trend followers: hold more when the price has been rising (less so the further it is from
  value), and may borrow from the bank to do it (margin loans) up to
  (1 - margin_requirement) x the value of their shares.

Each tick (Phase 6, after dividends): margin interest is paid; anyone whose debt exceeds
(1 - margin) x their shares' value must sell (a margin call: a fire sale); desired positions
are a share of wealth (cash + shares - debt); orders are matched with rationing (forced
sales first) at the current price; then the price moves with the order imbalance:
P <- P x exp(impact x (buys - sells) / (buys + sells + depth x shares)). Leveraged buying is
limited by the bank's capital: orders the bank cannot fund are never placed. A borrower left
with debt and no shares defaults on what cash cannot cover; the bank writes it off.

Shares are a ledger claim ("shares": +shares for households, -shares for the firms' equity
account). Trades between households move money inside the households account, so only
margin loans, interest, repayments and defaults appear as ledger postings.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.invariants import Invariant, Violation, close
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

ISSUER = "equity"
BANK = "bank:0"
PASSIVE, FUNDAMENTAL, TREND = 0, 1, 2


class StockMarketParams(LayerParams):
    shares: float = Field(1000.0, gt=0, description="index shares outstanding")
    fundamentalists: float = Field(0.3, ge=0, le=1, description="share of households trading on value")
    trend_followers: float = Field(0.3, ge=0, le=1, description="share of households trading on trends")
    start_tick: int = Field(20, ge=1, description="trading starts here; before, the price tracks the fundamental value")
    equity_premium: float = Field(0.05, ge=0, description="return over the policy rate shareholders require, per tick")
    dividend_memory: float = Field(0.05, gt=0, le=1, description="weight of the newest tick in average dividends")
    base_share: float = Field(0.5, ge=0, le=1, description="share of wealth held in shares at fair value")
    value_strength: float = Field(1.0, ge=0, description="fundamentalists: change in share held per log mispricing")
    trend_strength: float = Field(100.0, ge=0, description="trend followers: change in share held per unit of trend")
    doubt: float = Field(0.5, gt=0, description="log mispricing at which trend followers' conviction halves")
    trend_memory: float = Field(0.2, gt=0, le=1, description="weight of the newest return in the trend")
    speed: float = Field(0.2, gt=0, le=1, description="share of the gap to the desired position traded per tick")
    impact: float = Field(0.1, gt=0, description="largest log price move per tick (all orders on one side)")
    depth: float = Field(0.01, gt=0, description="market depth: order flow (share of all shares) that moves the price "
                                                 "by half of `impact`")
    noise: float = Field(0.02, ge=0, description="passive holders' random trading, sd as a share of holdings")
    margin_requirement: float = Field(1.0, gt=0, le=1, description="owned share of a position (1: no borrowing)")
    cash_reserve: float = Field(1.0, ge=0, description="ticks of income kept out of the market")
    fallback_rate: float = Field(0.003, ge=0, description="discount rate when there is no central bank")


class StockMarket(Layer):
    name = "stock_market"
    requires = ["agents", "bank"]
    provides = ["stock_market"]
    Params = StockMarketParams
    p: StockMarketParams

    def setup(self, world) -> None:
        L, p = world.ledger, self.p
        st = world.state["economy"]
        self.m = st["market_layer"]
        self.bank = st["bank_layer"]
        m = self.m
        n = len(m.size)
        rng = np.random.default_rng([world.seed, 7003])
        self.rng = rng
        u = rng.random(n)
        self.kind = np.where(u < p.fundamentalists, FUNDAMENTAL,
                             np.where(u < p.fundamentalists + p.trend_followers, TREND, PASSIVE)).astype(int)
        w = np.maximum(m.cash, 0.0) + 1e-9 * m.size       # listed shares are spread in proportion to money held
        self.n = p.shares                  # shares outstanding (rescaled once at the listing)
        self.held = self.n * w / w.sum()
        self.debt = np.zeros(n)
        L.register_item("shares", "claim")
        L.open_account(ISSUER)
        L.post("households", ISSUER, "shares", p.shares, "transfer", "listing")
        st["dividend_weights"] = lambda: self.held
        st.setdefault("agent_arrays", []).extend([(self, "held", "inherit"), (self, "debt", "inherit"), (self, "kind", PASSIVE)])
        st["extra_categories"] = list(st.get("extra_categories", [])) + ["margin_loan", "margin_repayment", "margin_interest"]
        self.div_avg = None
        self.value = 0.0
        self.price = None
        self.trend = 0.0
        self.t = self.zero()

    @staticmethod
    def zero():
        return dict(volume=0.0, forced=0.0, borrowed=0.0, repaid=0.0, interest=0.0, defaults=0.0, buys=0.0, sells=0.0)

    # ------------------------------------------------------------------ helpers
    def discount(self, world) -> float:
        pr = world.state["economy"].get("policy_rate")
        return (self.p.fallback_rate if pr is None else max(pr, 0.0)) + self.p.equity_premium

    def margin_loan(self, world, i: np.ndarray, amounts: np.ndarray) -> None:
        x = float(amounts.sum())
        if x <= 1e-12:
            return
        L = world.ledger
        L.post(BANK, "households", "loans", x, "transfer", "margin_loan")
        L.post("households", BANK, self.m.money_item, x, "transfer", "margin_loan")
        np.add.at(self.debt, i, amounts)
        np.add.at(self.m.cash, i, amounts)
        self.bank.tick_stats["lent"] += x
        self.t["borrowed"] += x

    def repay(self, world, amounts: np.ndarray) -> None:
        amounts = np.minimum(np.maximum(amounts, 0.0), self.debt)
        x = float(amounts.sum())
        if x <= 1e-12:
            return
        L = world.ledger
        L.post(BANK, "households", self.m.money_item, x, "transfer", "margin_repayment")
        L.post("households", BANK, "loans", x, "transfer", "margin_repayment")
        self.debt -= amounts
        self.m.cash -= amounts
        self.bank.tick_stats["repaid"] += x
        self.t["repaid"] += x

    # ------------------------------------------------------------------ phase 6
    def trade(self, world) -> None:
        p, m, L = self.p, self.m, world.ledger
        dividends = sum(getattr(f, "dividends", 0.0) for f in m.firms)
        self.div_avg = dividends if self.div_avg is None else (1 - p.dividend_memory) * self.div_avg + p.dividend_memory * dividends
        self.value = self.div_avg / self.n / self.discount(world)
        if self.price is None or world.tick < p.start_tick:
            self.price = max(self.value, 1e-9)
            return
        if world.tick == p.start_tick:
            self.list_shares(world)
            return
        P = self.price
        cash = m.cash
        # 1. margin interest (unpaid interest is added to the debt)
        rate = self.bank.rate(world)
        due = rate * self.debt
        paid = np.minimum(due, np.maximum(cash, 0.0))
        if paid.sum() > 1e-12:
            L.post(BANK, "households", m.money_item, float(paid.sum()), "transfer", "margin_interest")
            m.cash -= paid
            self.bank.tick_stats["interest_in"] += float(paid.sum())
            self.t["interest"] += float(paid.sum())
        unpaid = due - paid
        if unpaid.sum() > 1e-12:            # unpaid interest is lent to the borrower (it compounds)
            L.post(BANK, "households", "loans", float(unpaid.sum()), "transfer", "margin_interest")
        self.debt += unpaid
        # 2. margin calls: sell enough to bring debt back within (1 - margin) x value
        mr = p.margin_requirement
        excess = self.debt - (1 - mr) * self.held * P
        forced = np.where(excess > 1e-12, np.minimum(self.held, excess / (mr * P)), 0.0)
        # 3. desired positions
        spare = np.maximum(m.cash - p.cash_reserve * np.maximum(m.income, 0.0), 0.0)
        wealth = np.maximum(m.cash + self.held * P - self.debt, 0.0)
        theta = np.full(len(cash), p.base_share)
        gap = np.log(max(self.value, 1e-12) / P)
        theta = np.where(self.kind == FUNDAMENTAL, np.clip(p.base_share + p.value_strength * gap, 0.0, 1.0), theta)
        conviction = 1.0 / (1.0 + (gap / p.doubt) ** 2)      # trend followers trust the trend less far from value
        theta = np.where(self.kind == TREND,
                         np.clip(p.base_share + p.trend_strength * conviction * self.trend, 0.0, 1.0 / mr), theta)
        want = p.speed * (theta * wealth / P - self.held)
        noise = self.held * p.noise * self.rng.standard_normal(len(cash))
        want = np.where(self.kind == PASSIVE, noise, want)
        # buyers are limited by spare cash plus (trend followers) what they may borrow
        can_borrow = np.where(self.kind == TREND, np.maximum((spare + (1 - mr) * self.held * P - self.debt) / (mr * P), 0.0), spare / P)
        buy = np.where(forced > 0, 0.0, np.clip(want, 0.0, can_borrow))
        # leveraged buying is limited by what the bank can lend (capital requirement, freezes)
        kappa = self.bank.p.capital_requirement
        room = max(self.bank.equity(world) / kappa - self.bank.risk_assets(world), 0.0) if kappa > 0 else np.inf
        if self.bank.frozen > 0:
            room = 0.0
        need = np.maximum(buy * P - spare, 0.0)
        if need.sum() > room + 1e-12:
            k = room / need.sum()
            buy = np.where(need > 0, (spare + need * k) / P, buy)
        sell = np.maximum(np.minimum(-np.minimum(want, 0.0), self.held - forced), 0.0) + forced
        B, S = float(buy.sum()), float(sell.sum())
        X = min(B, S)
        if X > 1e-12:
            F = float(forced.sum())
            f_forced = min(1.0, X / F) if F > 0 else 0.0
            vol = sell - forced
            V_ = float(vol.sum())
            f_vol = min(1.0, (X - f_forced * F) / V_) if V_ > 1e-12 else 0.0
            sold = forced * f_forced + vol * f_vol
            bought = buy * (X / B)
            # money: buyers pay from spare cash, then borrow; sellers are paid and repay their debt
            cost = bought * P
            from_cash = np.minimum(cost, spare)
            borrow = cost - from_cash
            i = np.nonzero(borrow > 1e-12)[0]
            self.margin_loan(world, i, borrow[i])
            m.cash -= cost
            m.cash += sold * P
            self.held += bought - sold
            self.repay(world, np.minimum(sold * P, self.debt))
            self.t["volume"] += float(bought.sum())
            self.t["forced"] += float((forced * f_forced).sum())
        self.t["buys"], self.t["sells"] = B, S
        # 4. a borrower with no shares left pays what cash allows; the rest is written off
        broke = (self.held <= 1e-9) & (self.debt > 1e-12)
        if broke.any():
            self.repay(world, np.where(broke, np.minimum(self.debt, np.maximum(m.cash, 0.0)), 0.0))
            loss = float(np.where(broke, self.debt, 0.0).sum())
            if loss > 1e-12:
                L.post("households", BANK, "loans", loss, "transfer", "margin_default")
                self.debt = np.where(broke, 0.0, self.debt)
                self.bank.tick_stats["defaults"] += loss
                self.bank.tick_stats["written_off"] += loss
                self.t["defaults"] += loss
        self.held = np.maximum(self.held, 0.0)
        # 5. the price moves with the order imbalance
        new = P * float(np.exp(p.impact * (B - S) / (B + S + p.depth * self.n)))
        self.trend = (1 - p.trend_memory) * self.trend + p.trend_memory * np.log(new / P)
        self.price = new

    def list_shares(self, world) -> None:
        """At the listing the index is split so that, at fair value, shares are `base_share` of
        households' wealth: the market can then be priced near value with the money that exists."""
        p, L = self.p, world.ledger
        money = float(np.maximum(self.m.cash, 0.0).sum())
        want = p.base_share / (1 - p.base_share) * money          # market value at fair price
        k = want / max(self.value * self.n, 1e-12)
        extra = (k - 1) * self.n
        if extra > 0:
            L.post("households", ISSUER, "shares", extra, "transfer", "split")
        elif extra < 0:
            L.post(ISSUER, "households", "shares", -extra, "transfer", "split")
        self.held *= k
        self.n *= k
        self.value /= k
        self.price = self.value

    def end_of_tick(self, world) -> None:
        world.vars["stock_stats"] = self.t
        self.t = self.zero()

    def hooks(self):
        return {Phase.INVESTMENT: self.trade, Phase.DEMOGRAPHY: self.end_of_tick}

    # ------------------------------------------------------------------ invariants
    def invariants(self):
        def holdings(world, ctx):
            L, out = world.ledger, []
            if not close(float(self.held.sum()), L.balance("households", "shares"), self.n):
                out.append(Violation("stock_market", f"agent shares {self.held.sum()} != ledger {L.balance('households', 'shares')}"))
            if (self.held < -1e-9).any() or (self.debt < -1e-9).any():
                out.append(Violation("stock_market", "negative shares or margin debt (no short selling)"))
            bank_loans_to_households = -L.balance("households", "loans")
            if not close(float(self.debt.sum()), bank_loans_to_households, max(bank_loans_to_households, 1.0)):
                out.append(Violation("stock_market", f"margin debt {self.debt.sum()} != household loans {bank_loans_to_households}"))
            return out

        return [Invariant("stock_market", holdings)]

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        def stat(k):
            return lambda w: w.vars.get("stock_stats", {}).get(k, 0.0)

        def lev(w):
            t = self.kind == TREND
            eq = (self.held * (self.price or 0.0) - self.debt)[t].sum()
            return float((self.held[t] * (self.price or 0.0)).sum() / eq) if eq > 1e-9 else 0.0

        return [
            Metric("share_price", lambda w: self.price or 0.0, "index share price", "money"),
            Metric("fundamental_value", lambda w: self.value, "average dividends per share / (policy rate + premium)", "money"),
            Metric("price_to_value", lambda w: (self.price or 0.0) / self.value if self.value > 1e-12 else 1.0,
                   "share price / fundamental value", "ratio"),
            Metric("margin_debt", lambda w: float(self.debt.sum()), "bank loans to share buyers", "money"),
            Metric("trend_leverage", lev, "trend followers' shares / their own equity", "ratio"),
            Metric("trading_volume", stat("volume"), "shares traded", "shares"),
            Metric("forced_sales", stat("forced"), "shares sold in margin calls", "shares"),
            Metric("margin_defaults", stat("defaults"), "margin debt written off", "money"),
            Metric("market_cap_share", lambda w: (self.price or 0.0) * self.n / max(self.bank.money_supply(w), 1e-9),
                   "stock market value / money supply", "ratio"),
        ]
