"""Stage 7: risk sharing for farmers — harvest futures and crop insurance.

Futures: every tick each farm sells `futures_hedge` x its expected sales forward, for
`futures_maturity` ticks ahead, at today's wheat price. At maturity the contract is settled
in cash against the wheat price then: the farm receives (agreed - spot) x quantity, or pays
it if the price rose. The other side is taken by speculating households (per head).
Futures fix the *price*; they do nothing about a small *quantity*.

Insurance: farms insure `insurance_cover` of their normal harvest value. When the harvest
multiplier m < 1, the insurer pays cover x (1 - m) x normal harvest x price. The premium
is (1 + loading) x the insurer's running estimate of the expected loss. The insurer is a
separate account owned by households: it starts with capital paid in by them, pays its
excess capital back as dividends, and if a disaster exhausts it, pays only what it has
(correlated risk: every farm claims at once).

Hedging gains and losses, premiums and payouts count in the farm's revenue (and profit).
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.invariants import Invariant, Violation
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

INSURER = "insurer"
FARM = 0
CATS = ("futures", "insurance_premium", "insurance_payout", "insurer_capital", "insurer_dividends")


class RiskSharingParams(LayerParams):
    futures_hedge: float = Field(0.0, ge=0, le=1, description="share of expected wheat sales sold forward each tick")
    futures_maturity: int = Field(4, ge=1, description="ticks until a futures contract is settled")
    insurance_cover: float = Field(0.0, ge=0, le=1, description="share of the normal harvest value insured")
    insurance_loading: float = Field(0.1, ge=0, description="premium = (1 + loading) x expected loss")
    prior_loss_rate: float = Field(0.05, ge=0, description="insurer's initial estimate of the loss per unit insured")
    learning: float = Field(0.02, gt=0, le=1, description="how fast the insurer updates its loss estimate")
    insurer_capital: float = Field(30.0, ge=0, description="starting capital, in ticks of expected premiums")
    capital_target: float = Field(60.0, ge=0, description="capital beyond this many ticks of premiums is paid out")


class RiskSharing(Layer):
    name = "risk_sharing"
    requires = ["agents"]
    Params = RiskSharingParams
    p: RiskSharingParams

    def setup(self, world) -> None:
        L, p = world.ledger, self.p
        st = world.state["economy"]
        self.m = st["market_layer"]
        st["extra_categories"] = list(st.get("extra_categories", [])) + list(CATS)
        self.contracts: list[tuple[int, str, float, float]] = []      # (maturity tick, farm account, qty, price)
        self.loss_rate = p.prior_loss_rate
        self.t = self.zero()
        self.normal_output = {}
        if p.insurance_cover > 0:
            L.open_account(INSURER)
            st.setdefault("money_holders", []).append(INSURER)
            capital = p.insurer_capital * self.expected_premiums(world)
            self.pay(world, INSURER, "households", capital, "insurer_capital")

    @staticmethod
    def zero():
        return dict(futures_to_farms=0.0, premiums=0.0, payouts=0.0, unpaid_claims=0.0, insurer_dividends=0.0,
                    hedge_pnl=0.0)

    # ------------------------------------------------------------------ money moves
    def pay(self, world, to: str, frm: str, amount: float, category: str) -> float:
        """Move money; households pay out of their cash in proportion, and receive per head."""
        m, L = self.m, world.ledger
        mi = m.money_item
        if amount <= 1e-12:
            return 0.0
        if frm == "households":
            have = float(m.cash.sum())
            amount = min(amount, max(have, 0.0))
            if amount <= 1e-12:
                return 0.0
            m.cash -= amount * m.cash / have
        else:
            amount = min(amount, max(L.balance(frm, mi), 0.0))
            if amount <= 1e-12:
                return 0.0
        L.post(to, frm, mi, amount, "transfer", category)
        if to == "households":
            m.cash += amount * m.size / m.size.sum()
            m.income += amount * m.size / m.size.sum()
        return amount

    def farms(self):
        return [f for f in self.m.firms if f.sector == FARM]

    def wheat_price(self) -> float:
        fs = self.farms()
        return float(np.mean([f.price for f in fs])) if fs else 0.0

    def expected_premiums(self, world) -> float:
        p = self.p
        value = sum(f.expected * f.price for f in self.farms())
        return (1 + p.insurance_loading) * self.loss_rate * p.insurance_cover * value

    # ------------------------------------------------------------------ phase 2: after the harvest
    def insure(self, world) -> None:
        p = self.p
        if p.insurance_cover <= 0:
            return
        mult = world.vars.get("harvest_multiplier", 1.0)
        loss = max(1.0 - mult, 0.0)
        claims, insured = [], 0.0
        for f in self.farms():
            normal = f.output / mult if mult > 1e-9 else f.expected
            value = p.insurance_cover * normal * f.price
            insured += value
            premium = (1 + p.insurance_loading) * self.loss_rate * value
            got = self.pay(world, INSURER, f.account, premium, "insurance_premium")
            f.revenue -= got
            self.t["premiums"] += got
            claims.append((f, loss * value))
        total = sum(c for _, c in claims)
        have = max(world.ledger.balance(INSURER, self.m.money_item), 0.0)
        scale = min(1.0, have / total) if total > 1e-12 else 0.0
        for f, c in claims:
            got = self.pay(world, f.account, INSURER, c * scale, "insurance_payout")
            f.revenue += got
            self.t["payouts"] += got
        self.t["unpaid_claims"] = total - self.t["payouts"]
        if insured > 1e-12:
            self.loss_rate += p.learning * (total / insured - self.loss_rate)

    # ------------------------------------------------------------------ phase 6: futures and insurer dividends
    def hedge(self, world) -> None:
        p = self.p
        by_acc = {f.account: f for f in self.farms()}
        keep = []
        for (due, acc, qty, agreed) in self.contracts:
            if due > world.tick:
                keep.append((due, acc, qty, agreed))
                continue
            f = by_acc.get(acc)
            if f is None:
                continue                # the farm closed: the contract lapses
            gain = (agreed - f.price) * qty
            if gain > 0:
                got = self.pay(world, acc, "households", gain, "futures")
            else:
                got = -self.pay(world, "households", acc, -gain, "futures")
            f.revenue += got
            self.t["hedge_pnl"] += got
        self.contracts = keep
        if p.futures_hedge > 0:
            for f in self.farms():
                self.contracts.append((world.tick + p.futures_maturity, f.account, p.futures_hedge * f.expected, f.price))
        if p.insurance_cover > 0:
            excess = world.ledger.balance(INSURER, self.m.money_item) - p.capital_target * self.expected_premiums(world)
            if excess > 1e-9:
                self.t["insurer_dividends"] = self.pay(world, "households", INSURER, 0.2 * excess, "insurer_dividends")
        world.vars["risk_stats"] = self.t
        world.vars["farm_income"] = sum(f.revenue - f.op_costs - getattr(f, "interest_paid", 0.0) for f in self.farms())
        self.t = self.zero()

    def hooks(self):
        return {Phase.PRODUCTION: self.insure, Phase.INVESTMENT: self.hedge}

    def invariants(self):
        def insurer_solvent(world, ctx):
            if self.p.insurance_cover > 0 and world.ledger.balance(INSURER, self.m.money_item) < -1e-9:
                return [Violation("insurer", "the insurer paid out money it does not have")]
            return []
        return [Invariant("insurer", insurer_solvent)]

    def metrics(self):
        def stat(k):
            return lambda w: w.vars.get("risk_stats", {}).get(k, 0.0)
        out = [Metric("farm_income", lambda w: w.vars.get("farm_income", 0.0), "farms' profit incl. hedging and insurance", "money"),
               Metric("wheat_price", lambda w: self.wheat_price(), "average farm-gate wheat price", "money"),
               Metric("hedge_pnl", stat("hedge_pnl"), "farms' gains on futures this tick", "money"),
               Metric("open_futures", lambda w: sum(c[2] for c in self.contracts), "wheat sold forward, not yet settled", "units")]
        if self.p.insurance_cover > 0:
            out += [Metric("insurance_payouts", stat("payouts"), "insurance paid to farms", "money"),
                    Metric("insurance_premiums", stat("premiums"), "premiums paid by farms", "money"),
                    Metric("unpaid_claims", stat("unpaid_claims"), "claims the insurer could not pay", "money"),
                    Metric("insurer_capital", lambda w: w.ledger.balance(INSURER, self.m.money_item), "insurer's money", "money")]
        return out
