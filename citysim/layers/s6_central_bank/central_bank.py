"""Stage 6: a central bank and monetary policy.

The central bank ("cb" account) sets the policy rate. The commercial bank lends at the
policy rate plus a spread, government bonds pay the policy rate, and households save
more (spend less of their income and wealth) when the rate is above its neutral level:
spending multiplier = 1 - rate_sensitivity x (rate - neutral_rate), within multiplier_bounds.
Every rule keeps the rate at neutral for the first `start_tick` ticks while prices settle.

Rules for the rate (`rule`):
* "taylor": rate = neutral + phi_pi x (inflation - target) - phi_u x (unemployment - u_star),
  floored at `lower_bound` (the zero lower bound), with optional smoothing.
  Inflation is a smoothed per-tick change in a consumer price index (bread and services).
* "fixed_rate": the rate stays at `neutral_rate`.
* "money_growth": the money supply should follow the path M0 x (1 + money_growth)^t (a fixed
  money stock when money_growth = 0). Above the path the rate rises by phi_m per log point
  (dearer credit, more saving); below it, the central bank pays new money to every household
  per head until the path is reached.

Reserves: with reserve_ratio < 1 the bank keeps only that share of its deposits in cash and
holds the rest as central bank term bills (an illiquid but safe asset: it counts as capital
but cannot pay depositors in a run). Lender of last resort: in a bank run, a solvent bank borrows new cash from the central
bank ("cb_loans", kind claim) at the policy rate plus a penalty, and repays it once the run
is over. Bailouts: an insolvent bank is recapitalised with newly created cash instead of
bailing in depositors. If the bank expects bailouts (`moral_hazard` > 0), it worries less
about defaults when loosening its lending standards, and depositors do not run.

DemandShocks is its own layer: scheduled multipliers on household spending
(a sudden wish to save), the demand-side counterpart of HarvestShocks.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...core.invariants import Invariant, Violation
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

CB = "cb"
BANK = "bank:0"


class CentralBankParams(LayerParams):
    rule: Literal["taylor", "fixed_rate", "money_growth"] = "taylor"
    neutral_rate: float = Field(0.003, description="policy rate per tick when inflation and unemployment are on target")
    inflation_target: float = Field(0.0, description="target inflation per tick")
    phi_pi: float = Field(1.5, ge=0, description="rate rise per unit of inflation above target")
    phi_u: float = Field(0.05, ge=0, description="rate cut per unit of unemployment above u_star")
    u_star: float = Field(0.04, ge=0, le=1, description="the central bank's estimate of normal unemployment")
    lower_bound: float = Field(0.0, description="the rate cannot go below this (zero lower bound)")
    start_tick: int = Field(50, ge=0, description="the rule starts here (the rate stays neutral while prices settle)")
    multiplier_bounds: tuple[float, float] = Field((0.5, 1.5), description="limits of the interest-rate effect on spending")
    phi_m: float = Field(0.1, ge=0, description="money_growth rule: rate rise per log deviation of money above its path")
    smoothing: float = Field(0.5, ge=0, lt=1, description="weight on last tick's rate (gradual moves)")
    inflation_smoothing: float = Field(0.2, gt=0, le=1, description="weight of the newest tick in measured inflation")
    rate_sensitivity: float = Field(15.0, ge=0, description="households' spending falls by this x (rate - neutral)")
    money_growth: float = Field(0.0, description="money_growth rule: allowed growth of the money supply per tick")
    reserve_ratio: float = Field(1.0, ge=0, le=1, description="cash the bank keeps per unit of deposits; the rest is "
                                                              "placed at the central bank in term bills it cannot use in a run")
    lender_of_last_resort: bool = Field(True, description="lend cash to a solvent bank facing a run")
    lolr_penalty: float = Field(0.005, ge=0, description="emergency loans cost the policy rate plus this")
    lolr_repay: float = Field(0.2, ge=0, le=1, description="share of emergency loans repaid per calm tick")
    bailouts: bool = Field(False, description="recapitalise an insolvent bank with new money")
    moral_hazard: float = Field(0.0, ge=0, le=1, description="how much a bank that expects bailouts discounts "
                                                             "defaults when loosening its standards")
    services_weight: float = Field(0.3, ge=0, le=1, description="weight of services in the price index")


class CentralBank(Layer):
    name = "central_bank"
    requires = ["bank"]
    provides = ["central_bank"]
    Params = CentralBankParams
    p: CentralBankParams

    def setup(self, world) -> None:
        L, p = world.ledger, self.p
        st = world.state["economy"]
        self.m = st["market_layer"]
        self.bank = st["bank_layer"]
        L.open_account(CB)
        L.register_item("cb_loans", "claim")
        L.register_item("cb_bills", "claim")
        # fractional reserves: the bank swaps cash beyond reserve_ratio x deposits for term bills (no interest)
        deposits = -L.balance(BANK, "deposits")
        excess = L.balance(BANK, "cash") - p.reserve_ratio * deposits
        if excess > 1e-9:
            L.post("ext:destruction", BANK, "cash", excess, "destruction", "cb_bills")
            L.post(BANK, CB, "cb_bills", excess, "transfer", "cb_bills")
        st["policy_rate"] = p.neutral_rate
        st["hurdle_rate"] = self.bank.rate(world) + self.bank.p.amortization + self.bank.p.hurdle_margin
        st["central_bank"] = self
        st.setdefault("bank_obligations", []).append(self.collect)
        if p.lender_of_last_resort:
            st["lender_of_last_resort"] = self.emergency_loan
        if p.bailouts:
            st["bailout"] = self.bailout
            if p.moral_hazard > 0:             # depositors trust the guarantee; the bank fears losses less
                self.bank.p = self.bank.p.model_copy(update=dict(
                    run_threshold=-np.inf,
                    panic_sensitivity=self.bank.p.panic_sensitivity * (1 - p.moral_hazard)))
        self.rate = p.neutral_rate
        self.m0 = self.bank.money_supply(world)
        self.cpi_prev = None
        self.inflation = 0.0
        self.unemployment = p.u_star
        self.bailout_cost = 0.0
        self.t = self.zero()

    @staticmethod
    def zero():
        return dict(lolr=0.0, lolr_repaid=0.0, lolr_interest=0.0, bailout=0.0, transfer=0.0)

    # ------------------------------------------------------------------ measures
    def cpi(self, world) -> float:
        bread = world.vars.get("price_level", 1.0)
        svc = [f.price for f in self.m.firms if getattr(self.m, "sector_names", [""])[f.sector] == "services"] \
            if hasattr(self.m, "sector_names") else []
        if not svc:
            return bread
        w = self.p.services_weight
        return (1 - w) * bread + w * float(np.mean(svc))

    def unemployment_now(self, world) -> float:
        m = self.m
        N = float(m.size.sum())
        if N <= 0:
            return 0.0
        working = sum(f.hired + f.builders for f in m.firms) + world.vars.get("public_workers", 0.0)
        return float(np.clip(1.0 - working / N, 0.0, 1.0))

    # ------------------------------------------------------------------ phase 1: set the rate
    def set_policy(self, world) -> None:
        p = self.p
        st = world.state["economy"]
        shadow = (p.neutral_rate + p.phi_pi * (self.inflation - p.inflation_target)
                  - p.phi_u * (self.unemployment - p.u_star))
        if p.rule == "money_growth":       # monetary targeting: the rate leans against money above its path
            m_now = self.bank.money_supply(world)
            shadow = p.neutral_rate + p.phi_m * float(np.log(max(m_now, 1e-9) / self.path(world.tick)))
        if world.tick <= p.start_tick or p.rule == "fixed_rate":
            target = p.neutral_rate                  # start-up: prices are still finding their level
        else:
            target = shadow
        self.rate = max(p.smoothing * self.rate + (1 - p.smoothing) * target, p.lower_bound)
        st["policy_rate"] = self.rate
        k = 1.0 - p.rate_sensitivity * (self.rate - p.neutral_rate)
        st["spending_multiplier"] = st.get("spending_multiplier", 1.0) * float(np.clip(k, *p.multiplier_bounds))
        world.vars["shadow_rate"] = shadow

    def transfer(self, world, amount: float) -> None:
        """New money paid to every household per head (a 'helicopter drop' through the bank)."""
        L, m = world.ledger, self.m
        L.post(BANK, "ext:creation", "cash", amount, "creation", "cb_transfer")      # reserves for the bank
        L.post("households", BANK, "deposits", amount, "transfer", "cb_transfer")    # deposits for households
        m.cash += amount * m.size / m.size.sum()
        self.t["transfer"] += amount

    # ------------------------------------------------------------------ emergency lending
    def emergency_loan(self, world, amount: float) -> float:
        if amount <= 1e-12:
            return 0.0
        L = world.ledger
        L.post(BANK, "ext:creation", "cash", amount, "creation", "lolr")
        L.post(CB, BANK, "cb_loans", amount, "transfer", "lolr")
        self.t["lolr"] += amount
        return amount

    def bailout(self, world, amount: float) -> None:
        if amount <= 1e-12:
            return
        world.ledger.post(BANK, "ext:creation", "cash", amount, "creation", "bailout")
        self.t["bailout"] += amount
        self.bailout_cost += amount

    # ------------------------------------------------------------------ phase 7 (after the bank)
    def path(self, tick: int) -> float:
        return self.m0 * (1 + self.p.money_growth) ** tick

    def settle(self, world) -> None:
        if self.p.rule == "money_growth":
            gap = self.path(world.tick) - self.bank.money_supply(world)
            if gap > 1e-9:          # money fell short of the path (repayments, defaults): new money per head
                self.transfer(world, gap)

    def collect(self, world) -> None:
        """Called by the bank at the start of its settlement: interest and repayment of emergency loans."""
        p, L = self.p, world.ledger
        owed = L.balance(CB, "cb_loans")
        if owed > 1e-12:
            reserves = max(L.balance(BANK, "cash"), 0.0)
            due = (self.rate + p.lolr_penalty) * owed
            interest = min(due, reserves)
            if interest > 1e-12:
                L.post("ext:destruction", BANK, "cash", interest, "destruction", "lolr_interest")
                reserves -= interest
            if due - interest > 1e-12:          # what reserves cannot pay is added to the debt
                L.post(CB, BANK, "cb_loans", due - interest, "transfer", "lolr_interest")
            self.t["lolr_interest"] += due
            self.bank.tick_stats["cb_interest"] = self.bank.tick_stats.get("cb_interest", 0.0) + due
            if not self.bank.run_last:
                back = min(p.lolr_repay * owed if owed > 1e-6 else owed, reserves)
                if back > 1e-12:
                    L.post("ext:destruction", BANK, "cash", back, "destruction", "lolr_repay")
                    L.post(BANK, CB, "cb_loans", back, "transfer", "lolr_repay")
                    self.t["lolr_repaid"] += back

    # ------------------------------------------------------------------ phase 8: measure
    def observe(self, world) -> None:
        price = self.cpi(world)
        if self.cpi_prev is not None and self.cpi_prev > 0:
            infl = price / self.cpi_prev - 1
            a = self.p.inflation_smoothing
            self.inflation = (1 - a) * self.inflation + a * infl
        self.cpi_prev = price
        self.unemployment = self.unemployment_now(world)
        world.vars["cb_stats"] = self.t
        self.t = self.zero()

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.set_policy, Phase.FINANCE: self.settle, Phase.DEMOGRAPHY: self.observe}

    # ------------------------------------------------------------------ invariants
    def invariants(self):
        def rate_floor(world, ctx):
            if self.p.rule == "taylor" and self.rate < self.p.lower_bound - 1e-12:
                return [Violation("monetary_policy", f"policy rate {self.rate} below the lower bound")]
            return []

        def cb_claims(world, ctx):
            L = world.ledger
            if L.balance(CB, "cb_loans") < -1e-9:
                return [Violation("monetary_policy", "central bank owes the bank", ("cb_loans",), (CB,))]
            return []

        return [Invariant("monetary_policy", rate_floor), Invariant("cb_claims", cb_claims)]

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        def stat(k):
            return lambda w: w.vars.get("cb_stats", {}).get(k, 0.0)

        return [
            Metric("policy_rate", lambda w: self.rate, "central bank policy rate per tick", "1/tick"),
            Metric("shadow_rate", lambda w: w.vars.get("shadow_rate", self.rate), "rate the Taylor rule asks for, before the floor", "1/tick"),
            Metric("cpi", lambda w: self.cpi_prev if self.cpi_prev is not None else 0.0, "price index (bread and services)", "money"),
            Metric("cpi_inflation", lambda w: self.inflation, "smoothed inflation of the price index", "1/tick"),
            Metric("spending_multiplier", lambda w: w.state["economy"].get("spending_multiplier", 1.0),
                   "how much households spend relative to normal (rates and demand shocks)", "x"),
            Metric("emergency_loans", lambda w: w.ledger.balance(CB, "cb_loans"), "bank's debt to the central bank", "money"),
            Metric("lolr_lent", stat("lolr"), "emergency cash lent this tick", "money"),
            Metric("money_created", stat("transfer"), "new money paid to households this tick (money rule)", "money"),
            Metric("bailout", stat("bailout"), "new money injected into the bank this tick", "money"),
            Metric("bailout_cost", lambda w: self.bailout_cost, "all bailout money so far", "money"),
            Metric("credit_frozen", lambda w: float(self.bank.frozen > 0), "1 while the bank may not lend after a failure", "flag"),
        ]


# ---------------------------------------------------------------------------
class DemandShock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=1)
    duration: int = Field(1, ge=1)
    multiplier: float = Field(0.7, ge=0)


class DemandShocksParams(LayerParams):
    shocks: list[DemandShock] = Field(default_factory=list)
    volatility: float = Field(0.0, ge=0, description="random mood swings: log spending multiplier follows an AR(1) "
                                                     "with this shock sd (Stage 7)")
    persistence: float = Field(0.8, ge=0, lt=1, description="AR(1) coefficient of the random swings")


class DemandShocks(Layer):
    """A sudden wish to save: household spending is multiplied during each shock."""

    name = "demand_shocks"
    requires = ["market"]
    Params = DemandShocksParams
    p: DemandShocksParams

    def setup(self, world) -> None:
        self.mood_rng = np.random.default_rng([world.seed, 7002])
        self.mood = 0.0

    def apply(self, world) -> None:
        k = 1.0
        for s in self.p.shocks:
            if s.start <= world.tick < s.start + s.duration:
                k *= s.multiplier
        if self.p.volatility > 0:
            self.mood = self.p.persistence * self.mood + self.p.volatility * self.mood_rng.standard_normal()
            k *= float(np.exp(self.mood))
        st = world.state["economy"]
        st["spending_multiplier"] = st.get("spending_multiplier", 1.0) * k
        world.vars["demand_shock"] = k

    def hooks(self):
        return {Phase.SHOCKS_POLICY: self.apply}

    def metrics(self):
        return [Metric("demand_shock", lambda w: w.vars.get("demand_shock", 1.0), "spending shock multiplier", "x")]
