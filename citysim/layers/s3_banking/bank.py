"""Stage 3: credit and banking.

One bank takes everyone's money as deposits (holding the cash as reserves) and
lends to firms. A loan creates a deposit for the borrower, so the money supply
(deposits held outside the bank plus cash outside the bank) becomes endogenous.

Ledger items added: "deposits" and "loans" (kind "claim": an asset for the
holder, a negative balance — a liability — for the issuer). Cash stays outside
money: the bank holds it as reserves and pays it out in a run.

Per tick (Phase 7, FINANCE): borrowers pay interest and repay a fixed share of
principal; a borrower too far behind defaults and the bank writes off the loss;
the bank pays interest on deposits and dividends when well capitalised; if its
capital ratio falls below the run threshold, households withdraw cash, and if the
reserves run out the bank fails and depositors take a haircut.
Lending happens whenever the market asks for it (working capital for wages and
inputs, loans for building), within a borrower limit and the capital requirement.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.invariants import Invariant, Violation, close
from ...core.layer import Layer, LayerParams
from ...core.metrics import Metric
from ...core.phases import Phase

BANK = "bank:0"
BANK_CATEGORIES = ("deposit", "loan", "repayment", "loan_interest", "deposit_interest", "bank_dividends",
                   "withdrawal", "redeposit", "bank_failure", "default", "default_seizure",
                   "bond_purchase", "bond_interest", "bond_redemption", "cb_transfer",
                   "margin_loan", "margin_repayment", "margin_interest", "margin_default",
                   "remit_out", "remit_in")
# how each category changes the money supply (deposits outside the bank): + creates, - destroys, 0 neutral
MONEY_EFFECT = {"loan": +1, "repayment": -1, "loan_interest": -1, "deposit_interest": +1, "bank_dividends": +1,
                "withdrawal": -1, "redeposit": +1, "bank_failure": -1, "default_seizure": -1, "deposit": +1}


class BankParams(LayerParams):
    loan_rate: float = Field(0.01, ge=0, description="interest per tick on loans")
    pass_through: float = Field(0.5, ge=0, le=1, description="share of loan interest paid on as deposit interest")
    amortization: float = Field(0.05, ge=0, le=1, description="share of principal repaid per tick")
    capital_requirement: float = Field(0.08, ge=0, description="minimum bank equity / loans")
    initial_capital: float = Field(0.1, ge=0, description="bank equity at start, as a share of initial money")
    dividend_target: float = Field(0.15, ge=0, description="capital ratio above which the bank pays dividends")
    ltv: float = Field(0.5, ge=0, description="lend up to this share of a borrower's collateral (capital + stocks)")
    revenue_multiple: float = Field(1.0, ge=0, description="plus this many ticks of the borrower's average revenue")
    credit_scale: float = Field(1.0, ge=0, description="multiplies every borrower's limit (credit-limit experiment)")
    dsti: float = Field(0.15, ge=0, description="debt service (interest + repayment) may not exceed this share of "
                                                "the borrower's average revenue")
    working_capital: bool = True
    investment_lending: bool = True
    recovery: float = Field(0.4, ge=0, le=1, description="share of a defaulted loan the bank gets back")
    arrears_limit: float = Field(0.1, ge=0, description="default when unpaid dues exceed this share of the debt")
    procyclicality: float = Field(0.0, ge=0, description="Minsky: how fast lending standards loosen in good times")
    optimism_max: float = Field(3.0, ge=0, description="largest loosening (limit multiplier = 1 + optimism)")
    panic_sensitivity: float = Field(20.0, ge=0, description="how sharply optimism collapses with defaults")
    run_threshold: float = Field(0.04, ge=0, description="households run when the capital ratio falls below this")
    withdrawal_rate: float = Field(0.3, ge=0, le=1, description="share of deposits withdrawn per tick in a run")
    redeposit_rate: float = Field(0.1, ge=0, le=1, description="share of hoarded cash returned per tick when calm")
    freeze_ticks: int = Field(10, ge=0, description="no new lending for this long after a bank failure")
    exclusion_ticks: int = Field(20, ge=0, description="a borrower who defaulted gets no credit for this long")
    bond_risk_weight: float = Field(1.0, ge=0, description="capital needed for government bonds, relative to loans")
    panic_start: int = Field(0, ge=0, description="a rumour: depositors run from this tick whatever the bank's capital "
                                               "(0: never; Stage 6)")
    panic_duration: int = Field(0, ge=0, description="ticks the rumour lasts")
    panic_withdrawal: float = Field(0.4, ge=0, le=1, description="share of deposits withdrawn per tick in the panic")
    loan_spread: float = Field(0.007, ge=0, description="loan rate above the central bank's policy rate (Stage 6)")
    cash_target: float = Field(1.0, ge=0, description="borrowers repay early with cash beyond this many ticks of costs")
    hurdle_margin: float = Field(0.01, ge=0, description="a borrowed labor-saving project must return loan rate + "
                                                         "amortization + this, per tick, in wages saved")


class Banking(Layer):
    name = "banking"
    requires = ["market"]
    provides = ["bank"]
    Params = BankParams
    aliases = {"ltv": "ltv", "kappa": "capital_requirement"}
    p: BankParams

    # ================================================================== setup
    def setup(self, world) -> None:
        L, p = world.ledger, self.p
        st = world.state["economy"]
        self.m = st["market_layer"]
        self.extra_holders = st.setdefault("money_holders", [])     # e.g. the government (Stage 5)
        L.register_item("deposits", "claim")
        L.register_item("loans", "claim")
        L.register_item("bonds", "claim")
        L.open_account(BANK)
        # everyone's cash goes into the bank: deposits for them, reserves for the bank
        for acc in ["households", *[f.account for f in self.m.firms], *self.extra_holders]:
            x = L.balance(acc, "cash")
            if x > 0:
                L.post(BANK, acc, "cash", x, "transfer", "deposit")
                L.post(acc, BANK, "deposits", x, "transfer", "deposit")
        money0 = sum(L.balance(a, "deposits") for a in ["households", *[f.account for f in self.m.firms]])
        st["bond_buyer"] = self.buy_bonds
        L.post(BANK, "ext:creation", "cash", p.initial_capital * money0, "creation", "bank capital")
        self.m.money_item = "deposits"
        st["boldness"] = 0.0
        st["credit_providers"].append(self.working_capital)
        st["investment_finance"].append(self.investment_loan)
        st["extra_categories"] = list(st.get("extra_categories", [])) + list(BANK_CATEGORIES)
        st["bank_layer"] = self
        st["debt_service"] = self.debt_service
        st["debt_of"] = self.debt
        st["hurdle_rate"] = p.loan_rate + p.amortization + p.hurdle_margin
        self.excluded = {f.account: 0 for f in self.m.firms}
        self.optimism = 0.0
        self.frozen = 0
        self.failures = 0
        self.run_last = 0
        self.revenue_avg = {f.account: 0.0 for f in self.m.firms}
        self.arrears = {f.account: 0.0 for f in self.m.firms}
        self.tick_stats = self.zero_stats()
        self.money_prev = self.money_supply(world)

    @staticmethod
    def zero_stats():
        return dict(lent=0.0, repaid=0.0, interest_in=0.0, interest_out=0.0, defaults=0.0, written_off=0.0,
                    withdrawn=0.0, redeposited=0.0, haircut=0.0, dividends=0.0, n_defaults=0, run=0, failed=0)

    # ================================================================== measures
    def holders(self):
        return ["households", *[f.account for f in self.m.firms], *self.extra_holders]

    def money_supply(self, world) -> float:
        L = world.ledger
        return sum(L.balance(a, "deposits") for a in self.holders()) + L.balance("households", "cash")

    def loans_total(self, world) -> float:
        return world.ledger.balance(BANK, "loans")

    def equity(self, world) -> float:
        L = world.ledger
        return (L.balance(BANK, "cash") + L.balance(BANK, "loans") + L.balance(BANK, "bonds") + L.balance(BANK, "deposits")
                + L.balance(BANK, "cb_loans") + L.balance(BANK, "cb_bills"))   # central bank items (Stage 6)

    def risk_assets(self, world) -> float:
        """Loans plus government bonds at their risk weight (0: bonds use no capital)."""
        return self.loans_total(world) + self.p.bond_risk_weight * world.ledger.balance(BANK, "bonds")

    def capital_ratio(self, world) -> float:
        ra = self.risk_assets(world)
        return self.equity(world) / ra if ra > 1e-9 else np.inf

    def buy_bonds(self, world, seller: str, amount: float) -> float:
        """The bank buys newly issued government bonds, paying with new deposits (within its capital)."""
        if amount <= 1e-12 or self.frozen > 0:
            return 0.0
        kappa, w = self.p.capital_requirement, self.p.bond_risk_weight
        room = (self.equity(world) / kappa - self.risk_assets(world)) / w if (kappa > 0 and w > 0) else np.inf
        x = min(amount, max(room, 0.0))
        if x <= 1e-9:
            return 0.0
        L = world.ledger
        L.post(BANK, seller, "bonds", x, "transfer", "bond_purchase")
        L.post(seller, BANK, "deposits", x, "transfer", "bond_purchase")
        return x

    def restart_capital(self, world) -> float:
        """Capital a resolved (or bailed-out) bank is left with: the requirement on its assets, and at least
        enough to lend a tenth of its deposits, so it is not a zombie that can never lend again."""
        kappa = self.p.capital_requirement
        if not self.modern(world):           # Stages 3-5 keep their original rule (published results)
            return kappa * self.risk_assets(world)
        return kappa * max(self.risk_assets(world), 0.1 * max(-world.ledger.balance(BANK, "deposits"), 0.0))

    @staticmethod
    def modern(world) -> bool:
        """Resolution rules added in Stage 6 apply once there is a central bank."""
        return world.state["economy"].get("central_bank") is not None

    def debt(self, world, f) -> float:
        return -world.ledger.balance(f.account, "loans")

    def collateral(self, world, f) -> float:
        """Capital at replacement cost (labor-ticks x average wage) plus stocks at current prices."""
        m, L = self.m, world.ledger
        mp = m.p
        wage = float(np.mean([g.wage for g in m.firms]))
        value = (L.balance(f.account, "factory") * mp.factory_cost
                 + L.balance(f.account, "land_improvement") * mp.land_improvement_cost
                 + (L.balance(f.account, "tools_processing") + L.balance(f.account, "tools_distribution")) * mp.tool_cost) * wage
        value += (L.balance(f.account, "raw") + L.balance(f.account, "final")) * f.price
        return value

    def limit(self, world, f) -> float:
        """Collateral-based limit, capped by what the borrower's revenue can service."""
        p = self.p
        # recent revenue, or what the firm expects to sell, whichever is larger (lets a stalled firm restart)
        rev = max(self.revenue_avg.get(f.account, 0.0), f.expected * f.price)
        base = p.ltv * self.collateral(world, f) + p.revenue_multiple * rev
        serviceable = p.dsti * rev / max(self.rate(world) + p.amortization, 1e-9)
        return min(base, serviceable) * p.credit_scale * (1.0 + self.optimism)

    # ================================================================== lending
    def rate(self, world) -> float:
        """Loan rate: fixed, or the central bank's policy rate plus a spread (Stage 6)."""
        pr = world.state["economy"].get("policy_rate")
        return self.p.loan_rate if pr is None else max(pr + self.p.loan_spread, 0.0)

    def debt_service(self, world, f) -> float:
        """What the borrower owes at the next settlement (interest + scheduled repayment + arrears)."""
        d = self.debt(world, f)
        return (self.rate(world) + self.p.amortization) * d + self.arrears.get(f.account, 0.0)

    def lend(self, world, f, amount: float, note: str = "loan") -> float:
        if amount <= 1e-12 or self.frozen > 0 or self.excluded.get(f.account, 0) > 0:
            return 0.0
        room_borrower = max(0.0, self.limit(world, f) - self.debt(world, f))
        kappa = self.p.capital_requirement
        room_bank = max(0.0, self.equity(world) / kappa - self.risk_assets(world)) if kappa > 0 else np.inf
        x = min(amount, room_borrower, room_bank)
        if x <= 1e-9:
            return 0.0
        L = world.ledger
        L.post(BANK, f.account, "loans", x, "transfer", "loan")       # bank's asset, firm's liability
        L.post(f.account, BANK, "deposits", x, "transfer", "loan")    # new money for the firm
        self.tick_stats["lent"] += x
        return x

    def working_capital(self, world, f, amount):
        return self.lend(world, f, amount) if self.p.working_capital else 0.0

    def investment_loan(self, world, f, amount):
        return self.lend(world, f, amount) if self.p.investment_lending else 0.0

    # ================================================================== phase 7
    def settle(self, world) -> None:
        p, L, m = self.p, world.ledger, self.m
        stats = self.tick_stats
        if self.frozen > 0:
            self.frozen -= 1
        for k in self.excluded:
            self.excluded[k] = max(0, self.excluded[k] - 1)
        loans_before = max(self.loans_total(world), 1e-9)
        for pay in world.state["economy"].get("bank_obligations", []):     # e.g. central bank loans (Stage 6)
            pay(world)
        # borrowers pay interest and repay principal; too far behind -> default
        for f in m.firms:
            self.revenue_avg[f.account] = 0.8 * self.revenue_avg.get(f.account, f.revenue) + 0.2 * f.revenue
            self.arrears.setdefault(f.account, 0.0)
            f.interest_paid = 0.0
            debt = self.debt(world, f)
            if debt <= 1e-12:
                self.arrears[f.account] = 0.0
                continue
            have = max(L.balance(f.account, "deposits"), 0.0)
            interest = self.rate(world) * debt
            pay_i = min(interest, have)
            if pay_i > 0:
                L.post(BANK, f.account, "deposits", pay_i, "transfer", "loan_interest")
                f.interest_paid = pay_i
                stats["interest_in"] += pay_i
            have -= pay_i
            principal = min(p.amortization * debt, debt)
            pay_p = min(principal, have)
            if pay_p > 0:
                L.post(BANK, f.account, "deposits", pay_p, "transfer", "repayment")
                L.post(f.account, BANK, "loans", pay_p, "transfer", "repayment")
                stats["repaid"] += pay_p
            have -= pay_p
            # revolving credit: cash beyond one tick of running costs pays the loan down
            extra = min(max(have - p.cash_target * f.op_costs, 0.0), self.debt(world, f))
            if extra > 1e-9:
                L.post(BANK, f.account, "deposits", extra, "transfer", "repayment")
                L.post(f.account, BANK, "loans", extra, "transfer", "repayment")
                stats["repaid"] += extra
            self.arrears[f.account] += (interest - pay_i) + (principal - pay_p)
            debt = self.debt(world, f)
            if debt > 1e-9 and self.arrears[f.account] > p.arrears_limit * debt:
                self.default(world, f, debt)
        # an insolvent bank is resolved at once (depositors bailed in) unless the authorities bail it out
        if self.equity(world) < 0:
            bailout = world.state["economy"].get("bailout")
            if bailout is not None:
                bailout(world, self.restart_capital(world) - self.equity(world))
            tol = 1e-9 * max(-L.balance(BANK, "deposits"), 1.0) if self.modern(world) else 0.0
            if self.equity(world) < -tol:
                self.fail(world)
        # depositors get a share of this tick's loan interest; the bank keeps the spread
        income = stats["interest_in"]
        deposits_out = sum(max(L.balance(a, "deposits"), 0.0) for a in self.holders())
        pay = p.pass_through * income
        if pay > 1e-12 and deposits_out > 0:
            for acc in self.holders():
                d = L.balance(acc, "deposits")
                if d > 0:
                    x = pay * d / deposits_out
                    L.post(acc, BANK, "deposits", x, "transfer", "deposit_interest")
                    stats["interest_out"] += x
                    if acc == "households":
                        m.cash += x * m.cash / max(m.cash.sum(), 1e-12)
        # dividends: half of this tick's profit, only when the bank is well capitalised
        profit = income - stats["interest_out"] - stats["written_off"] - stats.get("cb_interest", 0.0)
        loans = self.risk_assets(world)
        if profit > 0 and (loans <= 1e-9 or self.equity(world) / loans > p.dividend_target):
            div = 0.5 * profit
            L.post("households", BANK, "deposits", div, "transfer", "bank_dividends")
            m.cash += div * m.size / m.size.sum()
            stats["dividends"] += div
        # depositors' confidence
        self.run_or_calm(world)
        # Minsky: lending standards loosen when loans perform and tighten sharply after defaults
        if p.procyclicality > 0:
            default_rate = stats["defaults"] / loans_before
            growth = stats["lent"] / loans_before
            self.optimism += p.procyclicality * (1.0 if default_rate < 1e-6 else -p.panic_sensitivity * default_rate) \
                * (1.0 if growth > 0 else 0.5)
            self.optimism = float(np.clip(self.optimism, 0.0, p.optimism_max))
        world.state["economy"]["boldness"] = self.optimism / p.optimism_max if p.optimism_max > 0 else 0.0
        world.state["economy"]["hurdle_rate"] = self.rate(world) + p.amortization + p.hurdle_margin

    def default(self, world, f, debt: float) -> None:
        L, p = world.ledger, self.p
        stats = self.tick_stats
        seize = min(max(L.balance(f.account, "deposits"), 0.0), p.recovery * debt)
        if seize > 0:
            L.post(BANK, f.account, "deposits", seize, "transfer", "default_seizure")
            L.post(f.account, BANK, "loans", seize, "transfer", "default_seizure")
        writeoff = self.debt(world, f)
        if writeoff > 0:
            L.post(f.account, BANK, "loans", writeoff, "transfer", "default")   # the debt is cancelled
        stats["defaults"] += debt
        stats["written_off"] += writeoff
        stats["n_defaults"] += 1
        self.arrears[f.account] = 0.0
        self.excluded[f.account] = p.exclusion_ticks

    def run_or_calm(self, world) -> None:
        p, L, m = self.p, world.ledger, self.m
        stats = self.tick_stats
        ratio = self.capital_ratio(world)
        dep = L.balance("households", "deposits")
        panic = p.panic_start > 0 and p.panic_start <= world.tick < p.panic_start + p.panic_duration
        if (ratio < p.run_threshold or panic) and dep > 1e-9:
            stats["run"] = 1
            want = (max(p.withdrawal_rate, p.panic_withdrawal) if panic else p.withdrawal_rate) * dep
            reserves = max(L.balance(BANK, "cash"), 0.0)
            lolr = world.state["economy"].get("lender_of_last_resort")
            if want > reserves and lolr is not None and self.equity(world) >= 0:
                reserves += lolr(world, want - reserves)          # cash lent against the bank's loans
            if want > reserves:
                self.fail(world)
                return
            L.post(BANK, "households", "deposits", want, "transfer", "withdrawal")
            L.post("households", BANK, "cash", want, "transfer", "withdrawal")
            m.cash *= (dep - want) / dep
            stats["withdrawn"] += want
        elif ratio >= 1.5 * p.run_threshold:
            hoard = L.balance("households", "cash")
            if hoard > 1e-9:
                back = p.redeposit_rate * hoard
                L.post(BANK, "households", "cash", back, "transfer", "redeposit")
                L.post("households", BANK, "deposits", back, "transfer", "redeposit")
                m.cash += back * m.size / m.size.sum()
                stats["redeposited"] += back

    def fail(self, world) -> None:
        """Reserves can't meet the run: depositors take a haircut that restores the capital requirement."""
        p, L, m = self.p, world.ledger, self.m
        stats = self.tick_stats
        deposits = -L.balance(BANK, "deposits")
        senior = -L.balance(BANK, "cb_loans")       # the central bank is paid in full (Stage 6)
        assets = L.balance(BANK, "cash") + L.balance(BANK, "loans") + L.balance(BANK, "bonds") + L.balance(BANK, "cb_bills")
        target = self.restart_capital(world)
        if senior > 0:          # a bank still owing the central bank keeps a cushion against that debt's interest
            target = max(target, p.capital_requirement * senior)
        needed = deposits + senior - (assets - target)
        h = float(np.clip(needed / deposits, 0.0, 1.0)) if deposits > 0 else 0.0
        for acc in self.holders():
            d = L.balance(acc, "deposits")
            if d > 0 and h > 0:
                L.post(BANK, acc, "deposits", d * h, "transfer", "bank_failure")
                stats["haircut"] += d * h
        m.cash *= (1 - h)
        self.failures += 1
        self.frozen = p.freeze_ticks
        stats["failed"] = 1
        self.optimism = 0.0

    def hooks(self):
        return {Phase.FINANCE: self.settle, Phase.DEMOGRAPHY: self.end_of_tick}

    def end_of_tick(self, world) -> None:
        world.vars["bank_stats"] = self.tick_stats
        self.run_last = self.tick_stats["run"]
        self.tick_stats = self.zero_stats()

    # ================================================================== invariants
    def invariants(self):
        def claim_signs(world, ctx):
            L, out = world.ledger, []
            for acc in self.holders():
                if L.balance(acc, "deposits") < -1e-9:
                    out.append(Violation("claims", f"{acc} has an overdraft {L.balance(acc, 'deposits')}", ("deposits",), (acc,)))
                if L.balance(acc, "loans") > 1e-9:
                    out.append(Violation("claims", f"{acc} holds loans as an asset", ("loans",), (acc,)))
            if L.balance(BANK, "deposits") > 1e-9 or L.balance(BANK, "loans") < -1e-9:
                out.append(Violation("claims", "bank balance sheet signs wrong", ("deposits", "loans"), (BANK,)))
            if L.balance(BANK, "cash") < -1e-9:
                out.append(Violation("claims", "bank reserves negative", ("cash",), (BANK,)))
            return out

        def money_explained(world, ctx):
            """The money supply changes only through the bank: loans, repayments, interest,
            dividends, defaults, haircuts. Withdrawals move money from deposits to cash outside."""
            out = []
            effect = 0.0
            for p_ in ctx.journal:
                if p_.item == "deposits" and BANK in (p_.debit, p_.credit):
                    if p_.note not in BANK_CATEGORIES:
                        out.append(Violation("money_explained", f"unclassified bank posting {p_}", ("deposits",)))
                        continue
                    if p_.note in ("withdrawal", "redeposit"):
                        continue          # deposits <-> cash outside: money unchanged
                    sign = 1.0 if p_.debit != BANK else -1.0      # deposits created for a holder vs destroyed
                    effect += sign * p_.amount
            now = self.money_supply(world)
            if not close(now - self.money_prev, effect, max(abs(now), 1.0)):
                out.append(Violation("money_explained", f"money moved by {now - self.money_prev:.9g} but bank flows explain {effect:.9g}"))
            self.money_prev = now
            return out

        return [Invariant("claims", claim_signs), Invariant("money_explained", money_explained)]

    # ================================================================== metrics
    def metrics(self):
        def stat(k):
            return lambda w: w.vars.get("bank_stats", {}).get(k, 0.0)

        def firm_debt(w):
            return sum(self.debt(w, f) for f in self.m.firms)

        def leverage(w):
            coll = sum(self.collateral(w, f) for f in self.m.firms)
            return firm_debt(w) / coll if coll > 1e-9 else 0.0

        return [
            Metric("loans", lambda w: self.loans_total(w), "loans outstanding", "money"),
            Metric("money_supply", lambda w: self.money_supply(w), "deposits + cash held outside the bank", "money"),
            Metric("bank_equity", lambda w: self.equity(w), "bank capital", "money"),
            Metric("capital_ratio", lambda w: min(self.capital_ratio(w), 10.0), "bank equity / loans (capped at 10)", "ratio"),
            Metric("reserves", lambda w: w.ledger.balance(BANK, "cash"), "cash held by the bank", "money"),
            Metric("leverage", leverage, "firm debt / firm collateral", "ratio"),
            Metric("new_lending", stat("lent"), "loans made this tick", "money"),
            Metric("repayments", stat("repaid"), "principal repaid this tick", "money"),
            Metric("defaults", stat("defaults"), "debt in default this tick", "money"),
            Metric("default_count", stat("n_defaults"), "borrowers defaulting this tick", "count"),
            Metric("bank_run", stat("run"), "1 if depositors are running", "flag"),
            Metric("bank_failure", stat("failed"), "1 if the bank failed this tick", "flag"),
            Metric("haircut", stat("haircut"), "deposits destroyed by a failure", "money"),
            Metric("cash_hoarded", lambda w: w.ledger.balance("households", "cash"), "cash households keep outside the bank", "money"),
            Metric("optimism", lambda w: self.optimism, "bank's loosening of lending standards", "x"),
        ]
