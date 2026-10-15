"""Double-entry ledger for goods and money.

Conventions
-----------
* Every account belongs to a holder (an agent/sector) or is *external*
  (a source or sink standing for nature, consumption, depreciation, ...).
* A holder's balance of an item is its stock: debit increases it, credit
  decreases it. Moving `x` of an item from A to B debits B and credits A.
* Bringing goods into existence credits an external source account;
  removing them debits an external sink account. So for every item the sum
  of balances over *all* accounts is always zero (the double-entry identity),
  and the change in internal stocks equals the net of the external flows.
* Every posting carries a `flow` tag. Goods-conservation checks group the
  journal by flow tag; money-conservation checks only allow internal money to
  change through `creation` / `destruction` postings, or `import` / `export`
  when it crosses to another city (Stage 8; a region-level check matches the two sides).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# Flow tags and the direction they must have relative to external accounts.
INFLOW_FLOWS = {"production", "creation", "import"}  # external source -> holder
OUTFLOW_FLOWS = {"use", "consumption", "depreciation", "spoilage", "waste", "destruction", "export"}
INTERNAL_FLOWS = {"transfer"}                         # holder -> holder
ALL_FLOWS = INFLOW_FLOWS | OUTFLOW_FLOWS | INTERNAL_FLOWS


@dataclass(frozen=True)
class Posting:
    tick: int
    debit: str
    credit: str
    item: str
    amount: float
    flow: str
    note: str = ""

    def __str__(self) -> str:
        return (f"t={self.tick} {self.flow:<12} {self.item:<20} {self.amount:>14.6f}  "
                f"Dr {self.debit:<24} Cr {self.credit:<24} {self.note}")


@dataclass
class Account:
    name: str
    external: bool = False
    allow_negative: bool = False


class LedgerError(RuntimeError):
    pass


@dataclass
class Ledger:
    accounts: dict[str, Account] = field(default_factory=dict)
    items: dict[str, str] = field(default_factory=dict)          # item -> kind ('good'|'capital'|'money'|'claim')
    balances: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    journal: list[Posting] = field(default_factory=list)         # postings of the current tick
    tick: int = 0
    keep_history: bool = False
    history: list[Posting] = field(default_factory=list)

    # ---- registration -------------------------------------------------
    def open_account(self, name: str, *, external: bool = False, allow_negative: bool = False) -> None:
        if name in self.accounts:
            acc = self.accounts[name]
            if acc.external != external:
                raise LedgerError(f"account {name!r} re-opened with different 'external' flag")
            acc.allow_negative = acc.allow_negative or allow_negative
            return
        self.accounts[name] = Account(name, external, allow_negative)

    def register_item(self, item: str, kind: str = "good") -> None:
        if kind not in ("good", "capital", "money", "claim"):
            raise LedgerError(f"unknown item kind {kind!r}")
        self.items.setdefault(item, kind)

    # ---- posting --------------------------------------------------------
    def post(self, debit: str, credit: str, item: str, amount: float, flow: str, note: str = "") -> None:
        if amount == 0.0:
            return
        if amount < 0:
            raise LedgerError(f"negative posting amount {amount} ({flow} {item} {debit}<-{credit})")
        if item not in self.items:
            raise LedgerError(f"unregistered item {item!r}")
        if flow not in ALL_FLOWS:
            raise LedgerError(f"unknown flow tag {flow!r}")
        dr, cr = self.accounts.get(debit), self.accounts.get(credit)
        if dr is None or cr is None:
            raise LedgerError(f"unknown account in posting: {debit!r} / {credit!r}")
        # Direction checks keep the flow tags meaningful.
        if flow in INFLOW_FLOWS and not (cr.external and not dr.external):
            raise LedgerError(f"{flow} must credit an external source and debit a holder")
        if flow in OUTFLOW_FLOWS and not (dr.external and not cr.external):
            raise LedgerError(f"{flow} must debit an external sink and credit a holder")
        if flow in INTERNAL_FLOWS and (dr.external or cr.external):
            raise LedgerError("transfer must be between two holders")
        self.balances[(debit, item)] += amount
        self.balances[(credit, item)] -= amount
        p = Posting(self.tick, debit, credit, item, float(amount), flow, note)
        self.journal.append(p)
        if self.keep_history:
            self.history.append(p)

    # convenience wrappers ------------------------------------------------
    def produce(self, holder: str, item: str, amount: float, note: str = "") -> None:
        self.post(holder, "ext:production", item, amount, "production", note)

    def remove(self, holder: str, item: str, amount: float, flow: str, note: str = "") -> None:
        self.post(f"ext:{flow}", holder, item, amount, flow, note)

    def transfer(self, src: str, dst: str, item: str, amount: float, note: str = "") -> None:
        self.post(dst, src, item, amount, "transfer", note)

    # ---- queries --------------------------------------------------------
    def balance(self, account: str, item: str) -> float:
        return self.balances.get((account, item), 0.0)

    def internal_total(self, item: str) -> float:
        return sum(v for (a, i), v in self.balances.items()
                   if i == item and not self.accounts[a].external)

    def snapshot(self) -> dict[tuple[str, str], float]:
        return dict(self.balances)

    def holdings(self, account: str) -> dict[str, float]:
        return {i: v for (a, i), v in self.balances.items() if a == account and v != 0.0}

    def begin_tick(self, tick: int) -> None:
        self.tick = tick
        self.journal = []


def open_standard_external_accounts(ledger: Ledger) -> None:
    ledger.open_account("ext:production", external=True, allow_negative=True)
    for flow in OUTFLOW_FLOWS | {"creation", "import"}:
        ledger.open_account(f"ext:{flow}", external=True, allow_negative=True)
