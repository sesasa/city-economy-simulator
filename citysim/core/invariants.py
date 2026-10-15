"""Conservation and sanity checks run at the end of every tick.

An invariant is a function `(world, ctx) -> list[Violation]`. Engine-level
invariants below are always on; layers add their own via `Layer.invariants()`.
Any violation stops the run with the tick number and a trace of the postings
involved (`InvariantViolation`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from .ledger import OUTFLOW_FLOWS, Posting

if TYPE_CHECKING:
    from .world import World

REL_TOL = 1e-9
ABS_TOL = 1e-9


def close(a: float, b: float, scale: float = 1.0) -> bool:
    return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b), scale)


@dataclass
class Violation:
    invariant: str
    message: str
    items: tuple[str, ...] = ()
    accounts: tuple[str, ...] = ()


@dataclass
class TickContext:
    """What the engine knows about the tick being checked."""

    tick: int
    opening: dict[tuple[str, str], float]  # ledger balances at tick start
    journal: list[Posting] = field(default_factory=list)


InvariantFn = Callable[["World", TickContext], list[Violation]]


@dataclass
class Invariant:
    name: str
    fn: InvariantFn

    def __call__(self, world: "World", ctx: TickContext) -> list[Violation]:
        return self.fn(world, ctx)


class InvariantViolation(RuntimeError):
    def __init__(self, tick: int, violations: list[Violation], journal: list[Posting]):
        self.tick = tick
        self.violations = violations
        lines = [f"Invariant violation at tick {tick}:"]
        for v in violations:
            lines.append(f"  [{v.invariant}] {v.message}")
        involved = [p for p in journal
                    if any(p.item in v.items or p.debit in v.accounts or p.credit in v.accounts
                           for v in violations)] or journal
        lines.append(f"  Trace ({len(involved)} postings this tick):")
        lines += [f"    {p}" for p in involved[:200]]
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------

def _internal_totals(world: "World", balances: dict[tuple[str, str], float]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    accounts = world.ledger.accounts
    for (acc, item), v in balances.items():
        if not accounts[acc].external:
            out[item] += v
    return out


def double_entry(world: "World", ctx: TickContext) -> list[Violation]:
    """For every item, balances over all accounts (incl. external) sum to zero."""
    totals: dict[str, float] = defaultdict(float)
    scale: dict[str, float] = defaultdict(float)
    for (_acc, item), v in world.ledger.balances.items():
        totals[item] += v
        scale[item] = max(scale[item], abs(v))
    return [Violation("double_entry", f"{item}: all-account sum = {t:.3e} != 0", (item,))
            for item, t in totals.items() if not close(t, 0.0, scale[item])]


def goods_conservation(world: "World", ctx: TickContext) -> list[Violation]:
    """next stock = stock + production - use - consumption - depreciation - spoilage - waste."""
    out: list[Violation] = []
    start = _internal_totals(world, ctx.opening)
    end = _internal_totals(world, world.ledger.balances)
    flows: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in ctx.journal:
        flows[p.item][p.flow] += p.amount
    for item, kind in world.ledger.items.items():
        if kind in ("money", "claim"):
            continue
        f = flows[item]
        expected = start.get(item, 0.0) + f["production"] + f["import"] - sum(f[x] for x in OUTFLOW_FLOWS)
        actual = end.get(item, 0.0)
        if not close(actual, expected, start.get(item, 0.0)):
            out.append(Violation(
                "goods_conservation",
                f"{item}: stock {actual:.9g} != opening {start.get(item, 0.0):.9g} + production "
                f"{f['production']:.9g} - outflows {sum(f[x] for x in OUTFLOW_FLOWS):.9g}",
                (item,)))
    return out


def money_conservation(world: "World", ctx: TickContext) -> list[Violation]:
    """Total internal money changes only through explicit creation/destruction postings."""
    out: list[Violation] = []
    start = _internal_totals(world, ctx.opening)
    end = _internal_totals(world, world.ledger.balances)
    for item, kind in world.ledger.items.items():
        if kind != "money":
            continue
        created = sum(p.amount for p in ctx.journal if p.item == item and p.flow in ("creation", "import"))
        destroyed = sum(p.amount for p in ctx.journal if p.item == item and p.flow in ("destruction", "export"))
        illegal = [p for p in ctx.journal if p.item == item
                   and p.flow not in ("creation", "destruction", "transfer", "import", "export")]
        if illegal:
            out.append(Violation("money_conservation", f"{item}: money moved by non-monetary flow "
                                 f"{sorted({p.flow for p in illegal})}", (item,)))
        if not close(end.get(item, 0.0), start.get(item, 0.0) + created - destroyed, start.get(item, 0.0)):
            out.append(Violation("money_conservation", f"{item}: unexplained change in money stock", (item,)))
    return out


def no_negative_stocks(world: "World", ctx: TickContext) -> list[Violation]:
    out: list[Violation] = []
    accounts = world.ledger.accounts
    for (acc, item), v in world.ledger.balances.items():
        a = accounts[acc]
        if world.ledger.items.get(item) == "claim":
            continue   # claims (deposits, loans) are negative for the debtor by design; layers check their signs
        if not a.external and not a.allow_negative and v < -ABS_TOL * max(1.0, abs(v)):
            out.append(Violation("no_negative_stocks", f"{acc} holds {v:.6g} of {item}", (item,), (acc,)))
    return out


def balance_sheets(world: "World", ctx: TickContext) -> list[Violation]:
    """Every agent's balance sheet balances, and so does the aggregate.

    Balance sheets are derived from the ledger: assets are positive balances,
    liabilities are negative balances (only allowed on accounts flagged
    `allow_negative`), net worth = assets - liabilities. Financial items
    (kind 'money') are claims, so across all internal holders they must net to
    the amount explicitly created from outside (ext:creation / ext:destruction).
    Real goods are assets without a counterpart liability.
    """
    out: list[Violation] = []
    ledger = world.ledger
    for item, kind in ledger.items.items():
        if kind != "money":
            continue
        internal = sum(v for (a, i), v in ledger.balances.items() if i == item and not ledger.accounts[a].external)
        outside = -sum(ledger.balance(f"ext:{f}", item) for f in ("creation", "destruction", "import", "export"))
        if not close(internal, outside):
            out.append(Violation("balance_sheets", f"{item}: internal net claims {internal:.6g} "
                                 f"!= outside money {outside:.6g}", (item,)))
    for name, acc in ledger.accounts.items():
        if acc.external:
            continue
        assets = sum(v for (a, _i), v in ledger.balances.items() if a == name and v > 0)
        liabilities = -sum(v for (a, _i), v in ledger.balances.items() if a == name and v < 0)
        net_worth = sum(v for (a, _i), v in ledger.balances.items() if a == name)
        if not close(assets - liabilities, net_worth, assets):
            out.append(Violation("balance_sheets", f"{name}: assets - liabilities != net worth", (), (name,)))
    return out


ENGINE_INVARIANTS: list[Invariant] = [
    Invariant("double_entry", double_entry),
    Invariant("goods_conservation", goods_conservation),
    Invariant("money_conservation", money_conservation),
    Invariant("no_negative_stocks", no_negative_stocks),
    Invariant("balance_sheets", balance_sheets),
]


def check_all(world: "World", ctx: TickContext, invariants: list[Invariant]) -> None:
    violations: list[Violation] = []
    for inv in invariants:
        violations.extend(inv(world, ctx))
    if violations:
        raise InvariantViolation(ctx.tick, violations, ctx.journal)
