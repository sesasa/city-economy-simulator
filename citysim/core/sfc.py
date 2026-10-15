"""Stock-flow consistent (Godley-Lavoie) matrices built from the ledger.

* Transaction-flow matrix: rows are kinds of payment (the posting's category note),
  columns are sectors; a payment received is +, a payment made is -. Every row sums
  to zero (what one sector pays another receives) and every column sums to the
  sector's change in money holdings this tick.
* Balance-sheet matrix: rows are items, columns are sectors, values are stocks
  (financial items in money; real items optionally valued with given prices).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import pandas as pd

from .ledger import Ledger, Posting


def transaction_flow_matrix(journal: Iterable[Posting], sector_of: Callable[[str], str | None],
                            money_items: set[str], sectors: list[str]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in journal:
        if p.item not in money_items:
            continue
        label = p.note or p.flow
        for acc, sign in ((p.debit, +1.0), (p.credit, -1.0)):
            s = sector_of(acc)
            if s is not None:
                rows[label][s] += sign * p.amount
            elif p.flow in ("creation", "destruction"):
                rows[f"money {p.flow}"]["(outside)"] += sign * p.amount
    df = pd.DataFrame(rows).T.reindex(columns=sectors).fillna(0.0)
    df.loc["change in money"] = -df.sum(axis=0)          # closes each column (holdings rise by what flowed in)
    df["total"] = df.sum(axis=1)
    return df


def balance_sheet_matrix(ledger: Ledger, sector_of: Callable[[str], str | None], sectors: list[str],
                         prices: dict[str, float] | None = None) -> pd.DataFrame:
    """Stocks by item and sector. Items with a price in `prices` are valued in money;
    the 'net worth' row sums every valued item."""
    prices = prices or {}
    table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (acc, item), v in ledger.balances.items():
        s = sector_of(acc)
        if s is None or abs(v) < 1e-12:
            continue
        kind = ledger.items.get(item)
        if kind == "money" or kind == "claim":
            table[item][s] += v
        elif item in prices:
            table[f"{item} (valued)"][s] += v * prices[item]
    df = pd.DataFrame(table).T.reindex(columns=sectors).fillna(0.0)
    if len(df):
        df.loc["net worth"] = df.sum(axis=0)
    df["total"] = df.sum(axis=1)
    return df
