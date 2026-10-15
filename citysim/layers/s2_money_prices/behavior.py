"""Stage 2 behavioural rules: pure, documented, swappable functions.

Nothing here touches the world or the ledger. The market layer calls these and
posts the resulting flows. All functions work on scalars or numpy arrays.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Expectations and plans
# ---------------------------------------------------------------------------

def adaptive(expected, observed, speed):
    """Adaptive expectations: E' = E + speed * (observed - E)."""
    return expected + speed * (observed - expected)


def production_target(expected_sales, inventory, target_ticks, correction_ticks, cap=np.inf):
    """Produce expected sales plus a correction towards target inventory (target_ticks * sales)."""
    want = expected_sales + (target_ticks * expected_sales - inventory) / correction_ticks
    return np.clip(want, 0.0, cap)


# ---------------------------------------------------------------------------
# Prices and wages
# ---------------------------------------------------------------------------

def inventory_price(price, inventory, target, speed, max_step):
    """Raise the price when stocks are below target, cut it when above.

    p' = p * (1 + clip(speed * (target - inventory) / target, -max_step, max_step))
    """
    gap = np.where(target > 0, (target - inventory) / np.maximum(target, 1e-12), 0.0)
    return price * (1.0 + np.clip(speed * gap, -max_step, max_step))


def inventory_gap(inventory, target):
    """+1 when shelves are empty, 0 at target, -1 at twice the target or more."""
    if target <= 1e-12:
        return 0.0 if inventory <= 1e-12 else -1.0
    return float(np.clip((target - inventory) / target, -1.0, 1.0))


def order_gap(orders, supply):
    """For services: +1 when nothing could be supplied, -1 when nobody ordered."""
    base = max(orders, supply, 1e-12)
    return float(np.clip((orders - supply) / base, -1.0, 1.0))


def markup_update(markup, gap, speed, normal, pull, lo, hi):
    """Markups rise when shelves are short and fall when they overflow, and drift back
    towards the normal markup: m' = m + speed * gap - pull * (m - normal)."""
    return float(np.clip(markup + speed * gap - pull * (markup - normal), lo, hi))


def markup_price(unit_cost, markup):
    """Normal-cost pricing: price = (1 + markup) * unit cost."""
    return (1.0 + markup) * unit_cost


def excess_demand_price(price, demand, supply, speed, max_step):
    """For services with no inventory: p' = p * (1 + clip(speed * (D - S) / max(S, D), ...))."""
    base = np.maximum(np.maximum(supply, demand), 1e-12)
    return price * (1.0 + np.clip(speed * (demand - supply) / base, -max_step, max_step))


def wage_update(wage, labor_demand, labor_supply, speed, max_step, normal_unemployment=0.0):
    """Wages rise with unfilled vacancies and fall when unemployment in that occupation
    is above its normal (frictional) level."""
    effective = labor_supply * (1.0 - normal_unemployment)
    base = np.maximum(np.maximum(effective, labor_demand), 1e-12)
    return wage * (1.0 + np.clip(speed * (labor_demand - effective) / base, -max_step, max_step))


# ---------------------------------------------------------------------------
# Occupational choice
# ---------------------------------------------------------------------------

def expected_income(wage, labor_demand, labor_supply):
    """Wage times the chance of being hired in that occupation."""
    p_hire = np.where(labor_supply > 1e-12, np.minimum(1.0, labor_demand / np.maximum(labor_supply, 1e-12)), 1.0)
    return wage * p_hire


def mobility(pools, income, jobs, share_moving, sensitivity):
    """A share of every occupation reconsiders each tick. Movers pick a new occupation
    with probability proportional to jobs on offer * expected income ** sensitivity
    (deterministic flows). With equal incomes and every job filled, nothing changes.

    Returns the new pools (same total).
    """
    pools = np.asarray(pools, float)
    total = pools.sum()
    if total <= 0:
        return pools
    w = np.maximum(np.asarray(jobs, float), 1e-9) * np.maximum(np.asarray(income, float), 0.0) ** sensitivity
    w = w / w.sum() if w.sum() > 0 else pools / total
    movers = share_moving * pools
    return pools - movers + movers.sum() * w


# ---------------------------------------------------------------------------
# Households
# ---------------------------------------------------------------------------

def consumption_spending(income, cash, price, need, c_max, alpha_income, alpha_wealth):
    """Nominal spending on bread (Godley-Lavoie style with a subsistence floor).

    planned = alpha_income * income + alpha_wealth * cash_before_income
    People first try to afford their need (need * price), and never buy more than
    c_max per person. Spending can't exceed the cash they hold.
    Returns nominal spending.
    """
    planned = alpha_income * income + alpha_wealth * np.maximum(cash - income, 0.0)
    planned = np.maximum(planned, need * price)            # buy what you need if you can
    planned = np.minimum(planned, c_max * price)           # satiation
    return np.clip(planned, 0.0, cash)


# ---------------------------------------------------------------------------
# Splitting a market among several sellers (used from Stage 4 with many firms)
# ---------------------------------------------------------------------------

def allocate_by_price(quantity, prices, stocks, choosiness):
    """Split `quantity` across sellers: shares proportional to price ** -choosiness,
    no seller sells more than its stock; unmet demand is passed on to the others.
    Returns sales per seller."""
    prices = np.asarray(prices, float)
    stocks = np.maximum(np.asarray(stocks, float), 0.0)
    sales = np.zeros_like(stocks)
    left = float(quantity)
    open_ = stocks > 1e-12
    for _ in range(len(stocks) + 1):
        if left <= 1e-12 or not open_.any():
            break
        w = np.where(open_, np.maximum(prices, 1e-12) ** (-choosiness), 0.0)
        share = left * w / w.sum()
        take = np.minimum(share, stocks - sales)
        sales += take
        left -= take.sum()
        open_ = (stocks - sales) > 1e-12
    return sales


# ---------------------------------------------------------------------------
# Barter: the double coincidence of wants
# ---------------------------------------------------------------------------

def trade_completion(n_goods, meetings, exchange):
    """Share of desired trades that get done in one tick.

    With money there are shops: anyone who wants a good finds a seller who takes
    money, so every desired trade happens (1.0).
    With barter, a trader meets random partners. A meeting is useful only if the
    partner has what you want (1 in n-1) AND wants what you have (1 in n-1).
    After `meetings` tries: 1 - (1 - 1/(n-1)^2) ** meetings.
    """
    if exchange == "money":
        return 1.0
    p = 1.0 / max(n_goods - 1, 1) ** 2
    return 1.0 - (1.0 - p) ** meetings
