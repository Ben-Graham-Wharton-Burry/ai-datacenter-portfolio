"""
Backtest engine (step 3c): fixed target weights, rebalanced once a year.

Between rebalances each holding simply drifts with its own daily price, as a
real account would; on the last trading day of each December the account
trades back to the target weights. Trading costs 15 bps per round trip,
i.e. 7.5 bps on every unit bought or sold:

    cost = 0.00075 * sum_i |w_target,i - w_drifted,i|   (as a share of value)

and the first purchase counts as trading from cash. The daily value path is
what the daily drawdown uses; monthly returns are taken from its month-end
values, so monthly and daily figures always describe the same portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ONE_WAY_COST = 0.00075


@dataclass
class Backtest:
    daily: pd.Series             # account value, starting at 1 on the start date
    monthly: pd.Series           # monthly returns from month-end values
    costs: pd.Series             # cost paid at each rebalance, as a share of value
    turnover: pd.Series          # sum |trade| at each rebalance


def run(prices: pd.DataFrame, weights: pd.Series, start: str, end: str | None = None,
        cost: float = ONE_WAY_COST) -> Backtest:
    """Annually rebalanced backtest of ``weights`` from the last trading day on or
    before ``start`` (a month-end) to ``end``."""
    names = [t for t in weights.index if weights[t] > 0]
    w_t = weights[names] / weights[names].sum()
    px = prices[names].loc[:end].dropna(how="all")
    days = px.index[px.index <= pd.Timestamp(start)]
    if not len(days):
        raise ValueError("No prices on or before the start date.")
    px = px.loc[days[-1]:]
    if px.iloc[0].isna().any():
        raise ValueError(f"No starting price for {list(px.columns[px.iloc[0].isna()])}.")
    px = px.ffill()                                   # a missing day (holiday) leaves the holding unchanged
    rel = (px / px.shift(1)).iloc[1:]                 # daily price relatives

    year_end = px.index.to_series().groupby(px.index.year).max()
    rebal = set(year_end.values) - {px.index[-1]}

    holdings = w_t.to_numpy() * (1 - cost)            # first purchase from cash: turnover 1
    # The account is worth 1 before the first trade; the purchase cost shows up in the first return.
    values, costs, turn = [1.0], {px.index[0]: cost}, {px.index[0]: 1.0}
    for day, r in zip(rel.index, rel.to_numpy()):
        holdings = holdings * r
        v = holdings.sum()
        if day in rebal:
            drift = holdings / v
            t = float(np.abs(w_t.to_numpy() - drift).sum())
            c = cost * t
            v *= (1 - c)
            holdings = w_t.to_numpy() * v
            costs[day], turn[day] = c, t
        values.append(v)
    daily = pd.Series(values, index=px.index, name="value")
    month_end = daily.groupby([daily.index.year, daily.index.month]).tail(1)
    month_end.index = month_end.index + pd.offsets.MonthEnd(0)
    # The first value belongs to the starting month-end; returns begin the month after.
    monthly = month_end.pct_change().iloc[1:]
    first_full = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    monthly = monthly.loc[monthly.index > first_full]
    return Backtest(daily=daily, monthly=monthly.rename("portfolio"),
                    costs=pd.Series(costs), turnover=pd.Series(turn))
