"""
Performance metrics for the summary tables (step 2 of the rebuild).

Every exhibit's performance summary comes from ``summary``. Definitions,
agreed before coding; r_t is the monthly return, b_t the benchmark's (SPY),
rf_t the T-bill rate of the same month, T the number of months:

  End balance            V_0 * prod(1 + r_t)
  Annualised return      (V_T / V_0)^(12/T) - 1                       (CAGR)
  Expected return        12 * mean(r_t)                               (arithmetic, as Portfolio Visualizer)
  Standard deviation     sqrt(12) * sd(r_t)
  Best / worst year      compounded return per calendar year; a partial
                         first or last year is labelled with its months
  Maximum drawdown       largest fall from a running peak of month-end values
  Max drawdown (daily)   the same on daily values, when supplied: month-ends
                         hide falls that recover within the month
  Sharpe (ex-ante)       (mu_p - rf) / sigma_p from the optimiser's inputs:
                         what the model expected
  Sharpe (ex-post)       sqrt(12) * mean(r_t - rf_t) / sd(r_t - rf_t): what happened
  Sortino                sqrt(12) * mean(r_t - rf_t) / DD, with
                         DD = sqrt( (1/T) * sum min(r_t - rf_t, 0)^2 ),
                         dividing by all T months (Sortino and Price, 1994)
  Beta                   cov(r - rf, b - rf) / var(b - rf)
  CVaR 95% (monthly)     mean of the worst ceil(0.05 T) monthly returns
  Active return          12 * mean(r_t - b_t)                         (Grinold and Kahn)
  Tracking error         sqrt(12) * sd(r_t - b_t)
  Information ratio      active return / tracking error
  Benchmark correlation  corr(r_t, b_t)

Drawdowns, worst year and CVaR are reported as negative returns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ROWS = ["Start balance", "End balance", "Annualised return (CAGR)", "Expected return",
        "Standard deviation", "Best year", "Worst year", "Maximum drawdown", "Maximum drawdown (daily)",
        "Sharpe ratio (ex-ante)", "Sharpe ratio (ex-post)", "Sharpe ratio, 95% half-width", "Sortino ratio", "Beta to SPY",
        "CVaR 95% (monthly)", "Active return", "Tracking error", "Information ratio",
        "Benchmark correlation"]
PERCENT = {"Annualised return (CAGR)", "Expected return", "Standard deviation", "Best year", "Worst year",
           "Maximum drawdown", "Maximum drawdown (daily)", "CVaR 95% (monthly)", "Active return",
           "Tracking error"}
MONEY = {"Start balance", "End balance"}


def cagr(r: pd.Series, periods: int = 12) -> float:
    return float(np.prod(1 + r.values) ** (periods / len(r)) - 1)


def max_drawdown(values: pd.Series) -> float:
    """Largest fall from a running peak, as a negative number. ``values`` is a wealth path."""
    v = np.asarray(values, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def wealth(r: pd.Series, start: float = 1.0) -> pd.Series:
    """Wealth path including the starting value, one step before the first return."""
    first = r.index[0] - pd.offsets.MonthEnd(1) if isinstance(r.index, pd.DatetimeIndex) else -1
    return pd.concat([pd.Series([start], index=[first]), start * (1 + r).cumprod()])


def calendar_years(r: pd.Series) -> pd.Series:
    """Compounded return per calendar year, labelled '2021' or '2020 (Mar-Dec)' for partial years."""
    out = {}
    for y, g in r.groupby(r.index.year):
        full = len(g) == 12
        lab = str(y) if full else f"{y} ({g.index[0]:%b}-{g.index[-1]:%b})"
        out[lab] = float(np.prod(1 + g.values) - 1)
    return pd.Series(out)


def sharpe_ex_post(r: pd.Series, rf: pd.Series, periods: int = 12) -> float:
    ex = r - rf.reindex(r.index)
    sd = ex.std(ddof=1)
    return float(np.sqrt(periods) * ex.mean() / sd) if sd > 0 else np.nan


def sharpe_half_width(r: pd.Series, rf: pd.Series, periods: int = 12, z: float = 1.959964) -> float:
    """Half-width of a 95% interval for the annualised ex-post Sharpe ratio, Lo (2002), iid returns.

    For T monthly excess returns with monthly Sharpe ratio S, Var(S_hat) ~ (1 + S^2 / 2) / T.
    Annualising multiplies the ratio, and so its standard error, by sqrt(periods).
    """
    x = (r - rf.reindex(r.index)).dropna()
    T = len(x)
    if T < 3:
        return np.nan
    s_m = sharpe_ex_post(r, rf, periods) / np.sqrt(periods)
    return float(z * np.sqrt(periods) * np.sqrt((1 + 0.5 * s_m ** 2) / T))


def sortino(r: pd.Series, rf: pd.Series, periods: int = 12) -> float:
    ex = r - rf.reindex(r.index)
    dd = np.sqrt(np.mean(np.minimum(ex.values, 0.0) ** 2))
    return float(np.sqrt(periods) * ex.mean() / dd) if dd > 0 else np.nan


def beta(r: pd.Series, b: pd.Series, rf: pd.Series) -> float:
    x, y = b - rf.reindex(b.index), r - rf.reindex(r.index)
    return float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))


def cvar(r: pd.Series, level: float = 0.95) -> float:
    k = max(1, math.ceil(round((1 - level) * len(r), 9)))   # round first: 0.05 * 60 is 3.0000000000000027 in floating point
    return float(np.sort(r.values)[:k].mean())


def summary(r: pd.Series, rf: pd.Series, bench: pd.Series | None = None, *, start: float = 10_000,
            daily_values: pd.Series | None = None, ex_ante: tuple[float, float] | None = None,
            periods: int = 12) -> tuple[pd.Series, dict]:
    """One column of a performance summary, plus labels (which year was best and worst).

    ``ex_ante`` is (expected annual return, expected annual volatility[, rf]) from
    the optimiser's inputs; without rf the ex-ante Sharpe uses the average
    risk-free rate of the same window. Benchmark-relative rows are NaN when ``bench`` is None.
    """
    r = r.dropna()
    rf = rf.reindex(r.index)
    years = calendar_years(r)
    s = pd.Series(np.nan, index=ROWS, dtype=float)
    s["Start balance"] = start
    s["End balance"] = start * float(np.prod(1 + r.values))
    s["Annualised return (CAGR)"] = cagr(r, periods)
    s["Expected return"] = periods * r.mean()
    s["Standard deviation"] = np.sqrt(periods) * r.std(ddof=1)
    s["Best year"], s["Worst year"] = years.max(), years.min()
    s["Maximum drawdown"] = max_drawdown(wealth(r))
    if daily_values is not None:
        # Start at the last trading day on or before the month-end the monthly path starts from
        # (28 Feb 2020 is a Friday; slicing from the 29th would skip it and the day's value).
        d = daily_values.dropna()
        first = d.index[d.index <= r.index[0] - pd.offsets.MonthEnd(1)]
        start_day = first[-1] if len(first) else d.index[0]
        s["Maximum drawdown (daily)"] = max_drawdown(d.loc[start_day:r.index[-1]])
    if ex_ante is not None:
        # (mu, sigma) uses this window's average rf; (mu, sigma, rf) fixes the rf the model
        # used when it was estimated, so the belief stays the same in every table.
        mu, sig, *rf0 = ex_ante
        s["Sharpe ratio (ex-ante)"] = (mu - (rf0[0] if rf0 else periods * rf.mean())) / sig
    s["Sharpe ratio (ex-post)"] = sharpe_ex_post(r, rf, periods)
    s["Sharpe ratio, 95% half-width"] = sharpe_half_width(r, rf, periods)
    s["Sortino ratio"] = sortino(r, rf, periods)
    s["CVaR 95% (monthly)"] = cvar(r)
    if bench is not None:
        b = bench.reindex(r.index)
        a = r - b
        s["Beta to SPY"] = beta(r, b, rf)
        s["Active return"] = periods * a.mean()
        s["Tracking error"] = np.sqrt(periods) * a.std(ddof=1)
        s["Information ratio"] = s["Active return"] / s["Tracking error"] if s["Tracking error"] > 0 else np.nan
        s["Benchmark correlation"] = float(np.corrcoef(r, b)[0, 1])
    labels = {"Best year": years.idxmax(), "Worst year": years.idxmin(),
              "window": f"{r.index[0]:%b %Y} to {r.index[-1]:%b %Y}, {len(r)} months"}
    return s, labels


def summary_table(series: dict, rf: pd.Series, bench_name: str = "SPY", **kw) -> tuple[pd.DataFrame, dict]:
    """Columns for each named return series; the benchmark column gets beta 1 and no active rows.

    ``kw`` may hold start=<balance> and per-series dicts: daily_values={name: path},
    ex_ante={name: (mu, sigma[, rf])}.
    """
    bench = series[bench_name]
    cols, labels = {}, {}
    for name, r in series.items():
        is_b = name == bench_name
        s, lab = summary(r, rf, None if is_b else bench, start=kw.get("start", 10_000),
                         daily_values=kw.get("daily_values", {}).get(name),
                         ex_ante=kw.get("ex_ante", {}).get(name))
        if is_b:
            s["Beta to SPY"], s["Benchmark correlation"] = 1.0, 1.0
        cols[name], labels[name] = s, lab
    return pd.DataFrame(cols), labels


DISPLAY_NAMES = {"Sharpe ratio, 95% half-width": "Sharpe ratio, 95% interval"}


def formatted(table: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Strings for display: money, percentages, ratios, the best and worst year named, N/A where empty.

    The Sharpe half-width is shown as an interval around the ex-post Sharpe ratio; rename
    the index with DISPLAY_NAMES when the table is drawn.
    """
    out = pd.DataFrame(index=table.index, columns=table.columns, dtype=object)
    for c in table:
        for row, v in table[c].items():
            if pd.isna(v):
                out.loc[row, c] = "N/A"
            elif row == "Sharpe ratio, 95% half-width":
                sr = table.loc["Sharpe ratio (ex-post)", c]
                out.loc[row, c] = f"{sr - v:.2f} to {sr + v:.2f}"
            elif row in MONEY:
                out.loc[row, c] = f"${v:,.0f}"
            elif row in PERCENT:
                out.loc[row, c] = f"{v:.2%}" + (f" ({labels[c][row]})" if row in ("Best year", "Worst year") else "")
            else:
                out.loc[row, c] = f"{v:.2f}"
    return out
