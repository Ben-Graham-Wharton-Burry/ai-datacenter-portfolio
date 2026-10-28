"""
The data layer for the rebuild: one place where prices become the monthly
returns every exhibit uses.

Steps, in order:
  1. Load the frozen daily price snapshot (data_cache/prices_daily.csv), so
     every run reproduces the same numbers until the snapshot is refreshed.
  2. Blank out prices that belong to a different company than the ticker now
     represents (config.CORPORATE_EVENTS; today only Vertiv's SPAC shell).
  3. Month-end prices to monthly simple returns. A final month that is not
     yet complete is dropped, so a refresh in mid-October cannot produce a
     half-month "October return".
  4. The monthly one-month T-bill rate from the Fama-French file as the
     risk-free rate, month by month.
  5. Windows: estimation March 2020 to December 2024 (config.EST_*), and out
     of sample from January 2025 to the last complete month.

Monthly returns are used throughout, as in the original analysis: daily
returns of US and Asian listings are not synchronous, and monthly returns
also match the annual rebalancing and the Monte Carlo step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

CACHE = Path(__file__).resolve().parents[1] / "data_cache"


@dataclass
class Dataset:
    prices: pd.DataFrame          # daily, after corrections
    monthly: pd.DataFrame         # monthly simple returns, month-end index
    rf: pd.Series                 # monthly risk-free rate, same index
    log: list = field(default_factory=list)   # every correction applied, in words

    # ---- windows --------------------------------------------------------
    def window(self, start: str, end: str | None = None, cols=None) -> pd.DataFrame:
        cols = list(cols) if cols is not None else C.SINGLE_NAMES
        return self.monthly.loc[start:end, cols]

    def est(self, cols=None) -> pd.DataFrame:
        """Estimation window: complete monthly returns, March 2020 to December 2024."""
        return self.window(C.EST_START, C.EST_END, cols)

    def oos(self, cols=None) -> pd.DataFrame:
        """Out of sample: January 2025 to the last complete month."""
        return self.window(C.OOS_START, None, cols)

    def excess(self, r: pd.DataFrame) -> pd.DataFrame:
        """Returns above the risk-free rate of the same month."""
        return r.sub(self.rf.reindex(r.index), axis=0)


def load_prices(cache: Path = CACHE) -> pd.DataFrame:
    p = pd.read_csv(Path(cache) / "prices_daily.csv", index_col=0, parse_dates=True).sort_index()
    p.index = pd.DatetimeIndex(p.index).tz_localize(None)
    return p


def apply_corporate_events(prices: pd.DataFrame, events: dict = C.CORPORATE_EVENTS) -> tuple[pd.DataFrame, list]:
    """Remove prices from before a ticker represented its current company."""
    p, log = prices.copy(), []
    for t, (first_day, why) in events.items():
        if t not in p:
            continue
        cut = p.index < pd.Timestamp(first_day)
        n = int(p.loc[cut, t].notna().sum())
        p.loc[cut, t] = np.nan
        log.append(f"{t}: {n} daily prices before {first_day} removed. {why}.")
    return p, log


def monthly_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Month-end prices to simple returns; an incomplete final month is dropped."""
    log = []
    last_day = prices.index[-1]
    month_end_bday = (last_day + pd.offsets.BMonthEnd(0)).normalize()
    m = prices.resample("ME").last()
    if last_day.normalize() < month_end_bday:
        log.append(f"Prices end on {last_day:%d %b %Y}, before that month is over; "
                   f"{last_day:%B %Y} is left out.")
        m = m.iloc[:-1]
    # fill_method=None: a missing month-end price gives a missing return, never a 0%.
    return m.pct_change(fill_method=None).iloc[1:], log


def risk_free(index: pd.DatetimeIndex, cache: Path = CACHE) -> tuple[pd.Series, list]:
    """Monthly one-month T-bill return from the Fama-French file, on ``index``."""
    ff = pd.read_csv(Path(cache) / "ff5_monthly.csv", index_col=0, parse_dates=True)
    rf = ff["RF"]
    rf.index = rf.index + pd.offsets.MonthEnd(0)
    out = rf.reindex(index)
    log = []
    if out.isna().any():
        missing = out.index[out.isna()]
        log.append(f"Risk-free rate missing for {len(missing)} month(s) from {missing[0]:%b %Y}; "
                   "the last published rate is carried forward. Refresh the Fama-French file to fill them.")
        out = out.ffill()
    return out.rename("RF"), log


def build(cache: Path = CACHE) -> Dataset:
    raw = load_prices(cache)
    prices, log = apply_corporate_events(raw)
    monthly, l2 = monthly_returns(prices)
    rf, l3 = risk_free(monthly.index, cache)
    log += l2 + l3
    est = monthly.loc[C.EST_START:C.EST_END, C.SINGLE_NAMES]
    gaps = est.columns[est.isna().any()]
    if len(gaps):
        raise ValueError(f"Missing estimation-window returns for {list(gaps)}; check the snapshot.")
    return Dataset(prices=prices, monthly=monthly, rf=rf, log=log)


def report(ds: Dataset, raw: pd.DataFrame | None = None) -> str:
    """A short account of what the data layer did and why it matters."""
    raw = load_prices() if raw is None else raw
    lines = ["Data corrections:"] + [f"  - {x}" for x in ds.log]
    # What the Vertiv correction changes: its volatility in the months it affects.
    vrt_raw = raw["VRT"].resample("ME").last().pct_change(fill_method=None)
    pre = vrt_raw.loc["2019-02":"2020-01"]
    lines.append(f"  - VRT before the merger (Feb 2019 to Jan 2020): annualised volatility "
                 f"{pre.std() * np.sqrt(12):.1%}, against {ds.monthly['VRT'].loc[:C.EST_END].std() * np.sqrt(12):.1%} "
                 "for Vertiv itself in the estimation window.")
    est, oos = ds.est(), ds.oos()
    rf_est = ds.rf.loc[est.index]
    lines += [
        "Windows:",
        f"  - Estimation: {est.index[0]:%b %Y} to {est.index[-1]:%b %Y}, {len(est)} months, "
        f"{est.shape[1]} single names, no gaps.",
        f"  - Out of sample: {oos.index[0]:%b %Y} to {oos.index[-1]:%b %Y}, {len(oos)} months.",
        "Risk-free rate:",
        f"  - Average {rf_est.mean() * 12:.2%} a year over the estimation window "
        f"(lowest month {rf_est.min() * 12:.2%}, highest {rf_est.max() * 12:.2%} annualised), "
        "against the fixed 4.50% the original analysis assumed.",
    ]
    return "\n".join(lines)
