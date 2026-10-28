"""
Inputs to the optimiser (step 3a): covariance and expected returns.

Everything here is annual: monthly moments times 12 (means) and times 12
(covariances), so expected returns, volatilities and views share one unit.

Covariance
----------
With 17 assets and 58 months the sample covariance estimates 153 numbers
from 58 observations each; its smallest eigenvalues are too small and the
optimiser leans on exactly those directions. Ledoit and Wolf (2004), "Honey,
I shrunk the sample covariance matrix", blend it with a constant-correlation
target that keeps every asset's own variance and gives every pair the
average correlation. The blend weight is estimated from the data.

Expected returns
----------------
* ``historical_mean``: the sample average, as Portfolio Visualizer used. The
  transparent baseline, and the input the optimiser exploits hardest.
* ``black_litterman``: start from the returns that would make the market-cap
  weighted universe optimal (reverse optimisation), then tilt them toward
  stated views by an amount that depends on how confident each view is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import market_caps as MC

PERIODS = 12

# Reference market capitalisations on 31 Dec 2024, USD billions, entered by
# hand as approximations. They are the cross-check and the fallback only: the
# prior uses data_cache/market_caps_2024.csv (shares x price, see
# market_caps.py) whenever refresh_data.py has written it.
REFERENCE_CAP_2024 = {
    "EQIX": 91, "DLR": 60, "AMT": 86, "CCI": 39, "IRM": 31,
    "GDS": 4.5, "VNET": 1.3,
    "VRT": 43, "ANET": 139, "CSCO": 236, "MRVL": 96, "SBGSY": 143, "ETN": 131,
    "JCI": 52, "HON": 147, "GNRC": 9.2, "CAT": 175,
}
MARKET_CAP_2024 = MC.load() or REFERENCE_CAP_2024
MARKET_CAP_SOURCE = "refreshed" if MC.PATH.exists() else "reference approximations"


def historical_mean(r: pd.DataFrame) -> pd.Series:
    return r.mean() * PERIODS


def sample_cov(r: pd.DataFrame) -> pd.DataFrame:
    return r.cov() * PERIODS


def ledoit_wolf_cc(r: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Constant-correlation Ledoit-Wolf shrinkage; returns (annual covariance, intensity)."""
    X = r.to_numpy(dtype=float)
    T, N = X.shape
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / T
    var = np.diag(S).copy()
    sd = np.sqrt(var)
    rbar = ((S / np.outer(sd, sd)).sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    Y2 = Y ** 2
    pi_mat = Y2.T @ Y2 / T - S ** 2
    theta = (Y ** 3).T @ Y / T - var[:, None] * S
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(pi_mat) + rbar * np.sum((sd[None, :] / sd[:, None]) * theta)
    gamma = np.sum((F - S) ** 2)
    delta = float(np.clip((pi_mat.sum() - rho) / gamma / T, 0, 1)) if gamma > 0 else 0.0
    shrunk = delta * F + (1 - delta) * S
    # The paper scales by 1/T; rescale to the unbiased sample covariance so that
    # delta = 0 reproduces r.cov() exactly.
    shrunk *= T / (T - 1)
    return pd.DataFrame(shrunk * PERIODS, index=r.columns, columns=r.columns), delta


def market_weights(tickers, caps: dict = MARKET_CAP_2024) -> pd.Series:
    w = pd.Series({t: caps[t] for t in tickers}, dtype=float)
    return w / w.sum()


@dataclass
class View:
    """'asset' beats 'versus' by ``excess`` a year MORE than the prior already implies.

    versus='universe' compares with the market-cap weighted universe; a ticker
    makes it a pairwise view. Stating the view as a tilt beyond the
    equilibrium spread is what makes "outperforms" mean what it says: a view
    pulls the posterior spread toward its target Q, so a target set without
    reference to the prior can sit below what the prior already implies and
    move the asset down. Here Q = P Pi + excess, so a positive excess always
    moves the spread up, by less than ``excess`` when the view is uncertain."""
    asset: str
    versus: str
    excess: float
    reason: str


# The four thematic views. The original stated them as absolute returns, and
# Vertiv's "bullish" 18% sat below the 21.8% its prior already implied, so
# the view pulled it down. Here each is a tilt beyond equilibrium (see View).
VIEWS = [
    View("VRT", "universe", 0.04, "Thermal and power demand from high-density AI racks"),
    View("ANET", "universe", 0.03, "AI networking buildout"),
    View("IRM", "universe", 0.02, "Data centre pivot in a records-storage REIT"),
    View("EQIX", "GDS", 0.06, "Chinese regulatory and funding risk at GDS"),
]


@dataclass
class BLResult:
    prior: pd.Series            # equilibrium excess returns
    posterior: pd.Series        # posterior excess returns
    cov_post: pd.DataFrame      # posterior covariance, Sigma + M
    w_mkt: pd.Series
    delta: float
    tau: float
    P: pd.DataFrame
    Q: pd.Series


def black_litterman(cov: pd.DataFrame, views: list[View] = VIEWS, *, premium: float = 0.055,
                    tau: float | None = None, T: int | None = None, caps: dict = MARKET_CAP_2024) -> BLResult:
    """Black-Litterman posterior excess returns (annual).

    Prior: Pi = delta * Sigma * w_mkt, with delta set so the market-cap
    weighted universe earns ``premium`` over cash: delta = premium / (w' Sigma w).
    That states the one assumption the prior rests on, an equity-like risk
    premium, instead of borrowing delta = 2.5 from the literature.

    Views: P mu = Q + e with Q = P Pi + tilt, e ~ N(0, Omega), Omega = diag(tau P Sigma P') (He and
    Litterman, 1999): a view's uncertainty scales with the variance of the
    portfolio that expresses it. tau defaults to 1/T, the sampling
    uncertainty of a mean estimated from T observations.

    Posterior: mu = [(tau S)^-1 + P' O^-1 P]^-1 [(tau S)^-1 Pi + P' O^-1 Q],
               Sigma_post = Sigma + [(tau S)^-1 + P' O^-1 P]^-1.
    """
    names = list(cov.index)
    S = cov.to_numpy()
    w = market_weights(names, caps).to_numpy()
    delta = premium / float(w @ S @ w)
    pi = delta * S @ w
    tau = tau if tau is not None else (1.0 / T if T else 0.05)

    rows, q = [], []
    for v in views:
        if v.asset not in names or (v.versus != "universe" and v.versus not in names):
            continue
        p = np.zeros(len(names))
        p[names.index(v.asset)] = 1.0
        if v.versus == "universe":
            p -= w
        else:
            p[names.index(v.versus)] -= 1.0
        rows.append(p)
        q.append(float(p @ pi) + v.excess)        # the equilibrium spread plus the stated tilt
    if not rows:
        post, M = pi, np.zeros_like(S)
        P, Q = np.zeros((0, len(names))), np.zeros(0)
    else:
        P, Q = np.array(rows), np.array(q)
        omega = np.diag(np.diag(tau * P @ S @ P.T))
        ts_inv = np.linalg.inv(tau * S)
        M = np.linalg.inv(ts_inv + P.T @ np.linalg.inv(omega) @ P)
        post = M @ (ts_inv @ pi + P.T @ np.linalg.inv(omega) @ Q)
    labels = [f"{v.asset} vs {v.versus}" for v in views
              if v.asset in names and (v.versus == "universe" or v.versus in names)]
    return BLResult(prior=pd.Series(pi, index=names), posterior=pd.Series(post, index=names),
                    cov_post=pd.DataFrame(S + M, index=names, columns=names),
                    w_mkt=pd.Series(w, index=names), delta=delta, tau=tau,
                    P=pd.DataFrame(P, index=labels, columns=names), Q=pd.Series(Q, index=labels))
