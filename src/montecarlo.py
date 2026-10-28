"""
Monte Carlo projection of the portfolios: Student-t marginals joined by a Student-t copula.

Why this model
--------------
A portfolio's bad outcomes come from two things a Gaussian simulation
understates: single assets have fat tails (a -20% month is far likelier than
a normal distribution allows), and assets crash together (tail dependence).
Sklar's theorem lets the two be modelled separately:

* marginals: each asset's monthly return is mu_i + sigma_i * z_i, where z_i is
  a Student-t variable rescaled to unit variance, with its own degrees of
  freedom nu_i fitted to that asset's history;
* dependence: the z_i are tied together by a Student-t copula with
  correlation R and degrees of freedom nu_c. Unlike the Gaussian copula it
  gives joint extremes positive probability (tail dependence).

Mean and volatility are therefore exactly the inputs, and only the shape of
the distribution changes; that isolates what fat tails do to the outcomes.

Returns are simulated as simple returns, floored at -95% a month, rather than
as log returns: with Student-t log returns the expectation of exp(x) does not
exist, so the mean of the simulated wealth would be unstable.

Expected returns are the uncertain input, so the drift is a scenario passed in
by the caller, while volatilities, nu_i, R and nu_c are estimated once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

FLOOR = -0.95


@dataclass
class Params:
    sigma: pd.Series          # monthly volatility per asset
    nu: pd.Series             # marginal degrees of freedom per asset
    corr: pd.DataFrame        # copula correlation
    nu_c: float               # copula degrees of freedom


def fit_marginal_nu(x: pd.Series, lo: float = 3.0, hi: float = 30.0) -> float:
    """Degrees of freedom of a Student-t fitted by maximum likelihood (location and scale free)."""
    nu, _, _ = stats.t.fit(x.dropna().to_numpy())
    return float(np.clip(nu, lo, hi))


def pseudo_obs(r: pd.DataFrame) -> np.ndarray:
    """Ranks scaled into (0, 1): the copula sees only the order of the data, not its marginal shape."""
    return (r.rank().to_numpy() - 0.5) / len(r)


def fit_copula_nu(u: np.ndarray, corr: np.ndarray, grid=None) -> tuple[float, pd.Series]:
    """Copula degrees of freedom by profile likelihood over a grid, with the correlation held fixed."""
    grid = grid if grid is not None else np.array([2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100])
    ll = {}
    d = u.shape[1]
    for nu in grid:
        x = stats.t.ppf(u, nu)
        joint = stats.multivariate_t(loc=np.zeros(d), shape=corr, df=nu).logpdf(x)
        ll[float(nu)] = float(joint.sum() - stats.t.logpdf(x, nu).sum())
    ll = pd.Series(ll)
    return float(ll.idxmax()), ll


def fit(r: pd.DataFrame, corr: pd.DataFrame) -> Params:
    """Estimate everything except the drift. ``corr`` is the copula correlation (the shrunk one in practice)."""
    nu = r.apply(fit_marginal_nu)
    nu_c, _ = fit_copula_nu(pseudo_obs(r), corr.loc[r.columns, r.columns].to_numpy())
    return Params(sigma=r.std(ddof=1), nu=nu, corr=corr.loc[r.columns, r.columns], nu_c=nu_c)


def draw(p: Params, n: int, rng: np.random.Generator, gaussian: bool = False) -> np.ndarray:
    """n joint draws of unit-variance shocks z, shape (n, assets)."""
    d = len(p.sigma)
    L = np.linalg.cholesky(p.corr.to_numpy())
    e = rng.standard_normal((n, d)) @ L.T
    if gaussian:
        return e
    w = rng.chisquare(p.nu_c, size=(n, 1))
    u = special.stdtr(p.nu_c, e * np.sqrt(p.nu_c / w))         # t copula: common chi-square mixing
    u = np.clip(u, 1e-12, 1 - 1e-12)
    nu = p.nu.to_numpy()
    return special.stdtrit(nu, u) * np.sqrt((nu - 2) / nu)      # Student-t marginals, unit variance


def shocks(p: Params, months: int, n_paths: int, seed: int = 7, gaussian: bool = False) -> np.ndarray:
    """All shocks for one projection, shape (months, paths, assets).

    Drawing them once and reusing them for every portfolio and drift scenario (common random
    numbers) means that differences between scenarios come from the inputs, not from sampling luck.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((months, n_paths, len(p.sigma)), dtype=np.float32)
    for m in range(months):
        out[m] = draw(p, n_paths, rng, gaussian)
    return out


@dataclass
class Result:
    terminal: np.ndarray        # nominal end value per path
    max_drawdown: np.ndarray    # worst month-end fall from peak per path
    paths: np.ndarray           # portfolio value at each month end, shape (months + 1, n_paths)


def simulate(p: Params, weights: pd.Series, mu_annual: pd.Series, *, years: int = 10, n_paths: int = 10_000,
             start: float = 500_000, cost: float = 0.00075, seed: int = 7, gaussian: bool = False,
             z: np.ndarray | None = None) -> Result:
    """Monthly simulation with annual rebalancing back to ``weights`` and proportional costs.

    ``mu_annual`` is each asset's expected simple return per year; the monthly mean is mu / 12,
    matching how the document annualises arithmetic means. Pass ``z`` from ``shocks`` to reuse draws.
    """
    names = list(p.sigma.index)
    w = weights.reindex(names).fillna(0.0).to_numpy()
    mu_m = mu_annual.reindex(names).to_numpy() / 12
    sig = p.sigma.to_numpy()
    months = years * 12
    if z is None:
        z = shocks(p, months, n_paths, seed, gaussian)
    n_paths = z.shape[1]
    hold = np.outer(np.full(n_paths, start * (1 - cost)), w)       # buy at the start, paying one-way cost
    paths = np.empty((months + 1, n_paths))
    paths[0] = hold.sum(axis=1)
    peak = paths[0].copy()
    mdd = np.zeros(n_paths)
    for m in range(1, months + 1):
        r = np.maximum(mu_m + sig * z[m - 1], FLOOR)
        hold *= 1 + r
        v = hold.sum(axis=1)
        if m % 12 == 0 and m < months:                              # rebalance at each year end
            target = np.outer(v, w)
            turnover = np.abs(target - hold).sum(axis=1) / 2
            v = v - cost * 2 * turnover                             # sells and buys both pay one-way cost
            hold = np.outer(v, w)
        paths[m] = v
        peak = np.maximum(peak, v)
        mdd = np.minimum(mdd, v / peak - 1)
    return Result(terminal=paths[-1], max_drawdown=mdd, paths=paths)


def summarise(res: Result, start: float, years: int, inflation: float) -> dict:
    """The numbers the document reports for one portfolio and one drift scenario."""
    deflator = (1 + inflation) ** years
    real = res.terminal / deflator
    return {"P(nominal loss)": float((res.terminal < start).mean()),
            "P(real loss)": float((real < start).mean()),
            "Median real end value": float(np.median(real)),
            "5th percentile real": float(np.percentile(real, 5)),
            "95th percentile real": float(np.percentile(real, 95)),
            "Mean nominal end value": float(res.terminal.mean()),
            "Median worst drawdown": float(np.median(res.max_drawdown))}
