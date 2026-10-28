"""
Portfolio optimisation (step 3b): long-only, fully invested, optional cap.

Maximum Sharpe as a convex problem
----------------------------------
max (mu - rf)'w / sqrt(w' S w) is not concave, which is why the original code
restarted SLSQP from many random points. It does not need to: substitute
y = w / k with k > 0 chosen so that (mu - rf)'y = 1. Then

    minimise  y' S y   subject to  (mu - rf)'y = 1,  y >= 0,  y_i <= cap * sum(y)

is a convex quadratic programme with one global optimum, and w = y / sum(y).
The position cap stays linear because it is homogeneous in y. At least one
asset must have mu above rf, or no long-only portfolio has a positive Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Solution:
    weights: pd.Series
    exp_return: float           # ex-ante, annual
    exp_vol: float
    sharpe: float               # ex-ante, against the rf used in the solve


def _stats(w, mu, S, rf):
    ret, vol = float(w @ mu), float(np.sqrt(w @ S @ w))
    return ret, vol, (ret - rf) / vol


def _clean(w, names, tol=1e-7):
    w = np.where(w < tol, 0.0, w)
    return pd.Series(w / w.sum(), index=names)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0, cap: float = 1.0) -> Solution:
    names = list(mu.index)
    m, S = mu.to_numpy() - rf, cov.loc[names, names].to_numpy()
    n = len(names)
    if (m <= 0).all():
        raise ValueError("No asset has an expected return above the risk-free rate.")
    cap = max(cap, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda y: m @ y - 1.0, "jac": lambda y: m}]
    if cap < 1:
        A = cap * np.ones((n, n)) - np.eye(n)                     # cap * sum(y) - y_i >= 0
        cons.append({"type": "ineq", "fun": lambda y: A @ y, "jac": lambda y: A})
    y0 = np.where(m > 0, m, 0.0)
    y0 = y0 / (m @ y0)
    res = minimize(lambda y: y @ S @ y, y0, jac=lambda y: 2 * S @ y, method="SLSQP",
                   bounds=[(0, None)] * n, constraints=cons, options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"max_sharpe did not converge: {res.message}")
    w = _clean(res.x / res.x.sum(), names)
    return Solution(w, *_stats(w.to_numpy(), mu.to_numpy(), S, rf))


def min_variance(cov: pd.DataFrame, cap: float = 1.0, mu: pd.Series | None = None, rf: float = 0.0) -> Solution:
    names = list(cov.index)
    S, n = cov.to_numpy(), len(names)
    cap = max(cap, 1.0 / n)
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w, method="SLSQP",
                   bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    w = _clean(res.x, names)
    mu_v = mu.loc[names].to_numpy() if mu is not None else np.zeros(n)
    return Solution(w, *_stats(w.to_numpy(), mu_v, S, rf))


def frontier(mu: pd.Series, cov: pd.DataFrame, cap: float = 1.0, n_points: int = 40) -> pd.DataFrame:
    """Minimum volatility for each target return, from the minimum-variance return to the top."""
    names = list(mu.index)
    m, S, n = mu.to_numpy(), cov.loc[names, names].to_numpy(), len(names)
    cap = max(cap, 1.0 / n)
    lo = min_variance(cov, cap, mu).exp_return
    # Highest return reachable under the cap: fill the best assets up to the cap.
    order, left, hi = np.argsort(-m), 1.0, 0.0
    for i in order:
        take = min(cap, left)
        hi += take * m[i]
        left -= take
        if left <= 1e-12:
            break
    out = []
    w0 = np.full(n, 1 / n)
    for target in np.linspace(lo, hi - 1e-6, n_points):
        res = minimize(lambda w: w @ S @ w, w0, jac=lambda w: 2 * S @ w, method="SLSQP",
                       bounds=[(0, cap)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                    {"type": "eq", "fun": lambda w, t=target: w @ m - t}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        if res.success:
            w0 = res.x
            out.append((float(np.sqrt(res.x @ S @ res.x)), float(target)))
    return pd.DataFrame(out, columns=["vol", "ret"])


def effective_n(w: pd.Series) -> float:
    """1 / sum(w^2): how many equal positions the portfolio is as concentrated as."""
    return float(1.0 / np.sum(np.square(w.to_numpy())))
