"""Portfolio optimisation with shrinkage estimators."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Optional, Tuple

def shrink_covariance(returns: pd.DataFrame, method='ledoit_wolf') -> np.ndarray:
    if method == 'ledoit_wolf':
        return LedoitWolf().fit(returns.values).covariance_
    return returns.cov().values

def portfolio_stats(w, mu, cov, rf=0.045, periods_per_year=12):
    ret = np.sum(mu * w) * periods_per_year
    vol = np.sqrt(w @ (cov * periods_per_year) @ w)
    sr  = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sr

def max_sharpe_portfolio(mu, cov, rf=0.045, max_weight=None, prev_weights=None,
                         turnover_penalty=0.0, n_restarts=5, seed=0):
    """
    Long-only tangency portfolio by multi-start SLSQP.

    Restarts are drawn from a seeded generator. An earlier version used the
    global NumPy random state, so the walk-forward results changed on every
    execution and the figures in the notebook could not be reproduced.

    If the universe is too small for the position cap to be feasible
    (n * max_weight < 1), the cap is relaxed to 1/n rather than letting the
    optimiser fail and fall back silently.
    """
    n = len(mu)
    rng = np.random.default_rng(seed)
    ub = max(max_weight, 1.0 / n) if max_weight else 1.0
    bounds = tuple((0.0, ub) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    def objective(w):
        _, _, sr = portfolio_stats(w, mu, cov, rf)
        cost = turnover_penalty * np.sum(np.abs(w - prev_weights)) if (turnover_penalty > 0 and prev_weights is not None) else 0.0
        return -sr + cost
    best = None
    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(objective, w0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    return best.x

def min_var_portfolio(cov, max_weight=None):
    n = cov.shape[0]
    ub = max(max_weight, 1.0 / n) if max_weight else 1.0
    bounds = tuple((0.0, ub) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    res = minimize(lambda w: w @ (cov * 12) @ w, np.ones(n)/n, method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res.x

def efficient_frontier(mu, cov, n_pts=80):
    w_mv = min_var_portfolio(cov)
    ret_min = np.sum(mu * w_mv) * 12
    ret_max = np.max(mu) * 12 * 0.95
    targets = np.linspace(ret_min, ret_max, n_pts)
    vols = []
    n = len(mu)
    bounds = tuple((0, 1) for _ in range(n))
    for tgt in targets:
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, t=tgt: np.sum(w * mu) * 12 - t}]
        res = minimize(lambda w: np.sqrt(w @ (cov * 12) @ w), np.ones(n)/n,
                       method='SLSQP', bounds=bounds, constraints=cons)
        vols.append(res.fun if res.success else np.nan)
    return targets, np.array(vols)

def black_litterman(Sigma, w_mkt, P, Q, tau=0.05, delta=2.5):
    pi = delta * Sigma @ w_mkt
    Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))
    tS_inv = np.linalg.inv(tau * Sigma)
    O_inv  = np.linalg.inv(Omega)
    M = np.linalg.inv(tS_inv + P.T @ O_inv @ P)
    mu_bl = M @ (tS_inv @ pi + P.T @ O_inv @ Q)
    Sigma_bl = Sigma + M
    return pi, mu_bl, Sigma_bl
