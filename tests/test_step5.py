"""Step 5 checks: the frontier is a lower bound on volatility, and the cap binds as intended."""
import numpy as np
import pandas as pd

from src import optimiser as OP


def _toy(seed=3, n=6):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = pd.DataFrame(A @ A.T / n * 0.04 + np.eye(n) * 0.01, index=list("ABCDEF"), columns=list("ABCDEF"))
    mu = pd.Series(rng.uniform(0.03, 0.15, n), index=cov.index)
    return mu, cov


def test_max_sharpe_point_is_not_left_of_frontier():
    mu, cov = _toy()
    for cap in (1.0, 0.3):
        sol = OP.max_sharpe(mu, cov, 0.02, cap)
        fr = OP.frontier(mu, cov, cap, n_points=200)
        # minimum frontier volatility at the solution's return, by interpolation
        v = np.interp(sol.exp_return, fr["ret"], fr["vol"])
        assert sol.exp_vol >= v - 1e-4


def test_cap_binds_and_weights_sum_to_one():
    mu, cov = _toy()
    w = OP.max_sharpe(mu, cov, 0.02, 0.25).weights
    assert abs(w.sum() - 1) < 1e-9
    assert w.max() <= 0.25 + 1e-6
    # with 6 assets and a 25% cap at least 4 names must be held
    assert (w > 1e-6).sum() >= 4
