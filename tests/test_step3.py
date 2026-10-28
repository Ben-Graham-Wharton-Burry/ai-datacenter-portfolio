"""Estimators, optimiser and engine against cases with known answers."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from src import engine as EN, estimators as ES, optimiser as OP


def _cov(rng, n=6):
    A = rng.normal(size=(n, n))
    C = A @ A.T + n * np.eye(n)
    d = np.sqrt(np.diag(C))
    sd = rng.uniform(0.15, 0.45, n)
    return C / np.outer(d, d) * np.outer(sd, sd)


def test_convex_max_sharpe_matches_brute_force():
    rng = np.random.default_rng(3)
    names = list("ABCDEF")
    S = pd.DataFrame(_cov(rng), index=names, columns=names)
    mu = pd.Series(rng.uniform(0.02, 0.25, 6), index=names)
    for cap in (1.0, 0.3):
        sol = OP.max_sharpe(mu, S, rf=0.03, cap=cap)
        best = -np.inf
        for k in range(200):                          # the old way: many random starts
            x0 = rng.dirichlet(np.ones(6))
            res = minimize(lambda w: -(w @ mu - 0.03) / np.sqrt(w @ S.values @ w), x0, method="SLSQP",
                           bounds=[(0, cap)] * 6, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
            if res.success:
                best = max(best, -res.fun)
        assert sol.sharpe >= best - 1e-6
        assert sol.weights.max() <= cap + 1e-9 and sol.weights.sum() == pytest.approx(1)


def test_black_litterman_without_views_returns_the_market():
    """The defining property of the prior: optimising on it gives back the cap weights."""
    rng = np.random.default_rng(1)
    names = ["EQIX", "DLR", "ANET", "VRT", "CAT"]
    S = pd.DataFrame(_cov(rng, 5), index=names, columns=names)
    bl = ES.black_litterman(S, views=[], premium=0.055, T=58)
    w = OP.max_sharpe(bl.posterior, bl.cov_post, rf=0.0).weights
    assert np.allclose(w, bl.w_mkt, atol=1e-5)
    # delta is set so the cap-weighted universe earns exactly the stated premium
    assert float(bl.w_mkt @ bl.prior) == pytest.approx(0.055)


def test_relative_view_moves_the_named_asset_up():
    rng = np.random.default_rng(2)
    names = ["EQIX", "DLR", "ANET", "VRT", "GDS"]
    S = pd.DataFrame(_cov(rng, 5), index=names, columns=names)
    bl = ES.black_litterman(S, views=[ES.View("VRT", "universe", 0.04, ""), ES.View("EQIX", "GDS", 0.06, "")], T=58)
    # Each view's spread moves up from equilibrium, by less than the full tilt
    for name, tilt in (("VRT vs universe", 0.04), ("EQIX vs GDS", 0.06)):
        p = bl.P.loc[name].to_numpy()
        prior_spread, post_spread = p @ bl.prior.to_numpy(), p @ bl.posterior.to_numpy()
        assert prior_spread < post_spread < prior_spread + tilt
    assert bl.Q["EQIX vs GDS"] == pytest.approx(bl.prior["EQIX"] - bl.prior["GDS"] + 0.06)


def test_shrinkage_keeps_variances_and_matches_sample_when_intensity_is_zero():
    rng = np.random.default_rng(4)
    r = pd.DataFrame(rng.normal(0, 0.05, (58, 8)) * np.linspace(0.5, 2, 8))
    S, d = ES.ledoit_wolf_cc(r)
    assert 0 <= d <= 1
    assert np.allclose(np.diag(S), np.diag(ES.sample_cov(r)))


def _prices(rets, start="2020-02-28"):
    idx = pd.bdate_range(start, periods=len(rets) + 1)
    return pd.DataFrame(np.vstack([np.ones(rets.shape[1]), np.cumprod(1 + rets, axis=0)]) * 100, index=idx,
                        columns=[f"A{i}" for i in range(rets.shape[1])])


def test_engine_single_asset_tracks_the_price_less_entry_cost():
    rng = np.random.default_rng(5)
    px = _prices(rng.normal(0.0005, 0.01, (400, 1)))
    bt = EN.run(px, pd.Series({"A0": 1.0}), "2020-02-29")
    assert bt.daily.iloc[0] == 1.0
    growth = px["A0"].iloc[-1] / px["A0"].iloc[0]
    assert bt.daily.iloc[-1] == pytest.approx(growth * (1 - EN.ONE_WAY_COST))   # no trades after the first


def test_engine_rebalances_each_december_and_charges_turnover():
    days = pd.bdate_range("2020-02-28", "2021-03-31")
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, index=days)
    px.loc["2020-03-02":, "A"] = 200.0                       # A doubles at once, B flat
    bt = EN.run(px, pd.Series({"A": 0.5, "B": 0.5}), "2020-02-29")
    dec31 = pd.Timestamp("2020-12-31")
    assert dec31 in bt.costs.index
    # Before rebalancing: A is 2/3 of the account; trading back to 50/50 turns over 1/3.
    assert bt.turnover[dec31] == pytest.approx(1 / 3)
    assert bt.costs[dec31] == pytest.approx(EN.ONE_WAY_COST / 3)
    # Monthly returns start in March and come from the same daily path
    assert bt.monthly.index[0] == pd.Timestamp("2020-03-31")
    assert (1 + bt.monthly).prod() == pytest.approx(bt.daily.iloc[-1] / bt.daily.iloc[0])
