"""Monte Carlo checks: inputs are reproduced, the copula behaves, the bookkeeping is exact."""
import numpy as np
import pandas as pd
import pytest

from src import montecarlo as MC

NAMES = ["A", "B", "C"]


def _params(nu=5.0, nu_c=5.0, rho=0.5):
    corr = pd.DataFrame(np.full((3, 3), rho) + np.eye(3) * (1 - rho), index=NAMES, columns=NAMES)
    return MC.Params(sigma=pd.Series([0.05, 0.08, 0.10], index=NAMES),
                     nu=pd.Series([nu] * 3, index=NAMES), corr=corr, nu_c=nu_c)


def test_shocks_have_unit_variance_and_target_correlation():
    z = MC.draw(_params(), 200_000, np.random.default_rng(1))
    assert np.allclose(z.std(axis=0), 1, atol=0.02)
    # with Student-t marginals the linear correlation is close to, not exactly, the copula correlation
    assert np.allclose(np.corrcoef(z.T)[np.triu_indices(3, 1)], 0.5, atol=0.05)


def test_t_copula_has_more_joint_crashes_than_gaussian():
    p, rng = _params(nu=30, nu_c=3), np.random.default_rng(2)
    zt, zg = MC.draw(p, 200_000, rng), MC.draw(p, 200_000, rng, gaussian=True)
    q = -2.33                                                     # roughly the 1% tail
    joint = lambda z: np.mean((z < q).all(axis=1))
    assert joint(zt) > 1.5 * joint(zg)


def test_simulation_without_risk_compounds_exactly():
    p = _params()
    p.sigma[:] = 0.0
    w = pd.Series([0.2, 0.3, 0.5], index=NAMES)
    res = MC.simulate(p, w, pd.Series(0.12, index=NAMES), years=3, n_paths=5, start=100.0, cost=0.0)
    assert np.allclose(res.terminal, 100 * 1.01 ** 36)
    assert np.allclose(res.max_drawdown, 0)


def test_monthly_means_match_the_drift_scenario():
    p, w = _params(), pd.Series(1 / 3, index=NAMES)
    mu = pd.Series([0.06, 0.09, 0.12], index=NAMES)
    res = MC.simulate(p, w, mu, years=1, n_paths=40_000, start=1.0, cost=0.0)
    # one year, no rebalancing inside it: expected value is the weighted compounded drift
    expected = float((w * (1 + mu / 12) ** 12).sum())
    assert res.terminal.mean() == pytest.approx(expected, rel=0.004)


def test_copula_dof_is_recovered_roughly():
    p = _params(nu=30, nu_c=4, rho=0.6)
    z = MC.draw(p, 3_000, np.random.default_rng(3))
    nu_c, ll = MC.fit_copula_nu(MC.pseudo_obs(pd.DataFrame(z)), p.corr.to_numpy())
    assert 2.5 <= nu_c <= 8
