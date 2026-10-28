"""Appendix checks: the distance is a proper transform, and clustering finds planted structure."""
import numpy as np
import pandas as pd

from src import correlation as CR


def test_mantegna_distance_endpoints():
    c = pd.DataFrame([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]], index=list("abc"), columns=list("abc"))
    d = CR.mantegna_distance(c)
    assert d.loc["a", "a"] == 0
    assert np.isclose(d.loc["a", "b"], 1 / np.sqrt(2))    # uncorrelated
    assert np.isclose(d.loc["a", "c"], 1.0)               # perfectly opposite


def _blocks(seed=1, T=240):
    """Three planted groups: each stock = its group factor + noise."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 3))
    labels = {f"S{i}": "ABC"[i % 3] for i in range(9)}
    X = {t: f[:, "ABC".index(g)] + 0.5 * rng.normal(size=T) for t, g in labels.items()}
    return pd.DataFrame(X).corr(), labels


def test_planted_clusters_are_recovered_and_order_is_a_permutation():
    corr, labels = _blocks()
    agr = CR.agreement(corr, labels, n_random=200)
    assert np.isclose(agr.table.loc[CR.METHOD, "ARI"], 1.0)
    assert agr.random_95 < 0.5
    assert sorted(CR.order(corr)) == sorted(corr.index)


def test_cut_height_gives_k_groups():
    from scipy.cluster.hierarchy import fcluster
    corr, _ = _blocks()
    Z = CR.tree(corr)
    assert len(set(fcluster(Z, CR.cut_height(Z, 3), "distance"))) == 3
