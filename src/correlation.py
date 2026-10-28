"""
Correlation structure of the universe (appendix).

Distance
--------
Two stocks are close when their returns move together. The Mantegna (1999)
distance d_ij = sqrt((1 - rho_ij) / 2) turns a correlation into a proper
metric: 0 for perfectly correlated stocks, 1/sqrt(2) for uncorrelated ones
and 1 for perfectly opposite ones. It equals half the Euclidean distance
between the two standardised return series, which is why every linkage
method below, Ward included, is valid on it.

Linkage
-------
Hierarchical clustering merges the two closest groups, then repeats. The
linkage method decides how far apart two *groups* are:

* single:   the closest pair across the groups (chains outliers onto groups)
* complete: the farthest pair (compact groups, but one outlier can dominate)
* average:  the mean of all cross-group pairs (between the two)
* ward:     the merge that adds the least within-group variance

The method used in the document is AVERAGE linkage, fixed before the results
were compared. The other three are reported only as a sensitivity check.

Agreement with our clusters
---------------------------
The tree is cut into as many groups as we have clusters and compared with our
labels by the adjusted Rand index (ARI): 1 for identical groupings, about 0
for a random one. Because 17 stocks in three clusters can match by luck, the
ARI is compared with the same statistic for randomly shuffled labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from . import estimators as ES

METHOD = "average"                       # committed choice, see the module docstring
METHODS = ("single", "average", "complete", "ward")


def mantegna_distance(corr: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(0.5 * (1.0 - corr.clip(-1, 1).to_numpy()))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def tree(corr: pd.DataFrame, method: str = METHOD) -> np.ndarray:
    """Linkage matrix; optimal leaf ordering puts similar stocks next to each other."""
    return linkage(squareform(mantegna_distance(corr).to_numpy(), checks=False), method, optimal_ordering=True)


def order(corr: pd.DataFrame, method: str = METHOD) -> list[str]:
    return [corr.index[i] for i in leaves_list(tree(corr, method))]


def shrunk_corr(r: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Correlation implied by the constant-correlation Ledoit-Wolf covariance, and the intensity."""
    S, delta = ES.ledoit_wolf_cc(r)
    sd = np.sqrt(np.diag(S.to_numpy()))
    return S / np.outer(sd, sd), delta


def mean_offdiag(corr: pd.DataFrame) -> float:
    m = corr.to_numpy()
    return float(m[~np.eye(len(m), dtype=bool)].mean())


@dataclass
class Agreement:
    table: pd.DataFrame          # one row per method: ARI and the groups found
    random_95: float             # 95th percentile of the ARI under shuffled labels


def agreement(corr: pd.DataFrame, labels: dict, n_random: int = 5000, seed: int = 0) -> Agreement:
    """How well each linkage method's k-group cut recovers our cluster labels."""
    true = [labels[t] for t in corr.index]
    k = len(set(true))
    rows = {}
    for m in METHODS:
        cut = fcluster(tree(corr, m), k, "maxclust")
        groups = pd.Series(corr.index, index=cut).groupby(level=0).agg(lambda s: ", ".join(s))
        rows[m] = {"ARI": adjusted_rand_score(true, cut), "Groups": " | ".join(groups.sort_values(key=lambda s: s.str.len()))}
    rng = np.random.default_rng(seed)
    null = [adjusted_rand_score(true, rng.permutation(true)) for _ in range(n_random)]
    return Agreement(pd.DataFrame(rows).T, float(np.percentile(null, 95)))


def cut_height(Z: np.ndarray, k: int) -> float:
    """A height that cuts the tree into k groups: midway between the merges that leave k and k-1 groups."""
    return float((Z[-k, 2] + Z[-(k - 1), 2]) / 2)
