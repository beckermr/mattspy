import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris

from mattspy.som._jax_impl import (
    SOMap
)

RANDOM_SEED = 42


def test_som_check_estimator():
    check_estimator(
        SOMap(random_state=RANDOM_SEED),
    )


def _mode_label(y, clst):
    from scipy.stats import mode
    vals = []
    for k in range(clst.n_clusters):
        vals.append((mode(y[clst.labels_ == k])[0]))
    return vals


def test_som_iris_oneshot():
    X, y = load_iris(return_X_y=True)
    clst = SOMap(
        random_state=RANDOM_SEED,
        n_clusters=len(np.unique(y)),
    )
    rng = np.random.default_rng(seed=42)
    inds = rng.permutation(X.shape[0])
    clst.partial_fit(X[inds, :])
    for i in range(10):
        clst.partial_fit(X[inds, :])
    ml = _mode_label(y, clst)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))
