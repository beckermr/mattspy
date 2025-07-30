import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris

from mattspy.som._jax_impl import SOMap

RANDOM_SEED = 42


def test_som_check_estimator():
    check_estimator(
        SOMap(random_state=RANDOM_SEED),
    )


def _mode_label(y, labels, n_clusters):
    from scipy.stats import mode

    vals = []
    for k in range(n_clusters):
        vals.append((mode(y[labels == k])[0]))
    return vals


def test_som_iris_oneshot():
    X, y = load_iris(return_X_y=True)
    clst = SOMap(
        random_state=RANDOM_SEED,
        n_clusters=len(np.unique(y)),
    )
    clst.partial_fit(X)
    for i in range(10):
        clst.partial_fit(X)
    ml = _mode_label(y, clst.labels_, clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))

    ml = _mode_label(y, clst.predict(X), clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))

    clst.fit(X)
    ml = _mode_label(y, clst.predict(X), clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))
