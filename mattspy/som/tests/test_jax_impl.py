from io import BytesIO
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris

from mattspy.som._jax_impl import SOMap

RANDOM_SEED = 42


@pytest.fixture
def clst():
    return SOMap(
        random_state=RANDOM_SEED,
        batch_size=4,
        n_clusters=3,
        sigma_frac=0.75,
        max_iter=400,
    )


def test_som_check_estimator(clst):
    check_estimator(clst)


def _mode_label(y, labels, n_clusters):
    from scipy.stats import mode

    vals = []
    for k in range(n_clusters):
        vals.append((mode(y[labels == k])[0]))
    return vals


def test_som_oneshot(clst):
    X, y = load_iris(return_X_y=True)
    clst.partial_fit(X)
    init_score = clst.score(X)
    for i in range(clst.max_iter + 10):
        clst.partial_fit(X)
    final_score = clst.score(X)
    ml = _mode_label(y, clst.labels_, clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))
    assert final_score > init_score

    ml = _mode_label(y, clst.predict(X), clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))

    clst.fit(X)
    ml = _mode_label(y, clst.predict(X), clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))


@pytest.mark.parametrize("with_jax", [True, False])
def test_som_pickling(with_jax, clst):
    X, y = load_iris(return_X_y=True)
    if with_jax:
        X = jnp.array(X)
    clst.partial_fit(X)
    labels = clst.predict(X)
    b = BytesIO()
    pickle.dump(clst, b)
    clst_pickled = pickle.loads(b.getvalue())
    labels_pickled = clst_pickled.predict(X)
    assert jnp.allclose(labels, labels_pickled)


@pytest.mark.parametrize("with_jax", [True, False])
def test_som_random_state_handling(with_jax, clst):
    X, y = load_iris(return_X_y=True)
    if with_jax:
        X = jnp.array(X)
    labels = clst.fit(X, y).predict(X)
    labels_again = clst.fit(X, y).predict(X)
    assert np.allclose(labels, labels_again)


def _apply_label_mapping(y, labels, n_clusters):
    from scipy.stats import mode

    vals = {}
    for k in range(n_clusters):
        vals[k] = mode(y[labels == k])[0]
    return jnp.array([vals[int(ll)] for ll in labels])
