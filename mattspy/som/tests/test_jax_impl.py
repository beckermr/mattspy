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


def test_som_to_from_json_fit(clst):
    X, y = load_iris(return_X_y=True)
    clst.fit(X)
    labels = clst.predict(X)
    est_json = clst.to_json()
    ml = _mode_label(y, clst.labels_, clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))

    print(est_json)

    new_clst = SOMap.from_json(est_json)
    assert est_json == new_clst.to_json()
    assert jnp.array_equal(clst.weights_, new_clst.weights_)
    new_labels = new_clst.predict(X)
    assert jnp.allclose(labels, new_labels)

    new_clst.fit(X)
    assert jnp.array_equal(clst.weights_, new_clst.weights_)
    new_fit_labels = new_clst.predict(X)
    assert jnp.allclose(labels, new_fit_labels)


def test_som_to_from_json_partial_fit(clst):
    X, y = load_iris(return_X_y=True)
    clst.partial_fit(X)
    new_clst = SOMap.from_json(clst.to_json())
    for _ in range(399):
        clst.partial_fit(X)
        new_clst.partial_fit(X)

    new_labels = new_clst.predict(X)
    labels = clst.predict(X)
    assert jnp.allclose(labels, new_labels)

    ml = _mode_label(y, labels, clst.n_clusters)
    assert np.array_equal(np.sort(ml), np.arange(clst.n_clusters))

    assert jnp.array_equal(clst.weights_, new_clst.weights_)
    assert jnp.array_equal(clst.n_features_in_, new_clst.n_features_in_)
