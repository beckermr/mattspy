from io import BytesIO
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score

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


def test_som_oneshot():
    X, y = load_iris(return_X_y=True)
    clst = SOMap(
        random_state=RANDOM_SEED,
        n_clusters=len(np.unique(y)),
    )
    clst.partial_fit(X)
    init_score = clst.score(X)
    for i in range(10):
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
def test_som_pickling(with_jax):
    X, y = load_iris(return_X_y=True)
    if with_jax:
        X = jnp.array(X)
    clst = SOMap(
        random_state=RANDOM_SEED,
        n_clusters=len(np.unique(y)),
    )
    clst.partial_fit(X)
    labels = clst.predict(X)
    b = BytesIO()
    pickle.dump(clst, b)
    clst_pickled = pickle.loads(b.getvalue())
    labels_pickled = clst_pickled.predict(X)
    assert jnp.allclose(labels, labels_pickled)


@pytest.mark.parametrize("with_jax", [True, False])
def test_som_random_state_handling(with_jax):
    X, y = load_iris(return_X_y=True)
    if with_jax:
        X = jnp.array(X)
    clst = SOMap(
        random_state=RANDOM_SEED,
        n_clusters=len(np.unique(y)),
    )
    labels = clst.fit(X, y).predict(X)
    labels_again = clst.fit(X, y).predict(X)
    assert np.allclose(labels, labels_again)


def _apply_label_mapping(y, labels, n_clusters):
    from scipy.stats import mode

    vals = {}
    for k in range(n_clusters):
        vals[k] = mode(y[labels == k])[0]
    return jnp.array([vals[ll] for ll in labels])


def test_som_oneshot_cross_val():
    X, y = load_iris(return_X_y=True)
    clst = SOMap(
        random_state=RANDOM_SEED,
        n_clusters=len(np.unique(y)),
    )
    cv = KFold(n_splits=4, random_state=457, shuffle=True)
    labels = cross_val_predict(clst, X, cv=cv, method="predict")
    labels = _apply_label_mapping(y, labels, clst.n_clusters)
    final_f1 = f1_score(y, labels, average="micro")
    assert final_f1 > 0.80
