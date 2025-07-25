import numpy as np

import pytest

from mattspy.fm._jax_impl import (
    _lowrank_twoway_term,
    _fm_eval,
    _extract_fm_params,
    _combine_fm_params,
)


def _gen_test_data(n_samples, n_features, rank, flatten=False, seed=42, n_classes=None):
    rng = np.random.default_rng(seed=seed)
    x = rng.normal(size=(n_samples, n_features))
    if flatten:
        x = x.flatten()

    if n_classes is None:
        vmat = rng.normal(size=(n_features, rank))
        w = rng.normal(size=(n_features))
        w0 = rng.normal()
    else:
        vmat = rng.normal(size=(n_features, rank, n_classes))
        w = rng.normal(size=(n_features, n_classes))
        w0 = rng.normal(size=n_classes)

    return {"x": x, "vmat": vmat, "w": w, "w0": w0}


@pytest.mark.parametrize("n_samples", (1, 10))
def test_fm_lowrank_twoway_term(n_samples):
    n_features = 13
    rank = 4
    data = _gen_test_data(n_samples, n_features, rank, flatten=False)
    x = data["x"]
    vmat = data["vmat"]

    vals = _lowrank_twoway_term(x, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = 0.0
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, :] * vmat[j, :]) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == x.shape[:1]
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", (1, 10))
@pytest.mark.parametrize("n_classes", (1, 7))
def test_fm_lowrank_twoway_term_classes(n_samples, n_classes):
    n_features = 13
    rank = 4
    data = _gen_test_data(
        n_samples, n_features, rank, flatten=False, n_classes=n_classes
    )
    x = data["x"]
    vmat = data["vmat"]

    vals = _lowrank_twoway_term(x, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = 0.0
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, ...] * vmat[j, ...], axis=0) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == (x.shape[0], n_classes)
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", (1, 10))
def test_fm_eval(n_samples):
    n_features = 13
    rank = 4
    data = _gen_test_data(n_samples, n_features, rank, flatten=False)
    x = data["x"]
    vmat = data["vmat"]
    w = data["w"]
    w0 = data["w0"]

    vals = _fm_eval(x, w0, w, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = w0
        _val += np.sum(x[k, :] * w)
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, :] * vmat[j, :]) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == x.shape[:1]
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", (1, 10))
@pytest.mark.parametrize("n_classes", (1, 7))
def test_fm_eval_classes(n_samples, n_classes):
    n_features = 13
    rank = 4
    data = _gen_test_data(
        n_samples, n_features, rank, flatten=False, n_classes=n_classes
    )
    x = data["x"]
    vmat = data["vmat"]
    w = data["w"]
    w0 = data["w0"]

    vals = _fm_eval(x, w0, w, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = w0.copy()
        _val += np.sum(x[k, :].reshape(-1, 1) * w, axis=0)
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, ...] * vmat[j, ...], axis=0) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == (x.shape[0], n_classes)
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_classes", [-1, 0, None, 1, 7])
@pytest.mark.parametrize("n_features", [1, 3, 11])
@pytest.mark.parametrize("rank", [1, 3])
def test_fm_extract_combine_fm_params(n_features, rank, n_classes):
    data = _gen_test_data(
        1,
        n_features,
        rank,
        n_classes=None if n_classes is None or n_classes < 1 else n_classes,
    )
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]

    params = _combine_fm_params(w0, w, vmat)
    if n_classes is None or n_classes < 1:
        np.testing.assert_allclose(params[0], w0)
        np.testing.assert_allclose(params[1 : 1 + n_features], w)
        np.testing.assert_allclose(params[1 + n_features :], vmat.flatten())
    else:
        np.testing.assert_allclose(params[:n_classes], w0)
        np.testing.assert_allclose(
            params[n_classes : n_classes + n_features * n_classes], w.flatten()
        )
        np.testing.assert_allclose(
            params[n_classes + n_features * n_classes :], vmat.flatten()
        )

    tw0, tw, tvmat = _extract_fm_params(params, n_features, rank, n_classes)
    np.testing.assert_allclose(tw0, w0)
    np.testing.assert_allclose(tw, w)
    np.testing.assert_allclose(tvmat, vmat)
