import scipy.special
import numpy as np

import pytest

from mattspy.fm._jax_impl import (
    _lowrank_twoway_term,
    _fm_eval,
    _extract_fm_parts,
    _combine_fm_parts,
    _fm_with_class_feature_eval,
    _log_softmax_fm_eval,
    _softmax_fm_eval,
)


def _gen_test_data(n_samples, n_features, rank, flatten=False, seed=42):
    rng = np.random.default_rng(seed=seed)
    x = rng.normal(size=(n_samples, n_features))
    if flatten:
        x = x.flatten()

    vmat = rng.normal(size=(n_features, rank))
    w = rng.normal(size=(n_features))
    w0 = rng.normal()

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


def test_fm_lowrank_twoway_term_flat():
    n_samples = 1
    n_features = 13
    rank = 4
    data = _gen_test_data(n_samples, n_features, rank, flatten=True)
    x = data["x"]
    vmat = data["vmat"]

    vals = _lowrank_twoway_term(x, vmat)
    true_vals = []
    _val = 0.0
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            _val += np.sum(vmat[i, :] * vmat[j, :]) * x[i] * x[j]
    true_vals.append(_val)

    assert vals.shape == ()
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


def test_fm_eval_flat():
    n_samples = 1
    n_features = 13
    rank = 4
    data = _gen_test_data(n_samples, n_features, rank, flatten=True)
    x = data["x"]
    vmat = data["vmat"]
    w = data["w"]
    w0 = data["w0"]

    vals = _fm_eval(x, w0, w, vmat)
    true_vals = []
    _val = w0
    _val += np.sum(x * w)
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            _val += np.sum(vmat[i, :] * vmat[j, :]) * x[i] * x[j]
    true_vals.append(_val)

    assert vals.shape == ()
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_features", [1, 3, 11])
@pytest.mark.parametrize("rank", [1, 3])
def test_fm_extract_combine_fm_parts(n_features, rank):
    data = _gen_test_data(1, n_features, rank)
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]

    params = _combine_fm_parts(w0, w, vmat)
    np.testing.assert_allclose(params[0], w0)
    np.testing.assert_allclose(params[1 : 1 + n_features], w)
    np.testing.assert_allclose(params[1 + n_features :], vmat.flatten())

    tw0, tw, tvmat = _extract_fm_parts(params, n_features, rank)
    np.testing.assert_allclose(tw0, w0)
    np.testing.assert_allclose(tw, w)
    np.testing.assert_allclose(tvmat, vmat)


@pytest.mark.parametrize("n_samples", [1, 10])
@pytest.mark.parametrize("n_classes", [2, 13])
@pytest.mark.parametrize("class_ind", [0, 1])
def test_fm_with_class_feature_eval(class_ind, n_classes, n_samples):
    data = _gen_test_data(n_samples, 17, 5)
    x = data["x"][:, :-n_classes]
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]
    _class_ind = np.zeros_like(x[:, 0], dtype=int) + class_ind
    if x.shape[0] > 1:
        _class_ind[5:] = 1 - class_ind
    vals = _fm_with_class_feature_eval(
        x,
        w0,
        w,
        vmat,
        _class_ind,
        n_classes,
    )
    assert vals.shape == x.shape[:1]

    class_feat = np.zeros((x.shape[0], n_classes))
    class_feat[:, class_ind] = 1.0
    if x.shape[0] > 1:
        class_feat[5:, class_ind] = 0.0
        class_feat[5:, 1 - class_ind] = 1.0
    x = np.concatenate([class_feat, x], axis=1)
    true_vals = []
    for k in range(x.shape[0]):
        _val = w0
        _val += np.sum(x[k, :] * w)
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, :] * vmat[j, :]) * x[k, i] * x[k, j]
        true_vals.append(_val)

    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_classes", [2, 13])
@pytest.mark.parametrize("class_ind", [0, 1])
def test_fm_with_class_feature_eval_flat(class_ind, n_classes):
    data = _gen_test_data(1, 17, 5, flatten=True)
    x = data["x"][:-n_classes]
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]
    vals = _fm_with_class_feature_eval(
        x,
        w0,
        w,
        vmat,
        class_ind,
        n_classes,
    )
    assert vals.shape == ()

    class_feat = np.zeros(n_classes)
    class_feat[class_ind] = 1.0
    x = np.concatenate([class_feat, x], axis=0)
    true_vals = []
    _val = w0
    _val += np.sum(x[:] * w)
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            _val += np.sum(vmat[i, :] * vmat[j, :]) * x[i] * x[j]
    true_vals.append(_val)

    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", [1, 10])
@pytest.mark.parametrize("n_classes", [2, 13])
def test_fm_log_softmax_fm_eval(n_classes, n_samples):
    data = _gen_test_data(n_samples, 17, 5)
    x = data["x"][:, :-n_classes]
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]
    vals = _log_softmax_fm_eval(x, w0, w, vmat, n_classes)
    assert vals.shape == (n_samples, n_classes)
    np.testing.assert_allclose(np.sum(np.exp(vals), axis=-1), 1.0, rtol=1e-5, atol=1e-5)

    exp_vals = _softmax_fm_eval(x, w0, w, vmat, n_classes)
    assert exp_vals.shape == (n_samples, n_classes)
    np.testing.assert_allclose(exp_vals, np.exp(vals), rtol=1e-5, atol=1e-5)

    true_vals = []
    for k in range(x.shape[0]):
        _true_vals = []
        for cind in range(n_classes):
            class_feat = np.zeros((x.shape[0], n_classes))
            class_feat[:, cind] = 1.0
            _x = np.concatenate([class_feat, x], axis=1)

            _val = w0
            _val += np.sum(_x[k, :] * w)
            for i in range(_x.shape[1]):
                for j in range(i + 1, _x.shape[1]):
                    _val += np.sum(vmat[i, :] * vmat[j, :]) * _x[k, i] * _x[k, j]

            _true_vals.append(_val)
        true_vals.append(_true_vals)

    true_vals = np.array(true_vals)
    true_vals -= scipy.special.logsumexp(true_vals, axis=1, keepdims=True)

    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_classes", [2, 13])
def test_fm_log_softmax_fm_eval_flat(n_classes):
    data = _gen_test_data(1, 17, 5, flatten=True)
    x = data["x"][:-n_classes]
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]
    vals = _log_softmax_fm_eval(x, w0, w, vmat, n_classes)
    assert vals.shape == (n_classes,)
    np.testing.assert_allclose(np.sum(np.exp(vals), axis=-1), 1.0, rtol=1e-5, atol=1e-5)

    exp_vals = _softmax_fm_eval(x, w0, w, vmat, n_classes)
    assert exp_vals.shape == (n_classes,)
    np.testing.assert_allclose(exp_vals, np.exp(vals), rtol=1e-5, atol=1e-5)

    true_vals = []
    for cind in range(n_classes):
        class_feat = np.zeros(n_classes)
        class_feat[cind] = 1.0
        _x = np.concatenate([class_feat, x], axis=-1)

        _val = w0
        _val += np.sum(_x * w)
        for i in range(_x.shape[0]):
            for j in range(i + 1, _x.shape[0]):
                _val += np.sum(vmat[i, :] * vmat[j, :]) * _x[i] * _x[j]

        true_vals.append(_val)
    true_vals = np.array(true_vals)
    true_vals -= scipy.special.logsumexp(true_vals, axis=-1, keepdims=True)

    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)
