from functools import partial

import jax
from jax import numpy as jnp


def _lowrank_twoway_term(x, vmat):
    fterm = jnp.dot(x, vmat)
    sterm = jnp.dot(x**2, vmat**2)
    return 0.5 * jnp.sum(fterm**2 - sterm, axis=-1)


def _fm_eval(x, w0, w, vmat):
    return w0 + jnp.dot(x, w) + _lowrank_twoway_term(x, vmat)


@partial(jax.jit, static_argnames=("n_features", "rank"))
def _extract_fm_parts(params, n_features, rank):
    w0 = params[0]
    w = params[1 : 1 + n_features]
    vmat = params[1 + n_features :].reshape((n_features, rank))
    return w0, w, vmat


def _combine_fm_parts(w0, w, vmat):
    return jnp.concatenate([jnp.atleast_1d(w0), w, vmat.flatten()])


@partial(jax.jit, static_argnames=("n_classes",))
def _fm_with_class_feature_eval(x, w0, w, vmat, class_ind, n_classes):
    class_feature = jax.nn.one_hot(class_ind, n_classes)
    x_with_class_feature = jnp.concatenate([class_feature, x], axis=-1)
    return _fm_eval(x_with_class_feature, w0, w, vmat)


@partial(jax.jit, static_argnames=("n_classes",))
def _log_softmax_fm_eval(x, w0, w, vmat, n_classes):
    class_inds = jnp.arange(n_classes, dtype=int)
    if jnp.ndim(x) == 2:
        class_inds = jnp.tile(class_inds, (x.shape[0], 1))
    return jax.nn.log_softmax(
        jax.vmap(
            _fm_with_class_feature_eval,
            in_axes=(None, None, None, None, -1, None),
            out_axes=-1,
        )(x, w0, w, vmat, class_inds, n_classes),
        axis=-1,
    )


@partial(jax.jit, static_argnames=("n_classes",))
def _softmax_fm_eval(x, w0, w, vmat, n_classes):
    class_inds = jnp.arange(n_classes, dtype=int)
    if jnp.ndim(x) == 2:
        class_inds = jnp.tile(class_inds, (x.shape[0], 1))
    return jax.nn.softmax(
        jax.vmap(
            _fm_with_class_feature_eval,
            in_axes=(None, None, None, None, -1, None),
            out_axes=-1,
        )(x, w0, w, vmat, class_inds, n_classes),
        axis=-1,
    )
