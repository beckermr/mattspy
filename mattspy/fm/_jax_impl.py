from functools import partial

import jax
from jax import numpy as jnp

from sklearn.base import ClassifierMixin, BaseEstimator


@jax.jit
def _lowrank_twoway_term(x, vmat):
    fterm = jnp.einsum("np,pk...->nk...", x, vmat)
    sterm = jnp.einsum("np,pk...->nk...", x**2, vmat**2)
    return 0.5 * jnp.sum(fterm**2 - sterm, axis=1)


@jax.jit
def _fm_eval(x, w0, w, vmat):
    return w0 + jnp.einsum("np,p...->n...", x, w) + _lowrank_twoway_term(x, vmat)


@partial(jax.jit, static_argnames=("n_features", "rank", "n_classes"))
def _extract_fm_params(params, n_features, rank, n_classes):
    if n_classes is None or n_classes < 1:
        w0 = params[0]
        w = params[1 : 1 + n_features]
        vmat = params[1 + n_features :].reshape((n_features, rank))
    else:
        w0 = params[:n_classes]
        w = params[n_classes : n_classes + n_features * n_classes].reshape(
            (n_features, n_classes)
        )
        vmat = params[n_classes + n_features * n_classes :].reshape(
            (n_features, rank, n_classes)
        )

    return w0, w, vmat


@jax.jit
def _combine_fm_params(w0, w, vmat):
    return jnp.concatenate([jnp.atleast_1d(w0).flatten(), w.flatten(), vmat.flatten()])


class FMClassifierJAX(ClassifierMixin, BaseEstimator):
    """A JAX-based Factorization Machine classifier.

    Parameters
    ----------
    rank : int, optional
        The dimension of the low-rank approximation to the
        two-way interaction terms.
    random_state : int, numpy RNG instance, or None
        The RNG to use for parameter initialization.
    batch_size : int, optional
        The number of examples to use when fitting the estimator
        and making predictions. The value None indicates to use all
        examples.
    lambda_v : float, optional
        The L2 regularization strength to use for the low-rank embedding
        matrix.
    lambda_w : float, optional
        The L2 regularization strength to use for the linear terms.
    init_scale : float, optional
        The RMS of the Gaussian parameter initialization.
    solver : optimistix.AbstractMinimiser or str, optional
        The solver the use with the `optimistix` package. If a string is passed, the
        corresponding attribute from the `optax` package is used via
        `optimistix.OptaxMinimiser`.
    solver_kwargs : dict, optional
        An optional dictionary of keyword arguments to pass to the solver. These
        keywords are passed directly, except for `max_steps` which is passed to
        `optimistix.minimise` directly. The `max_steps` keyword is ignored during
        partial fitting.

    Attributes
    ----------
    n_dims_ : int
        Number of features in X.
    classes_ : array
        Class labels from the data.
    n_classes_ : int
        Number of unique class labels from the data.
    """

    pass
