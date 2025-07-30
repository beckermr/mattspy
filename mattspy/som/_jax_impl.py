import jax
from jax import numpy as jnp

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.exceptions import NotFittedError


@jax.jit
def _jax_update_som_weights(inds, weights, dc_bar, X, alpha0, sigma02, k):
    def f(carry, ind):
        _weights, _dc_bar = carry

        x = X[ind, :]

        dc2 = jnp.sum((_weights - x) ** 2, axis=1)
        bmu_ind = jnp.argmin(dc2, axis=0)
        bmu = _weights[bmu_ind, :]
        dc2_bmu = dc2[bmu_ind]
        dc_bmu = jnp.sqrt(dc2_bmu)
        dc_bmu_dc_bar = dc_bmu / _dc_bar

        alpha = alpha0 * dc_bmu_dc_bar
        sigma2 = sigma02 * dc_bmu_dc_bar**2
        rci2 = jnp.sum((_weights - bmu) ** 2, axis=1)
        hci = alpha * jnp.exp(-rci2 / sigma2)
        _weights = _weights + hci.reshape(-1, 1) * (x - _weights)

        _dc_bar = k * _dc_bar + (1.0 - k) * dc_bmu

        return (_weights, _dc_bar), bmu_ind

    init_carry = (weights, dc_bar)
    final_carry, bmu_inds = jax.lax.scan(f, init_carry, xs=inds)
    final_weights, final_dc_bar = final_carry
    return final_weights, final_dc_bar, bmu_inds


@jax.jit
def _jax_predict_som(weights, X):
    return jnp.argmin(
        jnp.sum((weights[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]) ** 2, axis=-1),
        axis=1,
    )


class SOMap(ClusterMixin, BaseEstimator):
    """A SOM implementation.

    This SOM implementation uses the technique from

    "A New Self-Organizing Map with Continuous Learning Capability",
    H. Hikawa, H. Ito and Y. Maeda, 2018 IEEE Symposium Series on
    Computational Intelligence (SSCI)

    Parameters
    ----------
    n_clusters : int, optional
        The overall number of SOM weight vectors.
    alpha : float, optional
        The overall scaling of the neighborhood weight function
        amplitude.
    sigma : float, optional
        The overall scaling of the neighborhood weight function
        size.
    k : float, optional
        The weighting factor for the exponential time average
        of the adaptive neighborhood weight function width.
    random_state : int, numpy RNG instance, or None
        The RNG to use for parameter initialization.
    max_iter : int, optional
        The maximum number of times to iterate through the entire data
        set when fitting via `fit`.
    backend : str, optional
        The computational backend to use. Only "jax" is currently available.

    Attributes
    ----------
    weights_ : array-like
        An `(n**2, n_features_in)` shaped array of the SOM
        weight vectors.
    """

    def __init__(
        self,
        n_clusters=16,
        alpha=0.1,
        sigma=1e-2,
        k=0.9,
        random_state=None,
        max_iter=100,
        backend="jax",
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.sigma = sigma
        self.k = k
        self.random_state = random_state
        self.max_iter = max_iter
        self.backend = backend

    def fit(self, X, y=None):
        self._is_fit = False
        return self._partial_fit(self.max_iter, X)

    def partial_fit(self, X, y=None, xmin=None, xmax=None):
        return self._partial_fit(1, X)

    def _init_numpy(self, X):
        X = validate_data(self, X=X, reset=True)
        return X

    def _init_jax(self, X):
        self.n_features_in_ = X.shape[1]
        return X

    def _partial_fit(self, n_epochs, X, y=None, xmin=None, xmax=None):
        if not getattr(self, "_is_fit", False):
            self._rng = check_random_state(self.random_state)
            self._jax_rng_key = jax.random.key(
                self._rng.randint(low=1, high=int(2**31))
            )
            if not isinstance(X, jnp.ndarray):
                X = self._init_numpy(X)
            else:
                X = self._init_jax(X)

            self._sigma2 = self.sigma**2
            self._dc_bar = jnp.sqrt(self.n_features_in_)
        else:
            if not isinstance(X, jnp.ndarray):
                X = validate_data(self, X=X, reset=False)

        if not isinstance(X, jnp.ndarray):
            X = jnp.array(X)

        if not getattr(self, "_is_fit", False):
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            weights = jax.random.uniform(
                subkey, minval=0, maxval=1, shape=(self.n_clusters, self.n_features_in_)
            )

            if xmin is None:
                self._xmin = jnp.nanmin(X, axis=0)
            else:
                self._xmin = xmin
            if xmax is None:
                self._xmax = jnp.nanmax(X, axis=0)
            else:
                self._xmax = xmax
        else:
            weights = self.weights_.copy()

        Xs = (X - self._xmin) / (self._xmax - self._xmin)

        dc_bar = self._dc_bar
        inds = jnp.arange(X.shape[0])
        for epoch in range(n_epochs):
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            inds = jax.random.permutation(subkey, X.shape[0])
            weights, dc_bar, _ = _jax_update_som_weights(
                inds,
                weights,
                dc_bar,
                Xs,
                self.alpha,
                self._sigma2,
                self.k,
            )

        self.weights_ = weights
        self._dc_bar = dc_bar
        self._is_fit = True
        self.labels_ = _jax_predict_som(self.weights_, Xs)
        self.n_iter_ = epoch + 1

        return self

    def predict(self, X):
        if not isinstance(X, jnp.ndarray):
            X = validate_data(self, X=X, reset=False)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError("FMClassifier must be fit before calling `predict`!")

        Xs = (X - self._xmin) / (self._xmax - self._xmin)
        return _jax_predict_som(self.weights_, Xs)
