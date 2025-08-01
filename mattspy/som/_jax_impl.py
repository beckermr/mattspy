import jax
from jax import numpy as jnp

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.exceptions import NotFittedError


@jax.jit
def _jax_update_som_weights(inds, weights, wpos, dc_bar, X, alpha0, sigma0, k):
    def f(carry, ind):
        _weights, _dc_bar = carry

        x = X[ind, :]

        dc2 = jnp.sum((_weights - x) ** 2, axis=1)
        bmu_ind = jnp.argmin(dc2, axis=0)

        dc2_bmu = dc2[bmu_ind]
        dc_bmu = jnp.sqrt(dc2_bmu)
        dc_bmu_dc_bar = dc_bmu / _dc_bar
        exp_fac = k + (1.0 - k) * dc_bmu_dc_bar
        alpha = alpha0 * exp_fac
        sigma = sigma0 * exp_fac
        _dc_bar = k * _dc_bar + (1.0 - k) * dc_bmu

        rci2 = jnp.sum((wpos - wpos[bmu_ind, :]) ** 2, axis=1)
        hci = alpha * jnp.exp(-rci2 / sigma / sigma)
        _weights = _weights + hci.reshape(-1, 1) * (x - _weights)

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
    """An Adaptive SOM implementation.

    The adaptive option of this  SOM implementation uses a version of
    the technique from

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
        of the adaptive scaling parameter.
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
        An `(n_clusters, n_features_in)` shaped array of the SOM
        weight vectors.
    weight_positions_ : array-like
        An `(n_clusters, 2)` shaped array holding the 2d positions
        of the weight units in the SOM space.
    """

    def __init__(
        self,
        n_clusters=16,
        alpha=1e-2,
        sigma=1,
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
        """Fit the SOM to the data `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        self : ScheduledSOMap
            The fit estimator.
        """
        self._is_fit = False
        return self._partial_fit(self.max_iter, X)

    def partial_fit(self, X, y=None, xmin=None, xmax=None):
        """Update the SOM weight vectors (units) given some examples `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.
        xmin : float, optional
            If given, the minimum value of X over the entire dataset. If None,
            use the minimum of `X` from the first time `partial_fit` is called.
        xmax : float, optional
            If given, the maximum value of X over the entire dataset. If None,
            use the maximum of `X` from the first time `partial_fit` is called.

        Returns
        -------
        self : ScheduledSOMap
            The fit estimator.
        """
        return self._partial_fit(1, X)

    def _init_numpy(self, X):
        X = validate_data(self, X=X, reset=True)
        return X

    def _init_jax(self, X):
        self.n_features_in_ = X.shape[1]
        return X

    def _partial_fit(self, n_epochs, X, y=None):
        if not getattr(self, "_is_fit", False):
            self._rng = check_random_state(self.random_state)
            self._jax_rng_key = jax.random.key(
                self._rng.randint(low=1, high=int(2**31))
            )
            if not isinstance(X, jnp.ndarray):
                X = self._init_numpy(X)
            else:
                X = self._init_jax(X)

            if not isinstance(X, jnp.ndarray):
                X = jnp.array(X)

            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)

            if X.shape[1] == 1:
                std_scale = jnp.std(X)
                weights = (
                    jax.random.uniform(
                        subkey,
                        shape=(self.n_clusters, 1),
                        minval=-3,
                        maxval=3,
                    )
                    * std_scale
                )
            else:
                cov = jnp.cov(X.T)
                eigval, eigvec = jnp.linalg.eigh(cov)
                eigval = eigval.real
                eigvec = eigvec.real
                eiginds = jnp.argsort(eigval)[::-1]
                evec1 = eigvec[:, eiginds[0]]
                sqrt_eval1 = jnp.sqrt(eigval[eiginds[0]])
                evec2 = eigvec[:, eiginds[1]]
                sqrt_eval2 = jnp.sqrt(eigval[eiginds[1]])

                weights = jax.random.uniform(
                    subkey,
                    shape=(self.n_clusters, 2),
                    minval=-3,
                    maxval=3,
                )
                weights = (
                    weights[:, 0:1] * sqrt_eval1 * evec1
                    + weights[:, 1:2] * sqrt_eval2 * evec2
                )

            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            inds = jax.random.choice(subkey, X.shape[0], replace=True, shape=(200,))
            self._dc_bar = jnp.sqrt(
                jnp.mean(
                    jnp.min(
                        jnp.sum(
                            (
                                X[jnp.newaxis, inds[:100], :]
                                - X[inds[100:200], jnp.newaxis, :]
                            )
                            ** 2,
                            axis=-1,
                        ),
                        axis=1,
                    )
                )
            )
            if self._dc_bar == 0:
                self._dc_bar = 1.0
            print(self._dc_bar)

            n_grid = int(jnp.ceil(jnp.sqrt(self.n_clusters)))
            pos = jnp.linspace(0, 1, n_grid)
            xp, yp = jnp.meshgrid(pos, pos)
            xp = xp.ravel()
            yp = yp.ravel()
            wpos = jnp.vstack([xp, yp]).T

            if self.n_clusters < wpos.shape[0]:
                self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
                rind = jax.random.choice(
                    subkey, wpos.shape[0], replace=False, shape=(self.n_clusters,)
                )
                wpos = wpos[rind, :]

            self.weight_positions_ = wpos

            sigma_fac = jnp.sqrt(1 / self.n_clusters)

            self._sigma = self.sigma * sigma_fac
            self._alpha = self.alpha
        else:
            if not isinstance(X, jnp.ndarray):
                X = validate_data(self, X=X, reset=False)

            weights = self.weights_.copy()

        if not isinstance(X, jnp.ndarray):
            X = jnp.array(X)

        dc_bar = self._dc_bar
        for epoch in range(n_epochs):
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            inds = jax.random.permutation(subkey, X.shape[0])
            weights, dc_bar, _ = _jax_update_som_weights(
                inds,
                weights,
                self.weight_positions_,
                dc_bar,
                X,
                self._alpha,
                self._sigma,
                self.k,
            )

        self.weights_ = weights
        self._dc_bar = dc_bar
        self._is_fit = True
        self.labels_ = _jax_predict_som(self.weights_, X)
        self.n_iter_ = epoch + 1

        return self

    def predict(self, X):
        """Compute the SOM best-matching unit index for `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        labels : array-like
            An array of the integer indicies of the best-matching unit.
        """
        if not isinstance(X, jnp.ndarray):
            X = validate_data(self, X=X, reset=False)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError("FMClassifier must be fit before calling `predict`!")

        return _jax_predict_som(self.weights_, X)
