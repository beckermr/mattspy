import jax
from jax import numpy as jnp

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.exceptions import NotFittedError


@jax.jit
def _jax_predict_som(weights, X):
    return jnp.argmin(
        jnp.sum((weights[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]) ** 2, axis=-1),
        axis=1,
    )


@jax.jit
def _jax_update_som_weights_minibatch(weights, n_seen, wpos, X, sigma):
    bmu_inds = jnp.argmin(
        jnp.sum((weights[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]) ** 2, axis=-1),
        axis=1,
    )
    # shape of rci2 and hci is k,n
    rci2 = jnp.sum(
        (wpos[:, jnp.newaxis, :] - wpos[jnp.newaxis, bmu_inds, :]) ** 2, axis=-1
    )
    hci = jnp.exp(-0.5 * rci2 / sigma / sigma)
    # jax.debug.print("rci2 shape: {}\n", rci2.shape)
    # shape of hci_tot is k
    hci_tot = jnp.sum(hci, axis=-1)
    # jax.debug.print("hci_tot shape: {}\n", hci_tot.shape)
    n_seen = n_seen + hci_tot
    eta = 1.0 / n_seen
    # shape of kern is k,n,d
    kern = hci[:, :, jnp.newaxis] * (X[jnp.newaxis, :, :] - weights[:, jnp.newaxis, :])
    # jax.debug.print("kern shape: {}\n", kern.shape)
    weights = weights + (eta.reshape(-1, 1) * jnp.sum(kern, axis=1))
    return weights, n_seen, bmu_inds


class SOMap(ClusterMixin, BaseEstimator):
    """A mini-batch SOM implementation.

    Parameters
    ----------
    n_clusters : int, optional
        The overall number of SOM weight vectors.
    sigma : float, optional
        The neighborhood weight function
        size in units such that `sigma=1` corresponds to SOM cells
        that are adjacent on the underlying 2D SOM cell grid.
    random_state : int, numpy RNG instance, or None
        The RNG to use for parameter initialization.
    batch_size : int, optional
        The number of examples to use when fitting the estimator
        and making predictions.
    max_iter : int, optional
        The maximum number of times to iterate through the entire data
        set when fitting via `fit`.
    atol : float, optional
        The absolute tolerance for convergence in the SOM weight vectors.
    rtol : float, optional
        The relative tolerance for convergence in the SOM weight vectors.
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
    n_iter_ : int
        The number of iterations the estimator has been run.
    labels_ : array-like
        The labels of the last dataset used to fit the estimator.
    n_seen_ : array-like
        The effective number of examples seen per cluster.
    n_weight_grid_ : int
        The dimension of the 2D SOM weight grid.
    converged_ : bool
        Set to True if the fit converged. False otherwise.
    """

    def __init__(
        self,
        n_clusters=16,
        sigma=1,
        random_state=None,
        batch_size=128,
        max_iter=100,
        rtol=1e-4,
        atol=1e-4,
        backend="jax",
    ):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.random_state = random_state
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol
        self.backend = backend
        self.batch_size = batch_size

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

    def partial_fit(self, X, y=None):
        """Update the SOM weight vectors (units) given some examples `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        self : object
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
            self.n_seen_ = jnp.zeros(self.n_clusters)
            self.n_weight_grid_ = int(jnp.ceil(jnp.sqrt(self.n_clusters)))
            self.n_iter_ = 0
            self._sigma = self.sigma / self.n_weight_grid_

            # rng init
            self._rng = check_random_state(self.random_state)
            self._jax_rng_key = jax.random.key(
                self._rng.randint(low=1, high=int(2**31))
            )

            # check inputs and convert to JAX
            if not isinstance(X, jnp.ndarray):
                X = self._init_numpy(X)
                X = jnp.array(X)
            else:
                X = self._init_jax(X)

            # weight init
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            if X.shape[1] == 1:
                std_scale = jnp.std(X)
                self.weights_ = (
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

                eigscale = jax.random.uniform(
                    subkey,
                    shape=(self.n_clusters, 2),
                    minval=-3,
                    maxval=3,
                )
                self.weights_ = (
                    eigscale[:, 0:1] * sqrt_eval1 * evec1
                    + eigscale[:, 1:2] * sqrt_eval2 * evec2
                )

            # weight position init
            pos = jnp.linspace(0, 1, self.n_weight_grid_)
            xp, yp = jnp.meshgrid(pos, pos)
            xp = xp.ravel()
            yp = yp.ravel()
            self.weight_positions_ = jnp.vstack([xp, yp]).T
            if self.n_clusters < self.weight_positions_.shape[0]:
                self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
                rind = jax.random.choice(
                    subkey,
                    self.weight_positions_.shape[0],
                    replace=False,
                    shape=(self.n_clusters,),
                )
                self.weight_positions_ = self.weight_positions_[rind, :]
        else:
            if not isinstance(X, jnp.ndarray):
                X = validate_data(self, X=X, reset=False)
                X = jnp.array(X)

        converged = False
        for epoch in range(n_epochs):
            old_weights = self.weights_

            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            inds = jax.random.permutation(subkey, X.shape[0])
            for start in range(0, X.shape[0], self.batch_size):
                end = min(start + self.batch_size, X.shape[0])
                Xb = X[inds[start:end], :]

                _weights, _n_seen, _ = _jax_update_som_weights_minibatch(
                    self.weights_,
                    self.n_seen_,
                    self.weight_positions_,
                    Xb,
                    self._sigma,
                )
                self.weights_ = _weights
                self.n_seen_ = _n_seen

            self.n_iter_ += 1
            if self.n_iter_ > 1 and jnp.allclose(
                self.weights_, old_weights, rtol=self.rtol, atol=self.atol
            ):
                converged = True
                break

        self.converged_ = converged
        self._is_fit = True
        self.labels_ = _jax_predict_som(self.weights_, X)

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
            X = jnp.array(X)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError("MiniBatchSOMap must be fit before calling `predict`!")

        vals = []
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            Xb = X[start:end, :]
            vals.append(_jax_predict_som(self.weights_, Xb))
        return jnp.concatenate(vals, axis=0)
