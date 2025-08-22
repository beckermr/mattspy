import jax
from jax import numpy as jnp
import numpy as np
import optax

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.exceptions import NotFittedError

from mattspy.json import EstimatorToFromJSONMixin


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
    # shape of rci2 is k,n
    rci2 = jnp.sum(
        (wpos[:, jnp.newaxis, :] - wpos[jnp.newaxis, bmu_inds, :]) ** 2, axis=-1
    )
    sigma2 = sigma * sigma
    # hci has shape k,n
    hci = jnp.exp(-rci2 / sigma2)
    # shape of hci_tot is k
    hci_tot = jnp.sum(hci, axis=-1)
    n_seen = n_seen + hci_tot
    eta = 1.0 / n_seen
    # shape of kern is k,n,d
    kern = hci[:, :, jnp.newaxis] * (X[jnp.newaxis, :, :] - weights[:, jnp.newaxis, :])
    # weights have shape k,d w/ sum over kern on axis 1 which has dim n
    weights = weights + (eta.reshape(-1, 1) * jnp.sum(kern, axis=1))
    return weights, n_seen, bmu_inds


@jax.jit
def _jax_compute_extended_distortion(weights, wpos, X, sigma):
    bmu_inds = jnp.argmin(
        jnp.sum((weights[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]) ** 2, axis=-1),
        axis=1,
    )
    # shape of rci2 and hci is k,n
    rci2 = jnp.sum(
        (wpos[:, jnp.newaxis, :] - wpos[jnp.newaxis, bmu_inds, :]) ** 2, axis=-1
    )
    hci = jnp.exp(-0.5 * rci2 / sigma / sigma)
    # shape of dx is k,n,d
    dx = X[jnp.newaxis, :, :] - weights[:, jnp.newaxis, :]
    # shape of dx2 is k,n
    dx2 = jnp.sum(dx * dx, axis=-1)
    return jnp.sum(hci * dx2) / 2.0 / X.shape[0]


_grad_jax_compute_extended_distortion = jax.jit(
    jax.grad(_jax_compute_extended_distortion)
)


class SOMap(EstimatorToFromJSONMixin, ClusterMixin, BaseEstimator):
    """A mini-batch Self-organazing Map (SOM) implementation.

    This SOM implementation fits the data through a mini-batch technique
    based on either using a custom 'online' optimizer or directly minimizing
    the Extended Distortion.

    The 'online' mini-batch technique is based on extending the adaptive mini-batch
    K-means algorithm from Sculley (2010, "Web-Scale K-Means Clustering")
    to SOMs using gradients of the Extended Distortion (Ritter et al.,
    1992, "Neural Computation and Self-Organizing Maps: an Introduction").
    Unlike Sculley (2010), this implementation does the gradient descent update
    in a purely vectorized fashion for better performance when implementated
    in JAX.

    Parameters
    ----------
    n_clusters : int, optional
        The overall number of SOM units.
    sigma_frac : float, optional
        The neighborhood weight function size in fractions of the full 2D
        SOM grid size. A value of 0.05 indicates that the neighborhood
        weight function has a Gaussian width of 5% of the SOM grid.
    random_state : int, numpy RNG instance, or None
        The RNG to use for unit weight vector initialization.
    batch_size : int, optional
        The number of examples to use when fitting the SOM and labeling
        examples.
    solver : str, optional
        The solver the use from the `optax` package. If set to "online", then
        an online technique adapted from Sculley (2010) is used.
    solver_kwargs : tuple of key-value pairs, optional
        An optional tuple of tuples of keyword arguments to pass to the solver.
    max_iter : int, optional
        The maximum number of times to iterate through the entire data
        set when fitting via `fit`.
    atol : float, optional
        The absolute tolerance for convergence in the SOM unit weight vectors.
    rtol : float, optional
        The relative tolerance for convergence in the SOM unit weight vectors.
    backend : str, optional
        The computational backend to use. Only "jax" is currently available.

    Attributes
    ----------
    weights_ : array-like
        An `(n_clusters, n_features_in)` shaped array of the SOM
        unit weight vectors.
    weight_positions_ : array-like
        An `(n_clusters, 2)` shaped array holding the 2d positions
        of the SOM units.
    n_iter_ : int
        The number of iterations the estimator has been run.
    labels_ : array-like
        The labels of the last dataset used to fit the estimator.
    n_seen_ : array-like
        The effective number of examples seen per SOM unit. Only set if
        `solver` is 'online'.
    n_weight_grid_ : int
        The dimension of the 2D SOM unit grid.
    converged_ : bool
        Set to True if the fit converged. False otherwise.
    """

    json_attributes_ = (
        "_is_fit",
        "_rng",
        "_jax_rng_key",
        "n_seen_",
        "n_weight_grid_",
        "n_iter_",
        "converged_",
        "weights_",
        "weight_positions_",
    )

    def __init__(
        self,
        n_clusters=16,
        sigma_frac=0.1,
        random_state=None,
        batch_size=128,
        solver="adam",
        solver_kwargs=(("learning_rate", 1e-3),),
        max_iter=200,
        rtol=1e-4,
        atol=1e-4,
        backend="jax",
    ):
        self.n_clusters = n_clusters
        self.sigma_frac = sigma_frac
        self.random_state = random_state
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol
        self.backend = backend
        self.batch_size = batch_size
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def fit(self, X, y=None):
        """Fit the SOM to the data `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        self : SOMap
            The fit estimator.
        """
        self._is_fit = False
        return self._partial_fit(self.max_iter, X)

    def partial_fit(self, X, y=None):
        """Update the SOM unit weight vectors given some examples `X`.

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

    def _init_from_json(self, X=None, **kwargs):
        if X is None and "weights_" in kwargs:
            X = np.ones((1, kwargs["weights_"].shape[1]))

        self.n_seen_ = kwargs.get(
            "n_seen_",
            jnp.zeros(self.n_clusters),
        )
        self.n_weight_grid_ = kwargs.get(
            "n_weight_grid_",
            int(np.ceil(np.sqrt(self.n_clusters))),
        )
        self.n_iter_ = kwargs.get("n_iter_", 0)

        self._rng = kwargs.get("_rng", check_random_state(self.random_state))

        if "_jax_rng_key" in kwargs:
            self._jax_rng_key = kwargs["_jax_rng_key"]
        else:
            self._jax_rng_key = jax.random.key(
                self._rng.randint(low=1, high=int(2**31))
            )

        # check inputs and convert to JAX
        if not isinstance(X, jnp.ndarray):
            X = self._init_numpy(X)
            X = jnp.array(X)
        else:
            self._init_numpy(np.ones((1, X.shape[1])))

        if "weights_" in kwargs:
            self.weights_ = jnp.array(kwargs["weights_"])
        else:
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

        if "weight_positions_" in kwargs:
            self.weight_positions_ = jnp.array(kwargs["weight_positions_"])
        else:
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

        self.converged_ = kwargs.get("converged_", False)
        self._is_fit = kwargs.get("_is_fit", True)

        return X

    def _partial_fit(self, n_epochs, X, y=None):
        if not getattr(self, "_is_fit", False):
            X = self._init_from_json(X)
        else:
            if not isinstance(X, jnp.ndarray):
                X = validate_data(self, X=X, reset=False)
                X = jnp.array(X)

        if self.solver != "online":
            solver_kwargs = {k: v for k, v in (self.solver_kwargs or tuple())}
            optimizer = getattr(optax, self.solver)(**solver_kwargs)
            opt_state = optimizer.init(self.weights_)

        dw = 1.0 / self.n_weight_grid_
        sigma_frac_dw = np.maximum(1.0, self.sigma_frac / dw)

        converged = False
        for _ in range(n_epochs):
            old_weights = self.weights_

            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            inds = jax.random.permutation(subkey, X.shape[0])

            _sigma_frac = dw * np.power(
                sigma_frac_dw, np.maximum(1.0 - self.n_iter_ / self.max_iter, 0.0)
            )

            for start in range(0, X.shape[0], self.batch_size):
                end = min(start + self.batch_size, X.shape[0])
                Xb = X[inds[start:end], :]

                if self.solver != "online":
                    grads = _grad_jax_compute_extended_distortion(
                        self.weights_,
                        self.weight_positions_,
                        Xb,
                        _sigma_frac,
                    )
                    updates, opt_state = optimizer.update(
                        grads, opt_state, self.weights_
                    )
                    self.weights_ = optax.apply_updates(self.weights_, updates)
                else:
                    _weights, _n_seen, _ = _jax_update_som_weights_minibatch(
                        self.weights_,
                        self.n_seen_,
                        self.weight_positions_,
                        Xb,
                        _sigma_frac,
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
            raise NotFittedError("SOMap must be fit before calling `predict`!")

        vals = []
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            Xb = X[start:end, :]
            vals.append(_jax_predict_som(self.weights_, Xb))
        return jnp.concatenate(vals, axis=0)

    def score(self, X, y=None):
        """Compute the negative of the extended distortion.

        The extended distortion is the generalization of the K-means
        loss to SOMs. See Ritter et al. (1992,
        "Neural Computation and Self-Organizing Maps: an Introduction").

        A higher score (i.e., lower extended distortion) indicates a better
        SOM.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        neg_ext_dist : float
            The negative of the extended distortion.
        """
        if not isinstance(X, jnp.ndarray):
            X = validate_data(self, X=X, reset=False)
            X = jnp.array(X)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError("SOMap must be fit before calling `predict`!")

        val = 0.0
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            Xb = X[start:end, :]
            val = val + (
                _jax_compute_extended_distortion(
                    self.weights_, self.weight_positions_, Xb, self.sigma_frac
                )
                * Xb.shape[0]
            )
        return -val / X.shape[0]
