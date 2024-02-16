"""
Core utilities to fit a group factor analysis (GFA) model to data.

**Functions**

- :func:`fit` -- Fit a GFA model to data.
- :func:`init` -- Initialize GFA model parameters to data prior to fitting.
- :func:`infer_latents` -- Infer latent variables.
- :func:`infer_loadings` -- Infer loading matrices.
- :func:`infer_ard` -- Infer ARD parameters.
- :func:`infer_obs_mean` -- Infer observation mean parameter.
- :func:`infer_obs_prec` -- Infer observation precision parameters.
- :func:`compute_lower_bound` -- Compute the variational lower bound.
- :func:`compute_lower_bound_constants` -- Compute constants in the lower bound.

**Classes**

- :class:`GFAModel` -- A wrapper class to store GFA fitting results.

"""

from __future__ import annotations

import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln, psi
from scipy.stats import gmean

from latents.gfa.data_types import (
    GFAFitArgs,
    GFAFitFlags,
    GFAFitTracker,
    GFAParams,
    HyperPriorParams,
    ObsData,
    PosteriorARD,
    PosteriorLatent,
    PosteriorLoading,
    PosteriorObsMean,
    PosteriorObsPrec,
)

jsonpickle_numpy.register_handlers()


def fit(
    Y: ObsData,
    params: GFAParams,
    x_dim_init: int = 1,
    hyper_priors: HyperPriorParams | None = None,
    fit_tol: float = 1e-8,
    max_iter: int = int(1e6),
    verbose: bool = False,
    random_seed: int | None = None,
    min_var_frac: float = 0.001,
    prune_X: bool = True,
    prune_tol: float = 1e-7,
    save_X: bool = False,
    save_C_cov: bool = False,
    save_fit_progress: bool = True,
) -> tuple[GFAParams, GFAFitTracker, GFAFitFlags]:
    """Fit a GFA model to data.

    Fit a group factor analysis (GFA) model using an iterative variational
    inference scheme with mean-field approximation.

    Parameters
    ----------
    Y
        Observed data.
    params
        Initial GFA model parameters.
    x_dim_init
        Initial number of latent dimensions to fit (before pruning).
        Defaults to ``1``.
    hyper_priors
        Hyperparameters of the GFA prior distributions. If not provided,
        default hyperparameters will be used.
    fit_tol
        Tolerance for convergence. Defaults to ``1e-8``.
    max_iter
        Maximum number of iterations. Defaults to ``1e6``.
    verbose
        Specifies whether to display progress information. Defaults to
        ``False``.
    random_seed
        Seed the random number generator for reproducibility. Defaults to
        ``None``.
    min_var_frac
        Fraction of overall data variance for each observed dimension to set
        as the private variance floor. Defaults to ``0.001``.
    prune_X
        Set to ``True`` to remove latents that become inactive. Can speed up
        runtime and improve memory efficiency. Defaults to ``True``.
    prune_tol
        Tolerance for pruning. Sample second moment of each latent must
        remain larger than this value to remain in the model. Defaults to
        ``1e-7``.
    save_X
        Set to ``True`` to save posterior estimates of latent variables
        :math:`X`. For large datasets, ``X.mean`` may be very large.
        Defaults to ``False``.
    save_C_cov
        Set to true to save posterior covariance of :math:`C`. For large
        ``y_dim`` and ``x_dim``, these structures can use a lot of memory.
        Defaults to ``False``.
    save_fit_progress
        Set to ``True`` to save the lower bound and runtime at each iteration.
        Defaults to ``True``.

    Returns
    -------
    GFAParams
        Fitted model parameters.
    GFAFitTracker
        Quantities tracked during fitting.
    GFAFitFlags
        Status messages about the fitting process.

    Raises
    ------
    TypeError
        If ``hyper_priors`` is not a ``HyperPriorParams`` object.
    ValueError
        If ``params.y_dims`` does not match ``Y.dims`` or if ``params.x_dim``
        does not match ``x_dim_init``.
    """
    # Initialize GFA model parameters if they have not been initialized already
    if not params.is_initialized():
        if verbose:
            print("GFA model parameters not initialized. Initializing...")
        params = init(
            Y,
            x_dim_init=x_dim_init,
            hyper_priors=hyper_priors,
            random_seed=random_seed,
            save_C_cov=save_C_cov,
        )

    # Initialize hyper_priors if not provided
    if hyper_priors is None:
        hyper_priors = HyperPriorParams()
    elif not isinstance(hyper_priors, HyperPriorParams):
        msg = "hyper_priors must be a HyperPriorParams object."
        raise TypeError(msg)

    # Check that the observed data dimensions match between the data and the
    # parameters
    if not np.array_equal(params.y_dims, Y.dims):
        msg = "params.y_dims must match Y.dims."
        raise ValueError(msg)

    # Check that the initial latent dimensionality matches param.x_dim
    if params.x_dim != x_dim_init:
        msg = "params.x_dim must match x_dim_init."
        raise ValueError(msg)

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    N = Y.data.shape[1]  # Number of samples
    x_dim = params.x_dim  # Number of latent dimensions

    Y2 = np.sum(Y.data**2, axis=1)  # Sample second moments of observed data

    # Initialize the posterior covariance of C, if needed
    if params.C.cov is None:
        params.C.cov = np.zeros((y_dim, x_dim, x_dim))

    # Compute the variance floor for each observed dimension
    var_floor = min_var_frac * np.var(Y.data, axis=1, ddof=1)

    # Constant factors in the lower bound
    consts_lb = compute_lower_bound_constants(N, params, hyper_priors)

    # Initialize tracked quantities
    tracker = GFAFitTracker()
    if save_fit_progress:
        tracker.lb = np.array([])  # Lower bound
        tracker.iter_time = np.array([])  # Runtime per iteration
    lb_curr = -np.inf  # Initial lower bound

    # Initialize status flags
    flags = GFAFitFlags()

    fit_iter = 0
    for fit_iter in range(max_iter):
        # Check if any latents need to be removed
        if prune_X:
            # To be kept, the sample second moment of each latent must be
            # sufficiently large
            kept_x_dims = np.nonzero(np.mean(params.X.mean**2, axis=1) > prune_tol)[0]
            if len(kept_x_dims) < x_dim:
                # Remove inactive latents
                params.get_subset_dims(kept_x_dims, in_place=True)
                flags.x_dims_removed += x_dim - params.x_dim
                x_dim = params.x_dim
                if x_dim <= 0:
                    # Stop fitting if no significant latents remain
                    break

        # Start timer for current iteration
        if save_fit_progress:
            start_time = time.time()

        # Observation mean parameter, d
        infer_obs_mean(Y, params, hyper_priors, in_place=True)
        # Second moments, used for phi updates and the lower bound
        d_moment = params.d.cov + params.d.mean**2  # Only the diagonal is used

        # Latent variables, X
        infer_latents(Y, params, in_place=True)
        # Correlation matrix between current estimate of latents and
        # zero-centered observations. Used for C and phi updates.
        XY = params.X.mean @ (Y.data - params.d.mean[:, np.newaxis]).T

        # Loading matrices, C
        infer_loadings(Y, params, in_place=True, XY=XY)
        # Calculate the log-determinant of the covariance for the lower bound
        logdet_C = np.sum(np.linalg.slogdet(params.C.cov)[1])
        # Expected squared norm of each column of C. Used for ARD updates and
        # the lower bound.
        C_norm = params.C.compute_squared_norms(y_dims)

        # ARD parameters, alpha
        infer_ard(params, hyper_priors=hyper_priors, in_place=True, C_norm=C_norm)

        # Observation precision parameters, phi
        infer_obs_prec(
            Y, params, hyper_priors, in_place=True, d_moment=d_moment, XY=XY, Y2=Y2
        )
        # Set minimum private variance
        params.phi.mean[:] = np.minimum(1 / var_floor, params.phi.mean)
        params.phi.b[:] = params.phi.a / params.phi.mean

        # Compute the lower bound
        lb_old = lb_curr
        lb_curr = compute_lower_bound(
            Y,
            params,
            hyper_priors,
            consts=consts_lb,
            logdet_C=logdet_C,
            C_norm=C_norm,
            d_moment=d_moment,
        )

        # Save progress
        if save_fit_progress:
            # Compute the runtime of this iteration
            end_time = time.time()
            tracker.iter_time = np.append(tracker.iter_time, end_time - start_time)
            # Record the current lower bound
            tracker.lb = np.append(tracker.lb, lb_curr)

        # Display progress
        if verbose:
            print(
                f"\rIteration {fit_iter+1} of {max_iter}        lb {lb_curr}",
                end="",
                flush=True,
            )

        # Check stopping conditions or errors
        if fit_iter <= 1:
            lb_base = lb_curr
        elif lb_curr < lb_old:
            flags.decreasing_lb = True
        elif (lb_curr - lb_base) < (1 + fit_tol) * (lb_old - lb_base):
            flags.converged = True
            break

    # Display reasons for stopping
    if verbose:
        if flags.converged:
            print(f"\nLower bound converged after {fit_iter+1} iterations.")
        elif ((fit_iter + 1) < max_iter) and params.x_dim <= 0:
            print("\nFitting stopped because no significant latent dimensions remain.")
        else:
            print(f"\nFitting stopped after max_iter ({max_iter}) was reached.")

    # Check if the variance floor was reached by any observed dimension
    if np.any(params.phi.mean == 1 / var_floor):
        flags.private_var_floor = True

    if not save_C_cov:
        # Delete the loading matrix covariances to save memory
        params.C.cov = None

    if not save_X:
        # Delete the latent variable estimates to save memory
        params.X.clear()

    return (params, tracker, flags)


def init(
    Y: ObsData,
    x_dim_init: int = 1,
    hyper_priors: HyperPriorParams | None = None,
    random_seed: int | None = None,
    save_C_cov: bool = False,
) -> GFAParams:
    """
    Initialize GFA model parameters for fitting.

    Parameters
    ----------
    Y
        Observed data.
    x_dim_init
        Initial number of latent dimensions to fit (before pruning).
        Defaults to ``1``.
    hyper_priors
        Hyperparameters of the GFA prior distributions. If not provided,
        default hyperparameters will be used.
    random_seed
        Seed the random number generator for reproducibility. Defaults to
        ``None``.
    save_C_cov
        Set to ``True`` to save posterior covariance of :math:`C`. For large
        ``y_dim`` and ``x_dim``, these structures can use a lot of memory.
        Defaults to ``False``.

    Returns
    -------
    GFAParams
        Initialized GFA model parameters.

    Raises
    ------
    TypeError
        If ``hyper_priors`` is not a ``HyperPriorParams`` object.
    """
    # Initialize hyper_priors if not provided
    if hyper_priors is None:
        hyper_priors = HyperPriorParams()
    elif not isinstance(hyper_priors, HyperPriorParams):
        msg = "hyper_priors must be a HyperPriorParams object."
        raise TypeError(msg)

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    num_groups = len(y_dims)  # Number of observed groups
    N = Y.data.shape[1]  # Number of samples
    x_dim = x_dim_init  # Number of latent dimensions

    # Get views of the observed data for each group
    Ys = Y.get_groups()

    # Initialize GFA parameter object
    params = GFAParams(x_dim=x_dim, y_dims=y_dims)

    # Get the variance of each observed group
    Y_covs = [np.cov(Y_m) for Y_m in Ys]

    # Seed the random number generator for reproducible initialization.
    rng = np.random.default_rng(random_seed)

    # Latent variables
    params.X.mean = rng.normal(size=(x_dim, N))  # Mean
    params.X.cov = np.eye(x_dim)  # Covariance

    # Mean parameter
    params.d.mean = np.mean(Y.data, axis=1)
    params.d.cov = np.full(y_dim, 1 / hyper_priors.d_beta)

    # Noise precisions
    params.phi.a = hyper_priors.a_phi + N / 2
    params.phi.b = np.full(y_dim, hyper_priors.b_phi)
    params.phi.mean = np.concatenate([1 / np.diag(Y_cov) for Y_cov in Y_covs], axis=0)

    # Loading matrices

    # Mean
    params.C.mean = np.zeros((y_dim, x_dim))
    # Get views of the loading matrices for each group
    C_means, _, _ = params.C.get_groups(y_dims)
    for group_idx in range(num_groups):
        # The covariance of the current group might not be full rank
        # because N < y_dims[group_idx]. Scale the loading matrix according to
        # the geometric mean of the non-zero eigenvalues of the covariance.
        eigs = eigh(Y_covs[group_idx], eigvals_only=True)
        scale = gmean(eigs[eigs > 0])
        # Mean
        C_means[group_idx][:] = rng.normal(
            scale=np.sqrt(scale / x_dim), size=(y_dims[group_idx], x_dim)
        )

    # Covariance
    params.C.cov = np.zeros((y_dim, x_dim, x_dim))
    # Second moments
    params.C.compute_moment()
    # Get views of the loading matrix moments for each group
    _, _, C_moments = params.C.get_groups(y_dims)
    if not save_C_cov:
        # Delete the loading matrix covariances to save memory
        params.C.cov = None

    # ARD parameters
    params.alpha.a = hyper_priors.a_alpha + y_dims / 2
    params.alpha.b = np.full((num_groups, x_dim), hyper_priors.b_alpha)
    # Scale ARD parameters to match the data
    params.alpha.mean = np.zeros((num_groups, x_dim))
    for group_idx in range(num_groups):
        params.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    return params


def infer_latents(
    Y: ObsData,
    params: GFAParams,
    in_place: bool = True,
) -> PosteriorLatent | None:
    """
    Infer latent variables given GFA model parameters and observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    in_place
        If ``True``, update the posterior latents in place.
        If ``False``, compute the posterior latents and return as a
        new ``PosteriorLatent`` without modifying ``params``. Defaults to
        ``True``.

    Returns
    -------
    PosteriorLatent | None
        Posterior estimates of latent variables :math:`X`.
    """
    # Initialize X, if needed
    if in_place:
        if params.X.mean is None:
            params.X.mean = np.zeros((params.x_dim, Y.data.shape[1]))
        if params.X.cov is None:
            params.X.cov = np.zeros((params.x_dim, params.x_dim))
        if params.X.moment is None:
            params.X.moment = np.zeros((params.x_dim, params.x_dim))
        X = params.X
    else:
        X = PosteriorLatent(
            mean=np.zeros((params.x_dim, Y.data.shape[1])),
            cov=np.zeros((params.x_dim, params.x_dim)),
            moment=np.zeros((params.x_dim, params.x_dim)),
        )

    # Covariance
    X.cov[:] = np.linalg.inv(
        np.eye(params.x_dim)
        + np.sum(params.phi.mean[:, np.newaxis, np.newaxis] * params.C.moment, axis=0)
    )
    # Ensure symmetry
    X.cov[:] = 0.5 * (X.cov + X.cov.T)
    # Mean
    X.mean[:] = (
        X.cov
        @ (params.C.mean.T * params.phi.mean[np.newaxis, :])
        @ (Y.data - params.d.mean[:, np.newaxis])
    )
    # Second moment
    X.compute_moment(in_place=True)

    if not in_place:
        return X
    return None


def infer_loadings(
    Y: ObsData,
    params: GFAParams,
    in_place: bool = True,
    XY: np.ndarray | None = None,
) -> PosteriorLoading | None:
    """
    Infer loadings :math:`C` given current params and observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    in_place
        If ``True``, update the posterior loadings in place.
        If ``False``, compute the posterior loadings and return as a
        new ``PosteriorLoading`` without modifying ``params``. Defaults to
        ``True``.
    XY
        `ndarray` of `float`, shape ``(x_dim, y_dim)``.
        Correlation matrix between the latent variables and zero-centered
        observations. If not provided, it will be computed from ``params.X``
        and ``Y.data``.

    Returns
    -------
    PosteriorLoading | None
        Posterior estimates of loadings.
    """
    y_dim = params.y_dims.sum()  # Total number of observed dimensions
    x_dim = params.x_dim  # Number of latent dimensions
    num_groups = len(params.y_dims)  # Number of observed groups

    # Initialize C, if needed
    if in_place:
        if params.C.mean is None:
            params.C.mean = np.zeros((y_dim, x_dim))
        if params.C.cov is None:
            params.C.cov = np.zeros((y_dim, x_dim, x_dim))
        if params.C.moment is None:
            params.C.moment = np.zeros((y_dim, x_dim, x_dim))
        C = params.C
    else:
        C = PosteriorLoading(
            mean=np.zeros((y_dim, x_dim)),
            cov=np.zeros((y_dim, x_dim, x_dim)),
            moment=np.zeros((y_dim, x_dim, x_dim)),
        )

    # Correlation matrix between latents and zero-centered observations
    if XY is None:
        XY = params.X.mean @ (Y.data - params.d.mean[:, np.newaxis]).T

    # Get views of the loading matrices and precision parameters for each group
    _, C_covs, _ = C.get_groups(params.y_dims)
    phi_means, _ = params.phi.get_groups(params.y_dims)

    for group_idx in range(num_groups):
        # Covariance
        C_covs[group_idx][:] = np.linalg.inv(
            np.diag(params.alpha.mean[group_idx, :])
            + phi_means[group_idx][:, np.newaxis, np.newaxis] * params.X.moment
        )

    # Mean
    C.mean[:] = params.phi.mean[:, np.newaxis] * np.einsum("ijk,ij->ik", C.cov, XY.T)

    # Second moment
    C.compute_moment(in_place=True)

    if not in_place:
        return C
    return None


def infer_ard(
    params: GFAParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
    C_norm: np.ndarray | None = None,
) -> PosteriorARD | None:
    """
    Infer ARD parameters alpha given current params.

    Parameters
    ----------
    params
        GFA model parameters.
    hyper_priors
        Hyperparameters of the GFA prior distributions.
    in_place
        If ``True``, update the posterior ARD parameters in place.
        If ``False``, compute the posterior ARD parameters and return as a
        new ``PosteriorARD`` without modifying ``params``. Defaults to ``True``.
    C_norm
        `ndarray` of `float`, shape ``(num_groups, x_dim)``.
        ``C_norm[i,j]`` is the expected squared norm of column ``j`` of the
        loading matrix :math:`C` for group ``i``. If not provided, it will be
        computed from ``params.C``.

    Returns
    -------
    PosteriorARD | None
        Posterior estimates of ARD parameters.
    """
    num_groups = len(params.y_dims)  # Number of observed groups

    # Initialize alpha, if needed
    if in_place:
        if params.alpha.a is None:
            params.alpha.a = hyper_priors.a_alpha + params.y_dims / 2
        if params.alpha.b is None:
            params.alpha.b = np.zeros((num_groups, params.x_dim))
        if params.alpha.mean is None:
            params.alpha.mean = np.zeros((num_groups, params.x_dim))
        alpha = params.alpha
    else:
        alpha = PosteriorARD(
            a=hyper_priors.a_alpha + params.y_dims / 2,
            b=np.zeros((num_groups, params.x_dim)),
            mean=np.zeros((num_groups, params.x_dim)),
        )

    # Expected squared norm of each column of C
    if C_norm is None:
        C_norm = params.C.compute_squared_norms(params.y_dims)

    # Rate parameters
    alpha.b[:] = hyper_priors.b_alpha + 0.5 * C_norm

    # Mean
    alpha.compute_mean(in_place=True)

    if not in_place:
        return alpha
    return None


def infer_obs_mean(
    Y: ObsData,
    params: GFAParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
) -> PosteriorObsMean | None:
    """
    Infer observation mean parameter given GFA model parameters and observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    hyper_priors
        Hyperparameters of the GFA prior distributions.
    in_place
        If ``True``, update the posterior observation mean in place.
        If ``False``, compute the posterior observation mean and return as a
        new ``PosteriorObsMean`` without modifying ``params``. Defaults to
        ``True``.

    Returns
    -------
    PosteriorObsMean | None
        Posterior estimates of observation mean parameter.
    """
    y_dim, N = Y.data.shape

    # Initialize d, if needed
    if in_place:
        if params.d.mean is None:
            params.d.mean = np.zeros(y_dim)
        if params.d.cov is None:
            params.d.cov = np.zeros(y_dim)
        d = params.d
    else:
        d = PosteriorObsMean(mean=np.zeros(y_dim), cov=np.zeros(y_dim))

    # Covariance
    d.cov[:] = 1 / (hyper_priors.d_beta + N * params.phi.mean)
    d.mean[:] = (
        d.cov * params.phi.mean * np.sum(Y.data - params.C.mean @ params.X.mean, axis=1)
    )

    if not in_place:
        return d
    return None


def infer_obs_prec(
    Y: ObsData,
    params: GFAParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
    d_moment: np.ndarray | None = None,
    XY: np.ndarray | None = None,
    Y2: np.ndarray | None = None,
) -> PosteriorObsPrec | None:
    """
    Infer observation precision parameters given GFA model parameters and observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    hyper_priors
        Hyperparameters of the GFA prior distributions.
    in_place
        If ``True``, update the posterior observation precisions in place.
        If ``False``, compute the posterior observation precisions and return as
        a new ``PosteriorObsPrec`` without modifying ``params``.
    d_moment
        `ndarray` of `float`, shape ``(y_dim,)``.
        Second moment of the observation mean parameter. If not provided,
        it will be computed from ``params.d``.
    XY
        `ndarray` of `float`, shape ``(x_dim, y_dim)``.
        Correlation matrix between the latent variables and zero-centered
        observations. If not provided, it will be computed from ``params.X``
        and ``Y.data``.
    Y2
        `ndarray` of `float`, shape ``(y_dim,)``.
        Sample second moments of observed data. If not provided, it will be
        computed from ``Y.data``.

    Returns
    -------
    PosteriorObsPrec | None
        Posterior estimates of observation precision parameters.
    """
    y_dim, N = Y.data.shape

    # Initialize phi, if needed
    if in_place:
        if params.phi.mean is None:
            params.phi.mean = np.zeros(y_dim)
        if params.phi.a is None:
            params.phi.a = hyper_priors.a_phi + N / 2
        if params.phi.b is None:
            params.phi.b = np.zeros(y_dim)
        phi = params.phi
    else:
        phi = PosteriorObsPrec(
            mean=np.zeros(y_dim), a=hyper_priors.a_phi + N / 2, b=np.zeros(y_dim)
        )

    # Pre-computations
    # Sample second moments of observed data
    if Y2 is None:
        Y2 = np.sum(Y.data**2, axis=1)

    # Second moment of the observation mean parameter
    if d_moment is None:
        d_moment = params.d.cov + params.d.mean**2  # Only the diagonal is used

    # Correlation matrix between latents and zero-centered observations
    if XY is None:
        XY = params.X.mean @ (Y.data - params.d.mean[:, np.newaxis]).T

    # Rate parameter
    phi.b[:] = hyper_priors.b_phi + 0.5 * (
        N * d_moment
        + Y2
        - 2 * np.sum(params.d.mean[:, np.newaxis] * Y.data, axis=1)
        - 2 * np.sum(params.C.mean * XY.T, axis=1)
        + np.sum(params.C.moment * params.X.moment, axis=(1, 2))
    )

    # Mean
    phi.compute_mean(in_place=True)

    if not in_place:
        return phi
    return None


def compute_lower_bound(
    Y: ObsData,
    params: GFAParams,
    hyper_priors: HyperPriorParams,
    consts: tuple | None = None,
    logdet_C: float | None = None,
    C_norm: np.ndarray | None = None,
    d_moment: np.ndarray | None = None,
) -> float:
    """
    Compute the variational lower bound for a GFA model on observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    hyper_priors
        Hyperparameters of the GFA prior distributions.
    consts
        Constant factors in the lower bound. If not provided, they will be
        computed. See :func:`compute_lower_bound_constants` for details.
    logdet_C
        Log-determinant of the covariance of the loading matrices. If not
        provided, it will be computed from ``params.C``.
    C_norm
        `ndarray` of `float`, shape ``(num_groups, x_dim)``.
        ``C_norm[i,j]`` is the expected squared norm of column ``j`` of the
        loading matrix C for group ``i``. If not provided, it will be computed
        from ``params.C``.
    d_moment
        `ndarray` of `float`, shape ``(y_dim,)``.
        Second moment of the observation mean parameter. If not provided,
        it will be computed from ``params.d``.

    Returns
    -------
    float
        Variational lower bound.
    """
    y_dims = params.y_dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    x_dim = params.x_dim  # Number of latent dimensions
    num_groups = len(params.y_dims)  # Number of observed groups
    N = Y.data.shape[1]  # Number of samples

    # Constant factors in the lower bound
    if consts is None:
        consts = compute_lower_bound_constants(N, params, hyper_priors)

    (
        const_lik,
        const_d,
        alogb_phi,
        loggamma_a_phi_prior,
        loggamma_a_phi_post,
        digamma_a_phi,
        alogb_alpha,
        loggamma_a_alpha_prior,
        loggamma_a_alpha_post,
        digamma_a_alpha,
    ) = consts

    # Other pre-computations
    if logdet_C is None:
        logdet_C = np.sum(np.linalg.slogdet(params.C.cov)[1])
    if C_norm is None:
        C_norm = params.C.compute_squared_norms(y_dims)
    if d_moment is None:
        d_moment = params.d.cov + params.d.mean**2

    # Likelihood term
    log_phi = digamma_a_phi - np.log(params.phi.b)
    lb = (
        const_lik
        + 0.5 * N * np.sum(log_phi)
        - np.sum(params.phi.mean * (params.phi.b - hyper_priors.b_phi))
    )

    # X KL term
    lb += 0.5 * N * (x_dim + np.linalg.slogdet(params.X.cov)[1]) - 0.5 * np.trace(
        params.X.moment
    )

    log_alpha = digamma_a_alpha[:, np.newaxis] - np.log(params.alpha.b)

    # C KL term
    lb += 0.5 * (
        x_dim * y_dim
        + logdet_C
        + np.sum(y_dims[:, np.newaxis] * log_alpha - params.alpha.mean * C_norm)
    )

    # alpha KL term
    lb += (
        num_groups * x_dim * (alogb_alpha - loggamma_a_alpha_prior)
        + np.sum(
            -params.alpha.a[:, np.newaxis] * np.log(params.alpha.b)
            - hyper_priors.b_alpha * params.alpha.mean
            + (hyper_priors.a_alpha - params.alpha.a)[:, np.newaxis] * log_alpha
        )
        + np.sum(x_dim * (loggamma_a_alpha_post + params.alpha.a))
    )

    # phi KL term
    lb += y_dim * (
        alogb_phi + loggamma_a_phi_post - loggamma_a_phi_prior + params.phi.a
    ) + np.sum(
        -params.phi.a * np.log(params.phi.b)
        + hyper_priors.b_phi * params.phi.mean
        + (hyper_priors.a_phi - params.phi.a) * log_phi
    )

    # d KL term
    lb += const_d + 0.5 * (
        np.sum(np.log(params.d.cov)) - hyper_priors.d_beta * np.sum(d_moment)
    )

    return lb


def compute_lower_bound_constants(
    N: int,
    params: GFAParams,
    hyper_priors: HyperPriorParams,
) -> tuple[
    float, float, float, float, float, float, float, float, np.ndarray, np.ndarray
]:
    """
    Compute constant factors in the variational lower bound.

    Parameters
    ----------
    N
        Number of samples in the observed data.
    params
        GFA model parameters.
    hyper_priors
        Hyperparameters of the GFA prior distributions.

    Returns
    -------
    const_lik : float
        Constant factor related to the likelihood.
    const_d : float
        Constant factor related to the observation mean parameters.
    alogb_phi : float
        Constant factor related to the observation precision parameters.
    loggamma_a_phi_prior : float
        Constant factor related to the observation precision parameters.
    loggamma_a_phi_post : float
        Constant factor related to the observation precision parameters.
    digamma_a_phi : float
        Constant factor related to the observation precision parameters.
    alogb_alpha : float
        Constant factor related to the ARD parameters.
    loggamma_a_alpha_prior : float
        Constant factor related to the ARD parameters.
    loggamma_a_alpha_post : ndarray, shape ``(num_groups,)``
        Constant factor related to the ARD parameters.
    digamma_a_alpha : ndarray, shape ``(num_groups,)``
        Constant factor related to the ARD parameters.
    """
    y_dim = params.y_dims.sum()  # Total number of observed dimensions

    # Constant factors in the lower bound
    # Related to the likelihood
    const_lik = -(y_dim * N / 2) * np.log(2 * np.pi)
    # Related to observation mean parameters
    const_d = 0.5 * y_dim + 0.5 * y_dim * np.log(hyper_priors.d_beta)
    # Related to observation precision parameters
    alogb_phi = hyper_priors.a_phi * np.log(hyper_priors.b_phi)
    loggamma_a_phi_prior = gammaln(hyper_priors.a_phi)
    loggamma_a_phi_post = gammaln(params.phi.a)
    digamma_a_phi = psi(params.phi.a)
    # Related to ARD parameters
    alogb_alpha = hyper_priors.a_alpha * np.log(hyper_priors.b_alpha)
    loggamma_a_alpha_prior = gammaln(hyper_priors.a_alpha)
    loggamma_a_alpha_post = gammaln(params.alpha.a)
    digamma_a_alpha = psi(params.alpha.a)

    return (
        const_lik,
        const_d,
        alogb_phi,
        loggamma_a_phi_prior,
        loggamma_a_phi_post,
        digamma_a_phi,
        alogb_alpha,
        loggamma_a_alpha_prior,
        loggamma_a_alpha_post,
        digamma_a_alpha,
    )


class GFAModel:
    """
    A wrapper class to store a full set of GFA fitting results.

    Parameters
    ----------
    params
        Current GFA model parameters.
    tracker
        Contains quantities tracked during fitting.
    flags
        Contains status messages about the fitting process.
    fit_args
        Keyword arguments used to fit the model.

    Attributes
    ----------
    params
        Same as **params**, above.
    tracker
        Same as **tracker**, above.
    flags
        Same as **flags**, above.
    fit_args
        Same as **fit_args**, above.

    Raises
    ------
    TypeError
        If **params**, **tracker**, **flags**, or **fit_args** are not the
        respective types specified above.
    """

    def __init__(
        self,
        params: GFAParams | None = None,
        tracker: GFAFitTracker | None = None,
        flags: GFAFitFlags | None = None,
        fit_args: GFAFitArgs | None = None,
    ):
        # Estimated parameters
        if params is None:
            self.params = GFAParams()
        elif not isinstance(params, GFAParams):
            msg = "params must be a GFAParams object."
            raise TypeError(msg)
        else:
            self.params = params

        # Fit tracker
        if tracker is None:
            self.tracker = GFAFitTracker()
        elif not isinstance(tracker, GFAFitTracker):
            msg = "tracker must be a GFAFitTracker object."
            raise TypeError(msg)
        else:
            self.tracker = tracker

        # Fit flags
        if flags is None:
            self.flags = GFAFitFlags()
        elif not isinstance(flags, GFAFitFlags):
            msg = "flags must be a GFAFitFlags object."
            raise TypeError(msg)
        else:
            self.flags = flags

        # Fit keyword arguments
        if fit_args is None:
            self.fit_args = GFAFitArgs()
        elif not isinstance(fit_args, GFAFitArgs):
            msg = "fit_args must be a GFAFitArgs object."
            raise TypeError(msg)
        else:
            self.fit_args = fit_args

    def __repr__(self) -> str:
        return (
            f"GFAModel(params={self.params}, "
            f"tracker={self.tracker}, "
            f"flags={self.flags}, "
            f"fit_args={self.fit_args})"
        )

    def fit(self, Y: ObsData) -> None:
        """
        Fit a GFA model to data.

        Fit a GFA model to data. Uses the current model parameters as initial
        values, and uses the current keyword arguments.

        Parameters
        ----------
        Y
            Observed data.
        """
        # Initialize GFA model parameters if they have not been initialized
        if not self.params.is_initialized():
            if self.fit_args.verbose:
                print("GFA model parameters not initialized. Initializing...")
            self.init(Y)

        # Fit the model
        self.params, self.tracker, self.flags = fit(
            Y, self.params, **self.fit_args.get_args()
        )

    def init(self, Y: ObsData) -> None:
        """
        Initialize GFA model parameters.

        Parameters
        ----------
        Y
            Observed data.
        """
        # Get a subset of keyword arguments to pass to core.init_gfa
        init_args = ["x_dim_init", "hyper_priors", "random_seed", "save_C_cov"]
        kwargs = {
            key: value
            for key, value in self.fit_args.get_args().items()
            if key in init_args
        }
        self.params = init(Y, **kwargs)

    def save(self, filename: str, indent: int = 2) -> None:
        """
        Save a GFAModel object to a JSON file.

        Parameters
        ----------
        filename
            Name of JSON file to save to.
        indent
            Number of spaces to indent each line of the saved JSON file.
            Defaults to ``2``.
        """
        with open(filename, "w") as f:
            f.write(jsonpickle.encode(self, indent=indent))

    @staticmethod
    def load(filename: str) -> GFAModel:
        """
        Load a GFAModel object from a JSON file.

        Parameters
        ----------
        filename
            Name of JSON file to load from.

        Returns
        -------
        GFAModel
            Loaded GFAModel object.
        """
        with open(filename) as f:
            return jsonpickle.decode(f.read())

    def infer_latents(
        self,
        Y: ObsData,
        in_place: bool = True,
    ) -> PosteriorLatent | None:
        """
        Infer latent variables X given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update the posterior latents in place.
            If ``False``, compute the posterior latents and return as a
            new ``PosteriorLatent`` without modifying ``params``. Defaults to
            ``True``.

        Returns
        -------
        PosteriorLatent | None
            Posterior estimates of latent variables.
        """
        return infer_latents(Y, self.params, in_place=in_place)

    def infer_loadings(
        self,
        Y: ObsData,
        in_place: bool = True,
    ) -> PosteriorLoading | None:
        """
        Infer loadings C given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update the posterior loadings in place.
            If ``False``, compute the posterior loadings and return as a
            new ``PosteriorLoading`` without modifying ``params``. Defaults to
            ``True``.

        Returns
        -------
        PosteriorLoading | None
            Posterior estimates of loadings.
        """
        return infer_loadings(Y, self.params, in_place=in_place)

    def infer_ard(
        self,
        in_place: bool = True,
    ) -> PosteriorARD | None:
        """
        Infer ARD parameters alpha given current params.

        Parameters
        ----------
        in_place
            If ``True``, update the posterior ARD parameters in place.
            If ``False``, compute the posterior ARD parameters and return as a
            new ``PosteriorARD`` without modifying ``params``. Defaults to
            ``True``.

        Returns
        -------
        PosteriorARD | None
            Posterior estimates of ARD parameters.
        """
        return infer_ard(self.params, self.fit_args.hyper_priors, in_place=in_place)

    def infer_obs_mean(
        self,
        Y: ObsData,
        in_place: bool = True,
    ) -> PosteriorObsMean | None:
        """
        Infer observation mean parameters given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update the posterior observation mean parameters in
            place.
            If ``False``, compute the posterior observation mean parameters and
            return as a new ``PosteriorObsMean`` without modifying ``params``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorObsMean | None
            Posterior estimates of observation mean parameters.
        """
        return infer_obs_mean(
            Y, self.params, self.fit_args.hyper_priors, in_place=in_place
        )

    def infer_obs_prec(
        self,
        Y: ObsData,
        in_place: bool = True,
    ) -> PosteriorObsPrec | None:
        """
        Infer observation precision parameters given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update the posterior observation precision parameters
            in place.
            If ``False``, compute the posterior observation precision parameters
            and return as a new ``PosteriorObsPrec`` without modifying
            ``params``. Defaults to ``True``.

        Returns
        -------
        PosteriorObsPrec | None
            Posterior estimates of observation precision parameters.
        """
        return infer_obs_prec(
            Y, self.params, self.fit_args.hyper_priors, in_place=in_place
        )

    def compute_lower_bound(
        self,
        Y: ObsData,
    ) -> float:
        """
        Compute the variational lower bound given observed data.

        Parameters
        ----------
        Y
            Observed data.

        Returns
        -------
        float
            Variational lower bound.
        """
        return compute_lower_bound(Y, self.params, self.fit_args.hyper_priors)
