"""Core utilities to fit a group factor analysis (GFA) model to data."""

from __future__ import annotations

import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln, psi
from scipy.stats import gmean

from latents._core.numerics import stability_floor, validate_tolerance
from latents.gfa.config import GFAFitConfig
from latents.gfa.data_types import (
    GFAFitFlags,
    GFAFitTracker,
    GFAParams,
)
from latents.observation import (
    ARDPosterior,
    LoadingPosterior,
    ObsMeanPosterior,
    ObsParamsHyperPrior,
    ObsPrecPosterior,
)
from latents.observation_model.observations import ObsStatic
from latents.state_model.latents import PosteriorLatentStatic

jsonpickle_numpy.register_handlers()


def fit(
    Y: ObsStatic,
    params: GFAParams,
    config: GFAFitConfig | None = None,
    hyper_priors: ObsParamsHyperPrior | None = None,
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
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    hyper_priors
        Prior hyperparameters. If None, uses default ObsParamsHyperPrior().

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
    ValueError
        If ``params.y_dims`` does not match ``Y.dims`` or if ``params.x_dim``
        does not match ``config.x_dim_init``.
    """
    # Use defaults if not provided
    if config is None:
        config = GFAFitConfig()
    if hyper_priors is None:
        hyper_priors = ObsParamsHyperPrior()

    # Unpack config for local use (maintains readability in existing code)
    x_dim_init = config.x_dim_init
    fit_tol = config.fit_tol
    max_iter = config.max_iter
    verbose = config.verbose
    min_var_frac = config.min_var_frac
    prune_x = config.prune_x
    prune_tol = config.prune_tol
    save_x = config.save_x
    save_c_cov = config.save_c_cov
    save_fit_progress = config.save_fit_progress

    # Validate tolerances against data precision
    validate_tolerance(fit_tol, Y.data.dtype, "fit_tol")
    validate_tolerance(prune_tol, Y.data.dtype, "prune_tol")

    # Initialize GFA model parameters if they have not been initialized already
    if not params.is_initialized():
        if verbose:
            print("GFA model parameters not initialized. Initializing...")
        params = init(Y, config=config, hyper_priors=hyper_priors)
    obs_params = params.obs_params
    state_params = params.state_params

    # Check that the observed data dimensions match between the data and the
    # parameters
    if not np.array_equal(obs_params.y_dims, Y.dims):
        msg = "params.obs_params.y_dims must match Y.dims."
        raise ValueError(msg)

    # Check that the initial latent dimensionality matches param.x_dim
    if obs_params.x_dim != x_dim_init:
        msg = "params.obs_params.x_dim must match config.x_dim_init."
        raise ValueError(msg)

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    n_samples = Y.data.shape[1]  # Number of samples
    x_dim = obs_params.x_dim  # Number of latent dimensions

    Y2 = np.sum(Y.data**2, axis=1)  # Sample second moments of observed data

    # Initialize the posterior covariance of C, if needed
    if obs_params.C.cov is None:
        obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))

    # Compute the variance floor for each observed dimension
    # Apply stability_floor() to handle constant features (zero variance)
    floor = stability_floor(Y.data.dtype)
    var_floor = np.maximum(min_var_frac * np.var(Y.data, axis=1, ddof=1), floor)

    # Constant factors in the lower bound
    consts_lb = compute_lower_bound_constants(n_samples, params, hyper_priors)

    # Initialize tracked quantities
    tracker = GFAFitTracker()
    if save_fit_progress:
        # Pre-allocate arrays to max_iter; will truncate after loop
        tracker.lb = np.empty(max_iter)
        tracker.iter_time = np.empty(max_iter)
    lb_curr = -np.inf  # Initial lower bound

    # Initialize status flags
    flags = GFAFitFlags()

    fit_iter = 0
    for fit_iter in range(max_iter):
        # Check if any latents need to be removed
        if prune_x:
            # To be kept, the sample second moment of each latent must be
            # sufficiently large
            kept_x_dims = np.nonzero(
                np.mean(state_params.X.mean**2, axis=1) > prune_tol
            )[0]
            if len(kept_x_dims) < x_dim:
                # Remove inactive latents
                params.get_subset_dims(kept_x_dims, in_place=True)
                flags.x_dims_removed += x_dim - obs_params.x_dim
                x_dim = obs_params.x_dim
                if x_dim <= 0:
                    # Stop fitting if no significant latents remain
                    break

        # Start timer for current iteration
        if save_fit_progress:
            start_time = time.time()

        # Observation mean parameter, d
        infer_obs_mean(Y, params, hyper_priors, in_place=True)
        # Second moments, used for phi updates and the lower bound.
        # Only the diagonal is used.
        d_moment = obs_params.d.cov + obs_params.d.mean**2

        # Latent variables, X
        infer_latents(Y, params, in_place=True)
        # Correlation matrix between current estimate of latents and
        # zero-centered observations. Used for C and phi updates.
        # X.mean: (x_dim, n_samples)
        # d.mean: (y_dim,) -> broadcast to (y_dim, n_samples)
        # Result XY: (x_dim, n_samples) @ (n_samples, y_dim) -> (x_dim, y_dim)
        XY = state_params.X.mean @ (Y.data - obs_params.d.mean[:, np.newaxis]).T

        # Loading matrices, C
        infer_loadings(Y, params, in_place=True, XY=XY)
        # Calculate the log-determinant of the covariance for the lower bound
        logdet_C = np.sum(np.linalg.slogdet(obs_params.C.cov)[1])
        # Expected squared norm of each column of C. Used for ARD updates and
        # the lower bound.
        C_norm = obs_params.C.compute_squared_norms(y_dims)

        # ARD parameters, alpha
        infer_ard(params, hyper_priors=hyper_priors, in_place=True, C_norm=C_norm)

        # Observation precision parameters, phi
        infer_obs_prec(
            Y, params, hyper_priors, in_place=True, d_moment=d_moment, XY=XY, Y2=Y2
        )
        # Set minimum private variance
        obs_params.phi.mean[:] = np.minimum(1 / var_floor, obs_params.phi.mean)
        obs_params.phi.b[:] = obs_params.phi.a / obs_params.phi.mean

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
            end_time = time.time()
            tracker.iter_time[fit_iter] = end_time - start_time
            tracker.lb[fit_iter] = lb_curr

        # Display progress
        if verbose:
            print(
                f"\rIteration {fit_iter + 1} of {max_iter}        lb {lb_curr}",
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

    # Truncate pre-allocated arrays to actual iteration count
    if save_fit_progress:
        n_iters = fit_iter + 1
        tracker.lb = tracker.lb[:n_iters]
        tracker.iter_time = tracker.iter_time[:n_iters]

    # Display reasons for stopping
    if verbose:
        if flags.converged:
            print(f"\nLower bound converged after {fit_iter + 1} iterations.")
        elif ((fit_iter + 1) < max_iter) and params.x_dim <= 0:
            print("\nFitting stopped because no significant latent dimensions remain.")
        else:
            print(f"\nFitting stopped after max_iter ({max_iter}) was reached.")

    # Check if the variance floor was reached by any observed dimension
    if np.any(obs_params.phi.mean == 1 / var_floor):
        flags.private_var_floor = True

    if not save_c_cov:
        # Delete the loading matrix covariances to save memory
        obs_params.C.cov = None

    if not save_x:
        # Delete the latent variable estimates to save memory
        state_params.X.clear()

    return (params, tracker, flags)


def init(
    Y: ObsStatic,
    config: GFAFitConfig | None = None,
    hyper_priors: ObsParamsHyperPrior | None = None,
) -> GFAParams:
    """Initialize GFA model parameters for fitting.

    Parameters
    ----------
    Y
        Observed data.
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    hyper_priors
        Prior hyperparameters. If None, uses default ObsParamsHyperPrior().

    Returns
    -------
    GFAParams
        Initialized GFA model parameters.
    """
    # Use defaults if not provided
    if config is None:
        config = GFAFitConfig()
    if hyper_priors is None:
        hyper_priors = ObsParamsHyperPrior()

    # Unpack config for local use
    x_dim_init = config.x_dim_init
    random_seed = config.random_seed
    save_c_cov = config.save_c_cov

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    n_groups = len(y_dims)  # Number of observed groups
    n_samples = Y.data.shape[1]  # Number of samples
    x_dim = x_dim_init  # Number of latent dimensions

    # Get views of the observed data for each group
    Ys = Y.get_groups()

    # Initialize GFA parameter object
    params = GFAParams(x_dim=x_dim, y_dims=y_dims)
    obs_params = params.obs_params
    state_params = params.state_params

    # Get the variance of each observed group
    Y_covs = [np.cov(Y_m) for Y_m in Ys]

    # Seed the random number generator for reproducible initialization.
    rng = np.random.default_rng(random_seed)

    # Latent variables
    state_params.X.mean = rng.normal(size=(x_dim, n_samples))  # Mean
    state_params.X.cov = np.eye(x_dim)  # Covariance

    # Mean parameter
    obs_params.d.mean = np.mean(Y.data, axis=1)
    obs_params.d.cov = np.full(y_dim, 1 / hyper_priors.beta_d)

    # Noise precisions
    obs_params.phi.a = hyper_priors.a_phi + n_samples / 2
    obs_params.phi.b = np.full(y_dim, hyper_priors.b_phi)
    obs_params.phi.mean = np.concatenate(
        [1 / np.diag(Y_cov) for Y_cov in Y_covs], axis=0
    )

    # Loading matrices

    # Mean
    obs_params.C.mean = np.zeros((y_dim, x_dim))
    # Get views of the loading matrices for each group
    C_means, _, _ = obs_params.C.get_groups(y_dims)
    for group_idx in range(n_groups):
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
    obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))
    # Second moments
    obs_params.C.compute_moment()
    # Get views of the loading matrix moments for each group
    _, _, C_moments = obs_params.C.get_groups(y_dims)
    if not save_c_cov:
        # Delete the loading matrix covariances to save memory
        obs_params.C.cov = None

    # ARD parameters
    obs_params.alpha.a = hyper_priors.a_alpha + y_dims / 2
    obs_params.alpha.b = np.full((n_groups, x_dim), hyper_priors.b_alpha)
    # Scale ARD parameters to match the data
    obs_params.alpha.mean = np.zeros((n_groups, x_dim))
    for group_idx in range(n_groups):
        obs_params.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    return params


def infer_latents(
    Y: ObsStatic,
    params: GFAParams,
    in_place: bool = True,
) -> PosteriorLatentStatic:
    """
    Infer latent variables given GFA model parameters and observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    in_place
        If ``True``, update ``params.state_params.X`` in place and return it.
        If ``False``, return a new ``PosteriorLatentStatic``.
        Defaults to ``True``.

    Returns
    -------
    PosteriorLatentStatic
        Posterior estimates of latent variables :math:`X`.
    """
    obs_params = params.obs_params
    state_params = params.state_params

    # Initialize X, if needed
    if in_place:
        if state_params.X.mean is None:
            state_params.X.mean = np.zeros((state_params.x_dim, Y.data.shape[1]))
        if state_params.X.cov is None:
            state_params.X.cov = np.zeros((state_params.x_dim, state_params.x_dim))
        if state_params.X.moment is None:
            state_params.X.moment = np.zeros((state_params.x_dim, state_params.x_dim))
        X = state_params.X
    else:
        X = PosteriorLatentStatic(
            mean=np.zeros((state_params.x_dim, Y.data.shape[1])),
            cov=np.zeros((state_params.x_dim, state_params.x_dim)),
            moment=np.zeros((state_params.x_dim, state_params.x_dim)),
        )

    # Covariance: inv(I + sum_j phi_j * E[C_j^T C_j])
    # phi.mean: (y_dim,) -> (y_dim, 1, 1), C.moment: (y_dim, x_dim, x_dim)
    # Weighted sum over y_dim -> (x_dim, x_dim)
    X.cov[:] = np.linalg.inv(
        np.eye(state_params.x_dim)
        + np.sum(
            obs_params.phi.mean[:, np.newaxis, np.newaxis] * obs_params.C.moment,
            axis=0,
        )
    )
    # Ensure symmetry
    X.cov[:] = 0.5 * (X.cov + X.cov.T)
    # Mean: X.cov @ C^T diag(phi) @ (Y - d) -> X.mean: (x_dim, n_samples)
    # phi: (y_dim,) -> (1, y_dim) for broadcast with C.mean.T: (x_dim, y_dim)
    # d: (y_dim,) -> (y_dim, 1) for broadcast with Y: (y_dim, n_samples)
    X.mean[:] = (
        X.cov
        @ (obs_params.C.mean.T * obs_params.phi.mean[np.newaxis, :])
        @ (Y.data - obs_params.d.mean[:, np.newaxis])
    )
    # Second moment
    X.compute_moment(in_place=True)

    return X


def infer_loadings(
    Y: ObsStatic,
    params: GFAParams,
    in_place: bool = True,
    XY: np.ndarray | None = None,
) -> LoadingPosterior:
    """
    Infer loadings :math:`C` given current params and observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        GFA model parameters.
    in_place
        If ``True``, update ``params.obs_params.C`` in place and return it.
        If ``False``, return a new ``PosteriorLoading``.
        Defaults to ``True``.
    XY
        `ndarray` of `float`, shape ``(x_dim, y_dim)``.
        Correlation matrix between the latent variables and zero-centered
        observations. If not provided, it will be computed from ``params.X``
        and ``Y.data``.

    Returns
    -------
    PosteriorLoading
        Posterior estimates of loadings.
    """
    obs_params = params.obs_params
    state_params = params.state_params

    y_dim = obs_params.y_dims.sum()  # Total number of observed dimensions
    x_dim = obs_params.x_dim  # Number of latent dimensions
    n_groups = len(obs_params.y_dims)  # Number of observed groups

    # Initialize C, if needed
    if in_place:
        if obs_params.C.mean is None:
            obs_params.C.mean = np.zeros((y_dim, x_dim))
        if obs_params.C.cov is None:
            obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))
        if obs_params.C.moment is None:
            obs_params.C.moment = np.zeros((y_dim, x_dim, x_dim))
        C = obs_params.C
    else:
        C = LoadingPosterior(
            mean=np.zeros((y_dim, x_dim)),
            cov=np.zeros((y_dim, x_dim, x_dim)),
            moment=np.zeros((y_dim, x_dim, x_dim)),
        )

    # Correlation matrix between latents and zero-centered observations
    # X.mean: (x_dim, n_samples), (Y - d): (y_dim, n_samples) -> XY: (x_dim, y_dim)
    if XY is None:
        XY = state_params.X.mean @ (Y.data - obs_params.d.mean[:, np.newaxis]).T

    # Get views of the loading matrices and precision parameters for each group
    _, C_covs, _ = C.get_groups(obs_params.y_dims)
    phi_means, _ = obs_params.phi.get_groups(obs_params.y_dims)

    for group_idx in range(n_groups):
        # Covariance: inv(diag(alpha) + phi * E[X X^T]) -> (y_dim_m, x_dim, x_dim)
        # phi: (y_dim_m,) -> (y_dim_m, 1, 1) for broadcast with X.moment: (x_dim, x_dim)
        C_covs[group_idx][:] = np.linalg.inv(
            np.diag(obs_params.alpha.mean[group_idx, :])
            + phi_means[group_idx][:, np.newaxis, np.newaxis] * state_params.X.moment
        )

    # Mean: phi * einsum(C.cov, XY) -> C.mean: (y_dim, x_dim)
    # phi: (y_dim,) -> (y_dim, 1) for broadcast
    # einsum "ijk,ij->ik": contract over k dimension
    C.mean[:] = obs_params.phi.mean[:, np.newaxis] * np.einsum(
        "ijk,ij->ik", C.cov, XY.T
    )

    # Second moment
    C.compute_moment(in_place=True)

    return C


def infer_ard(
    params: GFAParams,
    hyper_priors: ObsParamsHyperPrior,
    in_place: bool = True,
    C_norm: np.ndarray | None = None,
) -> ARDPosterior:
    """
    Infer ARD parameters alpha given current params.

    Parameters
    ----------
    params
        GFA model parameters.
    hyper_priors
        Hyperparameters of the GFA prior distributions.
    in_place
        If ``True``, update ``params.obs_params.alpha`` in place and return it.
        If ``False``, return a new ``PosteriorARD``.
        Defaults to ``True``.
    C_norm
        `ndarray` of `float`, shape ``(n_groups, x_dim)``.
        ``C_norm[i,j]`` is the expected squared norm of column ``j`` of the
        loading matrix :math:`C` for group ``i``. If not provided, it will be
        computed from ``params.C``.

    Returns
    -------
    PosteriorARD
        Posterior estimates of ARD parameters.
    """
    obs_params = params.obs_params
    n_groups = len(obs_params.y_dims)  # Number of observed groups

    # Initialize alpha, if needed
    if in_place:
        if obs_params.alpha.a is None:
            obs_params.alpha.a = hyper_priors.a_alpha + obs_params.y_dims / 2
        if obs_params.alpha.b is None:
            obs_params.alpha.b = np.zeros((n_groups, obs_params.x_dim))
        if obs_params.alpha.mean is None:
            obs_params.alpha.mean = np.zeros((n_groups, obs_params.x_dim))
        alpha = obs_params.alpha
    else:
        alpha = ARDPosterior(
            a=hyper_priors.a_alpha + obs_params.y_dims / 2,
            b=np.zeros((n_groups, obs_params.x_dim)),
            mean=np.zeros((n_groups, obs_params.x_dim)),
        )

    # Expected squared norm of each column of C
    if C_norm is None:
        C_norm = obs_params.C.compute_squared_norms(obs_params.y_dims)

    # Rate parameters
    alpha.b[:] = hyper_priors.b_alpha + 0.5 * C_norm

    # Mean
    alpha.compute_mean(in_place=True)

    return alpha


def infer_obs_mean(
    Y: ObsStatic,
    params: GFAParams,
    hyper_priors: ObsParamsHyperPrior,
    in_place: bool = True,
) -> ObsMeanPosterior:
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
        If ``True``, update ``params.obs_params.d`` in place and return it.
        If ``False``, return a new ``PosteriorObsMean``.
        Defaults to ``True``.

    Returns
    -------
    PosteriorObsMean
        Posterior estimates of observation mean parameter.
    """
    obs_params = params.obs_params
    state_params = params.state_params
    y_dim, n_samples = Y.data.shape

    # Initialize d, if needed
    if in_place:
        if obs_params.d.mean is None:
            obs_params.d.mean = np.zeros(y_dim)
        if obs_params.d.cov is None:
            obs_params.d.cov = np.zeros(y_dim)
        d = obs_params.d
    else:
        d = ObsMeanPosterior(mean=np.zeros(y_dim), cov=np.zeros(y_dim))

    # Covariance
    d.cov[:] = 1 / (hyper_priors.beta_d + n_samples * obs_params.phi.mean)
    d.mean[:] = (
        d.cov
        * obs_params.phi.mean
        * np.sum(Y.data - obs_params.C.mean @ state_params.X.mean, axis=1)
    )

    return d


def infer_obs_prec(
    Y: ObsStatic,
    params: GFAParams,
    hyper_priors: ObsParamsHyperPrior,
    in_place: bool = True,
    d_moment: np.ndarray | None = None,
    XY: np.ndarray | None = None,
    Y2: np.ndarray | None = None,
) -> ObsPrecPosterior:
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
        If ``True``, update ``params.obs_params.phi`` in place and return it.
        If ``False``, return a new ``PosteriorObsPrec``.
        Defaults to ``True``.
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
    PosteriorObsPrec
        Posterior estimates of observation precision parameters.
    """
    obs_params = params.obs_params
    state_params = params.state_params
    y_dim, n_samples = Y.data.shape

    # Initialize phi, if needed
    if in_place:
        if obs_params.phi.mean is None:
            obs_params.phi.mean = np.zeros(y_dim)
        if obs_params.phi.a is None:
            obs_params.phi.a = hyper_priors.a_phi + n_samples / 2
        if obs_params.phi.b is None:
            obs_params.phi.b = np.zeros(y_dim)
        phi = obs_params.phi
    else:
        phi = ObsPrecPosterior(
            mean=np.zeros(y_dim),
            a=hyper_priors.a_phi + n_samples / 2,
            b=np.zeros(y_dim),
        )

    # Pre-computations
    # Sample second moments of observed data
    if Y2 is None:
        Y2 = np.sum(Y.data**2, axis=1)

    # Second moment of the observation mean parameter
    if d_moment is None:
        d_moment = obs_params.d.cov + obs_params.d.mean**2  # Only the diagonal is used

    # Correlation matrix between latents and zero-centered observations
    # X.mean: (x_dim, n_samples), (Y - d): (y_dim, n_samples) -> XY: (x_dim, y_dim)
    if XY is None:
        XY = state_params.X.mean @ (Y.data - obs_params.d.mean[:, np.newaxis]).T

    # Rate parameter: expected reconstruction error -> phi.b: (y_dim,)
    # d: (y_dim,) -> (y_dim, 1) for broadcast with Y: (y_dim, n_samples)
    phi.b[:] = hyper_priors.b_phi + 0.5 * (
        n_samples * d_moment
        + Y2
        - 2 * np.sum(obs_params.d.mean[:, np.newaxis] * Y.data, axis=1)
        - 2 * np.sum(obs_params.C.mean * XY.T, axis=1)
        + np.sum(obs_params.C.moment * state_params.X.moment, axis=(1, 2))
    )

    # Mean
    phi.compute_mean(in_place=True)

    return phi


def compute_lower_bound(
    Y: ObsStatic,
    params: GFAParams,
    hyper_priors: ObsParamsHyperPrior,
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
        `ndarray` of `float`, shape ``(n_groups, x_dim)``.
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
    obs_params = params.obs_params
    state_params = params.state_params
    y_dims = obs_params.y_dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    x_dim = obs_params.x_dim  # Number of latent dimensions
    n_groups = len(obs_params.y_dims)  # Number of observed groups
    n_samples = Y.data.shape[1]  # Number of samples

    # Constant factors in the lower bound
    if consts is None:
        consts = compute_lower_bound_constants(n_samples, params, hyper_priors)

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
        logdet_C = np.sum(np.linalg.slogdet(obs_params.C.cov)[1])
    if C_norm is None:
        C_norm = obs_params.C.compute_squared_norms(y_dims)
    if d_moment is None:
        d_moment = obs_params.d.cov + obs_params.d.mean**2

    # Stability floor for log operations on rate parameters
    floor = stability_floor(obs_params.phi.b.dtype)

    # Likelihood term
    log_phi = digamma_a_phi - np.log(np.maximum(obs_params.phi.b, floor))
    lb = (
        const_lik
        + 0.5 * n_samples * np.sum(log_phi)
        - np.sum(obs_params.phi.mean * (obs_params.phi.b - hyper_priors.b_phi))
    )

    # X KL term
    lb += 0.5 * n_samples * (
        x_dim + np.linalg.slogdet(state_params.X.cov)[1]
    ) - 0.5 * np.trace(state_params.X.moment)

    # digamma_a_alpha: (n_groups,) -> (n_groups, 1) for broadcast -> (n_groups, x_dim)
    log_alpha = digamma_a_alpha[:, np.newaxis] - np.log(
        np.maximum(obs_params.alpha.b, floor)
    )

    # C KL term
    # y_dims: (n_groups,) -> (n_groups, 1) for broadcast with log_alpha
    lb += 0.5 * (
        x_dim * y_dim
        + logdet_C
        + np.sum(y_dims[:, np.newaxis] * log_alpha - obs_params.alpha.mean * C_norm)
    )

    # alpha KL term
    # alpha.a: (n_groups,) -> (n_groups, 1) for broadcast with alpha.b
    lb += (
        n_groups * x_dim * (alogb_alpha - loggamma_a_alpha_prior)
        + np.sum(
            -obs_params.alpha.a[:, np.newaxis]
            * np.log(np.maximum(obs_params.alpha.b, floor))
            - hyper_priors.b_alpha * obs_params.alpha.mean
            + (hyper_priors.a_alpha - obs_params.alpha.a)[:, np.newaxis] * log_alpha
        )
        + np.sum(x_dim * (loggamma_a_alpha_post + obs_params.alpha.a))
    )

    # phi KL term
    lb += y_dim * (
        alogb_phi + loggamma_a_phi_post - loggamma_a_phi_prior + obs_params.phi.a
    ) + np.sum(
        -obs_params.phi.a * np.log(np.maximum(obs_params.phi.b, floor))
        + hyper_priors.b_phi * obs_params.phi.mean
        + (hyper_priors.a_phi - obs_params.phi.a) * log_phi
    )

    # d KL term
    lb += const_d + 0.5 * (
        np.sum(np.log(obs_params.d.cov)) - hyper_priors.beta_d * np.sum(d_moment)
    )

    return lb


def compute_lower_bound_constants(
    n_samples: int,
    params: GFAParams,
    hyper_priors: ObsParamsHyperPrior,
) -> tuple[
    float, float, float, float, float, float, float, float, np.ndarray, np.ndarray
]:
    """
    Compute constant factors in the variational lower bound.

    Parameters
    ----------
    n_samples
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
    loggamma_a_alpha_post : ndarray, shape ``(n_groups,)``
        Constant factor related to the ARD parameters.
    digamma_a_alpha : ndarray, shape ``(n_groups,)``
        Constant factor related to the ARD parameters.
    """
    obs_params = params.obs_params
    y_dim = obs_params.y_dims.sum()  # Total number of observed dimensions

    # Constant factors in the lower bound
    # Related to the likelihood
    const_lik = -(y_dim * n_samples / 2) * np.log(2 * np.pi)
    # Related to observation mean parameters
    const_d = 0.5 * y_dim + 0.5 * y_dim * np.log(hyper_priors.beta_d)
    # Related to observation precision parameters
    alogb_phi = hyper_priors.a_phi * np.log(hyper_priors.b_phi)
    loggamma_a_phi_prior = gammaln(hyper_priors.a_phi)
    loggamma_a_phi_post = gammaln(obs_params.phi.a)
    digamma_a_phi = psi(obs_params.phi.a)
    # Related to ARD parameters
    alogb_alpha = hyper_priors.a_alpha * np.log(hyper_priors.b_alpha)
    loggamma_a_alpha_prior = gammaln(hyper_priors.a_alpha)
    loggamma_a_alpha_post = gammaln(obs_params.alpha.a)
    digamma_a_alpha = psi(obs_params.alpha.a)

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
    """Interface with, fit, and store the fitting results of a GFA model.

    Parameters
    ----------
    params
        Current GFA model parameters.
    tracker
        Contains quantities tracked during fitting.
    flags
        Contains status messages about the fitting process.
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    hyper_priors
        Prior hyperparameters. If None, uses default ObsParamsHyperPrior().

    Attributes
    ----------
    params
        Same as **params**, above.
    tracker
        Same as **tracker**, above.
    flags
        Same as **flags**, above.
    config
        Same as **config**, above.
    hyper_priors
        Same as **hyper_priors**, above.

    Examples
    --------
    >>> from latents.gfa import GFAModel, GFAFitConfig
    >>> config = GFAFitConfig(x_dim_init=10, verbose=True)
    >>> model = GFAModel(config=config)
    >>> model.fit(Y)
    """

    def __init__(
        self,
        params: GFAParams | None = None,
        tracker: GFAFitTracker | None = None,
        flags: GFAFitFlags | None = None,
        config: GFAFitConfig | None = None,
        hyper_priors: ObsParamsHyperPrior | None = None,
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

        # Fitting configuration (immutable)
        self.config = config if config is not None else GFAFitConfig()

        # Prior hyperparameters (immutable)
        self.hyper_priors = (
            hyper_priors if hyper_priors is not None else ObsParamsHyperPrior()
        )

    def __repr__(self) -> str:
        return (
            f"GFAModel(params={self.params}, "
            f"tracker={self.tracker}, "
            f"flags={self.flags})"
        )

    def fit(self, Y: ObsStatic) -> None:
        """Fit a GFA model to data.

        Uses the current model parameters as initial values. Configuration
        and hyperparameters are set at model construction.

        Parameters
        ----------
        Y
            Observed data.
        """
        # Initialize GFA model parameters if they have not been initialized
        if not self.params.is_initialized():
            if self.config.verbose:
                print("GFA model parameters not initialized. Initializing...")
            self.init(Y)

        # Fit the model
        self.params, self.tracker, self.flags = fit(
            Y, self.params, config=self.config, hyper_priors=self.hyper_priors
        )

    def init(self, Y: ObsStatic) -> None:
        """Initialize GFA model parameters.

        Parameters
        ----------
        Y
            Observed data.
        """
        self.params = init(Y, config=self.config, hyper_priors=self.hyper_priors)

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
        Y: ObsStatic,
        in_place: bool = True,
    ) -> PosteriorLatentStatic:
        """
        Infer latent variables X given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update ``self.params.state_params.X`` in place and return it.
            If ``False``, return a new ``PosteriorLatentStatic``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorLatentStatic
            Posterior estimates of latent variables.
        """
        return infer_latents(Y, self.params, in_place=in_place)

    def infer_loadings(
        self,
        Y: ObsStatic,
        in_place: bool = True,
    ) -> LoadingPosterior:
        """
        Infer loadings C given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update ``self.params.obs_params.C`` in place and return it.
            If ``False``, return a new ``PosteriorLoading``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorLoading
            Posterior estimates of loadings.
        """
        return infer_loadings(Y, self.params, in_place=in_place)

    def infer_ard(
        self,
        in_place: bool = True,
    ) -> ARDPosterior:
        """
        Infer ARD parameters alpha given current params.

        Parameters
        ----------
        in_place
            If ``True``, update ``self.params.obs_params.alpha`` in place and return it.
            If ``False``, return a new ``PosteriorARD``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorARD
            Posterior estimates of ARD parameters.
        """
        return infer_ard(self.params, self.hyper_priors, in_place=in_place)

    def infer_obs_mean(
        self,
        Y: ObsStatic,
        in_place: bool = True,
    ) -> ObsMeanPosterior:
        """
        Infer observation mean parameters given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update ``self.params.obs_params.d`` in place and return it.
            If ``False``, return a new ``PosteriorObsMean``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorObsMean
            Posterior estimates of observation mean parameters.
        """
        return infer_obs_mean(Y, self.params, self.hyper_priors, in_place=in_place)

    def infer_obs_prec(
        self,
        Y: ObsStatic,
        in_place: bool = True,
    ) -> ObsPrecPosterior:
        """
        Infer observation precision parameters given current params and observed data.

        Parameters
        ----------
        Y
            Observed data.
        in_place
            If ``True``, update ``self.params.obs_params.phi`` in place and return it.
            If ``False``, return a new ``PosteriorObsPrec``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorObsPrec
            Posterior estimates of observation precision parameters.
        """
        return infer_obs_prec(Y, self.params, self.hyper_priors, in_place=in_place)

    def compute_lower_bound(
        self,
        Y: ObsStatic,
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
        return compute_lower_bound(Y, self.params, self.hyper_priors)
