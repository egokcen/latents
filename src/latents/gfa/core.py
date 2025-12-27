"""Core utilities to fit a group factor analysis (GFA) model to data."""

from __future__ import annotations

import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln, psi
from scipy.stats import gmean

from latents.gfa.config import GFAFitConfig
from latents.gfa.data_types import (
    GFAFitFlags,
    GFAFitTracker,
    GFAParams,
)
from latents.observation_model.observations import ObsStatic
from latents.observation_model.probabilistic import (
    HyperPriors,
    PosteriorARD,
    PosteriorLoading,
    PosteriorObsMean,
    PosteriorObsPrec,
)
from latents.state_model.latents import PosteriorLatentStatic

jsonpickle_numpy.register_handlers()


def fit(
    Y: ObsStatic,
    params: GFAParams,
    config: GFAFitConfig | None = None,
    hyper_priors: HyperPriors | None = None,
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
        Prior hyperparameters. If None, uses default HyperPriors().

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
        hyper_priors = HyperPriors()

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
    N = Y.data.shape[1]  # Number of samples
    x_dim = obs_params.x_dim  # Number of latent dimensions

    Y2 = np.sum(Y.data**2, axis=1)  # Sample second moments of observed data

    # Initialize the posterior covariance of C, if needed
    if obs_params.C.cov is None:
        obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))

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
            # Compute the runtime of this iteration
            end_time = time.time()
            tracker.iter_time = np.append(tracker.iter_time, end_time - start_time)
            # Record the current lower bound
            tracker.lb = np.append(tracker.lb, lb_curr)

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
    hyper_priors: HyperPriors | None = None,
) -> GFAParams:
    """Initialize GFA model parameters for fitting.

    Parameters
    ----------
    Y
        Observed data.
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    hyper_priors
        Prior hyperparameters. If None, uses default HyperPriors().

    Returns
    -------
    GFAParams
        Initialized GFA model parameters.
    """
    # Use defaults if not provided
    if config is None:
        config = GFAFitConfig()
    if hyper_priors is None:
        hyper_priors = HyperPriors()

    # Unpack config for local use
    x_dim_init = config.x_dim_init
    random_seed = config.random_seed
    save_c_cov = config.save_c_cov

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
    obs_params = params.obs_params
    state_params = params.state_params

    # Get the variance of each observed group
    Y_covs = [np.cov(Y_m) for Y_m in Ys]

    # Seed the random number generator for reproducible initialization.
    rng = np.random.default_rng(random_seed)

    # Latent variables
    state_params.X.mean = rng.normal(size=(x_dim, N))  # Mean
    state_params.X.cov = np.eye(x_dim)  # Covariance

    # Mean parameter
    obs_params.d.mean = np.mean(Y.data, axis=1)
    obs_params.d.cov = np.full(y_dim, 1 / hyper_priors.d_beta)

    # Noise precisions
    obs_params.phi.a = hyper_priors.a_phi + N / 2
    obs_params.phi.b = np.full(y_dim, hyper_priors.b_phi)
    obs_params.phi.mean = np.concatenate(
        [1 / np.diag(Y_cov) for Y_cov in Y_covs], axis=0
    )

    # Loading matrices

    # Mean
    obs_params.C.mean = np.zeros((y_dim, x_dim))
    # Get views of the loading matrices for each group
    C_means, _, _ = obs_params.C.get_groups(y_dims)
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
    obs_params.alpha.b = np.full((num_groups, x_dim), hyper_priors.b_alpha)
    # Scale ARD parameters to match the data
    obs_params.alpha.mean = np.zeros((num_groups, x_dim))
    for group_idx in range(num_groups):
        obs_params.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    return params


def infer_latents(
    Y: ObsStatic,
    params: GFAParams,
    in_place: bool = True,
) -> PosteriorLatentStatic | None:
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

    # Covariance
    X.cov[:] = np.linalg.inv(
        np.eye(state_params.x_dim)
        + np.sum(
            obs_params.phi.mean[:, np.newaxis, np.newaxis] * obs_params.C.moment,
            axis=0,
        )
    )
    # Ensure symmetry
    X.cov[:] = 0.5 * (X.cov + X.cov.T)
    # Mean
    X.mean[:] = (
        X.cov
        @ (obs_params.C.mean.T * obs_params.phi.mean[np.newaxis, :])
        @ (Y.data - obs_params.d.mean[:, np.newaxis])
    )
    # Second moment
    X.compute_moment(in_place=True)

    if not in_place:
        return X
    return None


def infer_loadings(
    Y: ObsStatic,
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
    obs_params = params.obs_params
    state_params = params.state_params

    y_dim = obs_params.y_dims.sum()  # Total number of observed dimensions
    x_dim = obs_params.x_dim  # Number of latent dimensions
    num_groups = len(obs_params.y_dims)  # Number of observed groups

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
        C = PosteriorLoading(
            mean=np.zeros((y_dim, x_dim)),
            cov=np.zeros((y_dim, x_dim, x_dim)),
            moment=np.zeros((y_dim, x_dim, x_dim)),
        )

    # Correlation matrix between latents and zero-centered observations
    if XY is None:
        XY = state_params.X.mean @ (Y.data - obs_params.d.mean[:, np.newaxis]).T

    # Get views of the loading matrices and precision parameters for each group
    _, C_covs, _ = C.get_groups(obs_params.y_dims)
    phi_means, _ = obs_params.phi.get_groups(obs_params.y_dims)

    for group_idx in range(num_groups):
        # Covariance
        C_covs[group_idx][:] = np.linalg.inv(
            np.diag(obs_params.alpha.mean[group_idx, :])
            + phi_means[group_idx][:, np.newaxis, np.newaxis] * state_params.X.moment
        )

    # Mean
    C.mean[:] = obs_params.phi.mean[:, np.newaxis] * np.einsum(
        "ijk,ij->ik", C.cov, XY.T
    )

    # Second moment
    C.compute_moment(in_place=True)

    if not in_place:
        return C
    return None


def infer_ard(
    params: GFAParams,
    hyper_priors: HyperPriors,
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
    obs_params = params.obs_params
    num_groups = len(obs_params.y_dims)  # Number of observed groups

    # Initialize alpha, if needed
    if in_place:
        if obs_params.alpha.a is None:
            obs_params.alpha.a = hyper_priors.a_alpha + obs_params.y_dims / 2
        if obs_params.alpha.b is None:
            obs_params.alpha.b = np.zeros((num_groups, obs_params.x_dim))
        if obs_params.alpha.mean is None:
            obs_params.alpha.mean = np.zeros((num_groups, obs_params.x_dim))
        alpha = obs_params.alpha
    else:
        alpha = PosteriorARD(
            a=hyper_priors.a_alpha + obs_params.y_dims / 2,
            b=np.zeros((num_groups, obs_params.x_dim)),
            mean=np.zeros((num_groups, obs_params.x_dim)),
        )

    # Expected squared norm of each column of C
    if C_norm is None:
        C_norm = obs_params.C.compute_squared_norms(obs_params.y_dims)

    # Rate parameters
    alpha.b[:] = hyper_priors.b_alpha + 0.5 * C_norm

    # Mean
    alpha.compute_mean(in_place=True)

    if not in_place:
        return alpha
    return None


def infer_obs_mean(
    Y: ObsStatic,
    params: GFAParams,
    hyper_priors: HyperPriors,
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
    obs_params = params.obs_params
    state_params = params.state_params
    y_dim, N = Y.data.shape

    # Initialize d, if needed
    if in_place:
        if obs_params.d.mean is None:
            obs_params.d.mean = np.zeros(y_dim)
        if obs_params.d.cov is None:
            obs_params.d.cov = np.zeros(y_dim)
        d = obs_params.d
    else:
        d = PosteriorObsMean(mean=np.zeros(y_dim), cov=np.zeros(y_dim))

    # Covariance
    d.cov[:] = 1 / (hyper_priors.d_beta + N * obs_params.phi.mean)
    d.mean[:] = (
        d.cov
        * obs_params.phi.mean
        * np.sum(Y.data - obs_params.C.mean @ state_params.X.mean, axis=1)
    )

    if not in_place:
        return d
    return None


def infer_obs_prec(
    Y: ObsStatic,
    params: GFAParams,
    hyper_priors: HyperPriors,
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
    obs_params = params.obs_params
    state_params = params.state_params
    y_dim, N = Y.data.shape

    # Initialize phi, if needed
    if in_place:
        if obs_params.phi.mean is None:
            obs_params.phi.mean = np.zeros(y_dim)
        if obs_params.phi.a is None:
            obs_params.phi.a = hyper_priors.a_phi + N / 2
        if obs_params.phi.b is None:
            obs_params.phi.b = np.zeros(y_dim)
        phi = obs_params.phi
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
        d_moment = obs_params.d.cov + obs_params.d.mean**2  # Only the diagonal is used

    # Correlation matrix between latents and zero-centered observations
    if XY is None:
        XY = state_params.X.mean @ (Y.data - obs_params.d.mean[:, np.newaxis]).T

    # Rate parameter
    phi.b[:] = hyper_priors.b_phi + 0.5 * (
        N * d_moment
        + Y2
        - 2 * np.sum(obs_params.d.mean[:, np.newaxis] * Y.data, axis=1)
        - 2 * np.sum(obs_params.C.mean * XY.T, axis=1)
        + np.sum(obs_params.C.moment * state_params.X.moment, axis=(1, 2))
    )

    # Mean
    phi.compute_mean(in_place=True)

    if not in_place:
        return phi
    return None


def compute_lower_bound(
    Y: ObsStatic,
    params: GFAParams,
    hyper_priors: HyperPriors,
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
    obs_params = params.obs_params
    state_params = params.state_params
    y_dims = obs_params.y_dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    x_dim = obs_params.x_dim  # Number of latent dimensions
    num_groups = len(obs_params.y_dims)  # Number of observed groups
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
        logdet_C = np.sum(np.linalg.slogdet(obs_params.C.cov)[1])
    if C_norm is None:
        C_norm = obs_params.C.compute_squared_norms(y_dims)
    if d_moment is None:
        d_moment = obs_params.d.cov + obs_params.d.mean**2

    # Likelihood term
    log_phi = digamma_a_phi - np.log(obs_params.phi.b)
    lb = (
        const_lik
        + 0.5 * N * np.sum(log_phi)
        - np.sum(obs_params.phi.mean * (obs_params.phi.b - hyper_priors.b_phi))
    )

    # X KL term
    lb += 0.5 * N * (x_dim + np.linalg.slogdet(state_params.X.cov)[1]) - 0.5 * np.trace(
        state_params.X.moment
    )

    log_alpha = digamma_a_alpha[:, np.newaxis] - np.log(obs_params.alpha.b)

    # C KL term
    lb += 0.5 * (
        x_dim * y_dim
        + logdet_C
        + np.sum(y_dims[:, np.newaxis] * log_alpha - obs_params.alpha.mean * C_norm)
    )

    # alpha KL term
    lb += (
        num_groups * x_dim * (alogb_alpha - loggamma_a_alpha_prior)
        + np.sum(
            -obs_params.alpha.a[:, np.newaxis] * np.log(obs_params.alpha.b)
            - hyper_priors.b_alpha * obs_params.alpha.mean
            + (hyper_priors.a_alpha - obs_params.alpha.a)[:, np.newaxis] * log_alpha
        )
        + np.sum(x_dim * (loggamma_a_alpha_post + obs_params.alpha.a))
    )

    # phi KL term
    lb += y_dim * (
        alogb_phi + loggamma_a_phi_post - loggamma_a_phi_prior + obs_params.phi.a
    ) + np.sum(
        -obs_params.phi.a * np.log(obs_params.phi.b)
        + hyper_priors.b_phi * obs_params.phi.mean
        + (hyper_priors.a_phi - obs_params.phi.a) * log_phi
    )

    # d KL term
    lb += const_d + 0.5 * (
        np.sum(np.log(obs_params.d.cov)) - hyper_priors.d_beta * np.sum(d_moment)
    )

    return lb


def compute_lower_bound_constants(
    N: int,
    params: GFAParams,
    hyper_priors: HyperPriors,
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
    obs_params = params.obs_params
    y_dim = obs_params.y_dims.sum()  # Total number of observed dimensions

    # Constant factors in the lower bound
    # Related to the likelihood
    const_lik = -(y_dim * N / 2) * np.log(2 * np.pi)
    # Related to observation mean parameters
    const_d = 0.5 * y_dim + 0.5 * y_dim * np.log(hyper_priors.d_beta)
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
        Prior hyperparameters. If None, uses default HyperPriors().

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
        hyper_priors: HyperPriors | None = None,
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
        self.hyper_priors = hyper_priors if hyper_priors is not None else HyperPriors()

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
    ) -> PosteriorLatentStatic | None:
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
        Y: ObsStatic,
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
        return infer_ard(self.params, self.hyper_priors, in_place=in_place)

    def infer_obs_mean(
        self,
        Y: ObsStatic,
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
        return infer_obs_mean(Y, self.params, self.hyper_priors, in_place=in_place)

    def infer_obs_prec(
        self,
        Y: ObsStatic,
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
