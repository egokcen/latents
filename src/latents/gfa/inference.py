"""Inference functions for Group Factor Analysis."""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln, psi
from scipy.stats import gmean
from tqdm.auto import tqdm

from latents._core.fitting import FitFlags, FitTracker
from latents._core.numerics import stability_floor, validate_tolerance
from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.observation import (
    ARDPosterior,
    LoadingPosterior,
    ObsMeanPosterior,
    ObsParamsHyperPrior,
    ObsParamsPosterior,
    ObsPrecPosterior,
)
from latents.state import LatentsPosteriorStatic

# -----------------------------------------------------------------------------
# Tracker and Flags classes
# -----------------------------------------------------------------------------


class GFAFitTracker(FitTracker):
    """Quantities tracked during a GFA model fit.

    Attributes
    ----------
    lb : ndarray of float, shape (num_iter,)
        Variational lower bound at each iteration.
    iter_time : ndarray of float, shape (num_iter,)
        Runtime on each iteration.
    """

    pass


class GFAFitFlags(FitFlags):
    """Status messages during a GFA model fit.

    Attributes
    ----------
    converged : bool
        True if the lower bound converged before reaching max_iter.
    decreasing_lb : bool
        True if lower bound decreased during fitting.
    private_var_floor : bool
        True if the private variance floor was used on any values of phi.
    x_dims_removed : int
        Number of latent dimensions removed due to low variance.
    """

    def __init__(
        self,
        converged: bool = False,
        decreasing_lb: bool = False,
        private_var_floor: bool = False,
        x_dims_removed: int = 0,
    ):
        super().__init__(converged, decreasing_lb, private_var_floor)
        self.x_dims_removed = x_dims_removed

    def __repr__(self) -> str:
        return (
            f"GFAFitFlags("
            f"converged={self.converged}, "
            f"decreasing_lb={self.decreasing_lb}, "
            f"private_var_floor={self.private_var_floor}, "
            f"x_dims_removed={self.x_dims_removed})"
        )

    def display(self) -> None:
        """Print out the fit flags."""
        super().display()
        print(f"Latent dimensions removed: {self.x_dims_removed}")


# -----------------------------------------------------------------------------
# Main fit function
# -----------------------------------------------------------------------------


def fit(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    config: GFAFitConfig | None = None,
    obs_hyperprior: ObsParamsHyperPrior | None = None,
    tracker: GFAFitTracker | None = None,
    flags: GFAFitFlags | None = None,
    max_iter: int | None = None,
) -> tuple[ObsParamsPosterior, LatentsPosteriorStatic, GFAFitTracker, GFAFitFlags]:
    """Fit a GFA model to data via variational inference.

    Parameters
    ----------
    Y
        Observed data.
    obs_posterior
        Observation model posterior (modified in place).
    latents_posterior
        Latent posterior (modified in place).
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    obs_hyperprior
        Prior hyperparameters. If None, uses default ObsParamsHyperPrior().
    tracker
        If provided, append to existing tracker (resume). If None, create fresh.
    flags
        If provided, preserve existing flags (resume). If None, create fresh.
    max_iter
        Override config.max_iter. Useful for resume with different budget.

    Returns
    -------
    obs_posterior : ObsParamsPosterior
        Fitted observation model posterior.
    latents_posterior : LatentsPosteriorStatic
        Fitted latent posterior.
    tracker : GFAFitTracker
        Quantities tracked during fitting.
    flags : GFAFitFlags
        Status messages about the fitting process.

    Raises
    ------
    ValueError
        If ``obs_posterior.y_dims`` does not match ``Y.dims``, if
        ``obs_posterior.x_dim`` does not match ``config.x_dim_init``, or if
        ``tracker`` and ``flags`` are not both provided or both None.
    """
    if config is None:
        config = GFAFitConfig()
    if obs_hyperprior is None:
        obs_hyperprior = ObsParamsHyperPrior()

    # Validate tracker/flags consistency for resume
    if (tracker is None) != (flags is None):
        msg = (
            "tracker and flags must both be provided (resume) or both be None (fresh)."
        )
        raise ValueError(msg)
    resuming = tracker is not None

    # Unpack config for local use, with max_iter override for resume
    x_dim_init = config.x_dim_init
    fit_tol = config.fit_tol
    if max_iter is None:
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

    # Check that observed data dimensions match
    if not np.array_equal(obs_posterior.y_dims, Y.dims):
        msg = "obs_posterior.y_dims must match Y.dims."
        raise ValueError(msg)

    # Check that initial latent dimensionality matches
    if obs_posterior.x_dim != x_dim_init:
        msg = "obs_posterior.x_dim must match config.x_dim_init."
        raise ValueError(msg)

    # Data size characteristics
    y_dims = Y.dims
    y_dim = y_dims.sum()
    n_samples = Y.data.shape[1]
    x_dim = obs_posterior.x_dim

    # Sample second moments of observed data
    Y2 = np.sum(Y.data**2, axis=1)

    # Initialize the posterior covariance of C if needed
    if obs_posterior.C.cov is None:
        obs_posterior.C.cov = np.zeros((y_dim, x_dim, x_dim))

    # Compute the variance floor for each observed dimension
    floor = stability_floor(Y.data.dtype)
    var_floor = np.maximum(min_var_frac * np.var(Y.data, axis=1, ddof=1), floor)

    # Constant factors in the lower bound
    consts_lb = compute_lower_bound_constants(n_samples, obs_posterior, obs_hyperprior)

    # Create new tracker/flags if not resuming
    if not resuming:
        tracker = GFAFitTracker()
        flags = GFAFitFlags()

    # Determine starting point based on existing history
    iter_offset = len(tracker.lb) if tracker.lb is not None else 0
    lb_curr = tracker.lb[iter_offset - 1] if iter_offset > 0 else -np.inf

    # Allocate or extend tracking arrays
    if save_fit_progress:
        if iter_offset > 0:
            tracker.lb = np.concatenate([tracker.lb, np.empty(max_iter)])
            tracker.iter_time = np.concatenate([tracker.iter_time, np.empty(max_iter)])
        else:
            tracker.lb = np.empty(max_iter)
            tracker.iter_time = np.empty(max_iter)

    # Progress bar (disabled when not verbose)
    pbar = tqdm(range(max_iter), desc="Fitting", disable=not verbose)

    fit_iter = 0
    for fit_iter in pbar:
        # Check if any latents need to be removed
        if prune_x:
            kept_x_dims = np.nonzero(
                np.mean(latents_posterior.mean**2, axis=1) > prune_tol
            )[0]
            if len(kept_x_dims) < x_dim:
                # Remove inactive latents
                obs_posterior.get_subset_dims(kept_x_dims, in_place=True)
                latents_posterior.get_subset_dims(kept_x_dims, in_place=True)
                flags.x_dims_removed += x_dim - obs_posterior.x_dim
                x_dim = obs_posterior.x_dim
                if x_dim <= 0:
                    break

        # Start timer for current iteration
        if save_fit_progress:
            start_time = time.time()

        # Observation mean parameter, d
        infer_obs_mean(Y, obs_posterior, latents_posterior, obs_hyperprior)
        # Second moments for phi updates and lower bound
        d_moment = obs_posterior.d.cov + obs_posterior.d.mean**2

        # Latent variables, X
        infer_latents(Y, obs_posterior, latents_posterior)
        # Correlation matrix between latents and zero-centered observations
        # X.mean: (x_dim, n_samples), d.mean: (y_dim,) -> XY: (x_dim, y_dim)
        XY = latents_posterior.mean @ (Y.data - obs_posterior.d.mean[:, np.newaxis]).T

        # Loading matrices, C
        infer_loadings(Y, obs_posterior, latents_posterior, XY=XY)
        # Log-determinant for lower bound
        logdet_C = np.sum(np.linalg.slogdet(obs_posterior.C.cov)[1])
        # Expected squared norm of each column of C
        C_norm = obs_posterior.C.compute_squared_norms(y_dims)

        # ARD parameters, alpha
        infer_ard(obs_posterior, obs_hyperprior, C_norm=C_norm)

        # Observation precision parameters, phi
        infer_obs_prec(
            Y,
            obs_posterior,
            latents_posterior,
            obs_hyperprior,
            d_moment=d_moment,
            XY=XY,
            Y2=Y2,
        )
        # Set minimum private variance
        obs_posterior.phi.mean[:] = np.minimum(1 / var_floor, obs_posterior.phi.mean)
        obs_posterior.phi.b[:] = obs_posterior.phi.a / obs_posterior.phi.mean

        # Compute the lower bound
        lb_old = lb_curr
        lb_curr = compute_lower_bound(
            Y,
            obs_posterior,
            latents_posterior,
            obs_hyperprior,
            consts=consts_lb,
            logdet_C=logdet_C,
            C_norm=C_norm,
            d_moment=d_moment,
        )

        # Save progress
        if save_fit_progress:
            end_time = time.time()
            tracker.iter_time[iter_offset + fit_iter] = end_time - start_time
            tracker.lb[iter_offset + fit_iter] = lb_curr

        # Update progress bar postfix
        postfix = {"lb": f"{lb_curr:.2e}"}
        # Relative change: quantity compared to fit_tol for convergence
        # Only compute after burn-in and when denominator is non-zero
        denom = lb_old - tracker.lb_base if tracker.lb_base is not None else 0.0
        if fit_iter > 1 and denom != 0.0:
            rel_change = (lb_curr - lb_old) / denom
            postfix["Δ"] = f"{rel_change:.1e}"
        if prune_x:
            postfix["x_dim"] = x_dim
        pbar.set_postfix(postfix)

        # Check stopping conditions
        # Set lb_base during burn-in period (fresh fit only)
        if not resuming and fit_iter <= 1:
            tracker.lb_base = lb_curr
        elif lb_curr < lb_old:
            flags.decreasing_lb = True
        elif (lb_curr - tracker.lb_base) < (1 + fit_tol) * (lb_old - tracker.lb_base):
            flags.converged = True
            break

    # Close progress bar before final messages
    pbar.close()

    # Truncate pre-allocated arrays to actual iteration count
    if save_fit_progress:
        total_iters = iter_offset + fit_iter + 1
        tracker.lb = tracker.lb[:total_iters]
        tracker.iter_time = tracker.iter_time[:total_iters]

    # Display reasons for stopping
    if verbose:
        if flags.converged:
            print(f"Lower bound converged after {fit_iter + 1} iterations.")
        elif ((fit_iter + 1) < max_iter) and obs_posterior.x_dim <= 0:
            print("Fitting stopped because no significant latent dimensions remain.")
        else:
            print(f"Fitting stopped after max_iter ({max_iter}) was reached.")

    # Check if the variance floor was reached
    if np.any(obs_posterior.phi.mean == 1 / var_floor):
        flags.private_var_floor = True

    if not save_c_cov:
        obs_posterior.C.cov = None

    if not save_x:
        latents_posterior.clear()

    return obs_posterior, latents_posterior, tracker, flags


def init_posteriors(
    Y: ObsStatic,
    config: GFAFitConfig | None = None,
    obs_hyperprior: ObsParamsHyperPrior | None = None,
) -> tuple[ObsParamsPosterior, LatentsPosteriorStatic]:
    """Initialize GFA model posteriors for fitting.

    Parameters
    ----------
    Y
        Observed data.
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    obs_hyperprior
        Prior hyperparameters. If None, uses default ObsParamsHyperPrior().

    Returns
    -------
    obs_posterior : ObsParamsPosterior
        Initialized observation model posterior.
    latents_posterior : LatentsPosteriorStatic
        Initialized latent posterior.
    """
    if config is None:
        config = GFAFitConfig()
    if obs_hyperprior is None:
        obs_hyperprior = ObsParamsHyperPrior()

    x_dim_init = config.x_dim_init
    random_seed = config.random_seed
    save_c_cov = config.save_c_cov

    # Data size characteristics
    y_dims = Y.dims
    y_dim = y_dims.sum()
    n_groups = len(y_dims)
    n_samples = Y.data.shape[1]
    x_dim = x_dim_init

    Ys = Y.get_groups()

    # Initialize posteriors
    obs_posterior = ObsParamsPosterior(x_dim=x_dim, y_dims=y_dims)
    latents_posterior = LatentsPosteriorStatic()

    # Covariance of each observed group
    Y_covs = [np.cov(Y_m) for Y_m in Ys]

    rng = np.random.default_rng(random_seed)

    # Latent variables
    latents_posterior.mean = rng.normal(size=(x_dim, n_samples))
    latents_posterior.cov = np.eye(x_dim)

    # Mean parameter d
    obs_posterior.d.mean = np.mean(Y.data, axis=1)
    obs_posterior.d.cov = np.full(y_dim, 1 / obs_hyperprior.beta_d)

    # Noise precisions phi
    obs_posterior.phi.a = obs_hyperprior.a_phi + n_samples / 2
    obs_posterior.phi.b = np.full(y_dim, obs_hyperprior.b_phi)
    obs_posterior.phi.mean = np.concatenate(
        [1 / np.diag(Y_cov) for Y_cov in Y_covs], axis=0
    )

    # Loading matrices C - mean
    obs_posterior.C.mean = np.zeros((y_dim, x_dim))
    C_means, _, _ = obs_posterior.C.get_groups(y_dims)
    for group_idx in range(n_groups):
        eigs = eigh(Y_covs[group_idx], eigvals_only=True)
        scale = gmean(eigs[eigs > 0])
        C_means[group_idx][:] = rng.normal(
            scale=np.sqrt(scale / x_dim), size=(y_dims[group_idx], x_dim)
        )

    # Loading matrices C - covariance and moments
    obs_posterior.C.cov = np.zeros((y_dim, x_dim, x_dim))
    obs_posterior.C.compute_moment()
    _, _, C_moments = obs_posterior.C.get_groups(y_dims)
    if not save_c_cov:
        obs_posterior.C.cov = None

    # ARD parameters alpha
    obs_posterior.alpha.a = obs_hyperprior.a_alpha + y_dims / 2
    obs_posterior.alpha.b = np.full((n_groups, x_dim), obs_hyperprior.b_alpha)
    obs_posterior.alpha.mean = np.zeros((n_groups, x_dim))
    for group_idx in range(n_groups):
        obs_posterior.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    return obs_posterior, latents_posterior


# -----------------------------------------------------------------------------
# Inference functions for individual parameters
# -----------------------------------------------------------------------------


def infer_latents(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic | None = None,
) -> LatentsPosteriorStatic:
    """Infer latent posterior q(X) given observations and fitted parameters.

    Parameters
    ----------
    Y
        Observed data.
    obs_posterior
        Posterior over observation parameters. Reads C, phi, d.
    latents_posterior
        If provided, update in-place and return.
        If None, create and return a new LatentsPosteriorStatic.

    Returns
    -------
    LatentsPosteriorStatic
        Posterior over latent variables.
    """
    x_dim = obs_posterior.x_dim

    if latents_posterior is None:
        latents_posterior = LatentsPosteriorStatic(
            mean=np.zeros((x_dim, Y.data.shape[1])),
            cov=np.zeros((x_dim, x_dim)),
            moment=np.zeros((x_dim, x_dim)),
        )
    else:
        # Initialize arrays if needed for in-place update
        if latents_posterior.mean is None:
            latents_posterior.mean = np.zeros((x_dim, Y.data.shape[1]))
        if latents_posterior.cov is None:
            latents_posterior.cov = np.zeros((x_dim, x_dim))
        if latents_posterior.moment is None:
            latents_posterior.moment = np.zeros((x_dim, x_dim))

    # Covariance: inv(I + sum_j phi_j * E[C_j^T C_j])
    # phi.mean: (y_dim,) -> (y_dim, 1, 1), C.moment: (y_dim, x_dim, x_dim)
    # Weighted sum over y_dim -> (x_dim, x_dim)
    latents_posterior.cov[:] = np.linalg.inv(
        np.eye(x_dim)
        + np.sum(
            obs_posterior.phi.mean[:, np.newaxis, np.newaxis] * obs_posterior.C.moment,
            axis=0,
        )
    )
    # Ensure symmetry
    latents_posterior.cov[:] = 0.5 * (latents_posterior.cov + latents_posterior.cov.T)

    # Mean: cov @ C^T diag(phi) @ (Y - d) -> mean: (x_dim, n_samples)
    # phi: (y_dim,) -> (1, y_dim) for broadcast with C.mean.T: (x_dim, y_dim)
    # d: (y_dim,) -> (y_dim, 1) for broadcast with Y: (y_dim, n_samples)
    latents_posterior.mean[:] = (
        latents_posterior.cov
        @ (obs_posterior.C.mean.T * obs_posterior.phi.mean[np.newaxis, :])
        @ (Y.data - obs_posterior.d.mean[:, np.newaxis])
    )

    # Second moment
    latents_posterior.compute_moment(in_place=True)

    return latents_posterior


def infer_loadings(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    XY: np.ndarray | None = None,
) -> LoadingPosterior:
    """Infer loading posterior q(C). Updates obs_posterior.C in-place.

    Parameters
    ----------
    Y
        Observed data.
    obs_posterior
        Posterior over observation parameters. Reads alpha, phi, d; writes C.
    latents_posterior
        Posterior over latents. Reads mean, moment.
    XY
        Pre-computed correlation matrix (x_dim, y_dim). Computed if not provided.

    Returns
    -------
    LoadingPosterior
        Reference to obs_posterior.C (updated in-place).
    """
    y_dim = obs_posterior.y_dims.sum()
    x_dim = obs_posterior.x_dim
    n_groups = len(obs_posterior.y_dims)

    C = obs_posterior.C

    # Initialize C arrays if needed
    if C.mean is None:
        C.mean = np.zeros((y_dim, x_dim))
    if C.cov is None:
        C.cov = np.zeros((y_dim, x_dim, x_dim))
    if C.moment is None:
        C.moment = np.zeros((y_dim, x_dim, x_dim))

    # Correlation matrix between latents and zero-centered observations
    # X.mean: (x_dim, n_samples), (Y - d): (y_dim, n_samples) -> XY: (x_dim, y_dim)
    if XY is None:
        XY = latents_posterior.mean @ (Y.data - obs_posterior.d.mean[:, np.newaxis]).T

    # Get views of the loading matrices and precision parameters for each group
    _, C_covs, _ = C.get_groups(obs_posterior.y_dims)
    phi_means, _ = obs_posterior.phi.get_groups(obs_posterior.y_dims)

    for group_idx in range(n_groups):
        # Covariance: inv(diag(alpha) + phi * E[X X^T]) -> (y_dim_m, x_dim, x_dim)
        # phi: (y_dim_m,) -> (y_dim_m, 1, 1) for broadcast with X.moment: (x_dim, x_dim)
        C_covs[group_idx][:] = np.linalg.inv(
            np.diag(obs_posterior.alpha.mean[group_idx, :])
            + phi_means[group_idx][:, np.newaxis, np.newaxis] * latents_posterior.moment
        )

    # Mean: phi * einsum(C.cov, XY) -> C.mean: (y_dim, x_dim)
    # phi: (y_dim,) -> (y_dim, 1) for broadcast
    # einsum "ijk,ij->ik": contract over k dimension
    C.mean[:] = obs_posterior.phi.mean[:, np.newaxis] * np.einsum(
        "ijk,ij->ik", C.cov, XY.T
    )

    # Second moment
    C.compute_moment(in_place=True)

    return C


def infer_ard(
    obs_posterior: ObsParamsPosterior,
    hyperprior: ObsParamsHyperPrior,
    C_norm: np.ndarray | None = None,
) -> ARDPosterior:
    """Infer ARD posterior q(alpha). Updates obs_posterior.alpha in-place.

    Parameters
    ----------
    obs_posterior
        Posterior over observation parameters. Reads C; writes alpha.
    hyperprior
        Hyperprior parameters (a_alpha, b_alpha).
    C_norm
        Pre-computed squared column norms of C per group, shape (n_groups, x_dim).

    Returns
    -------
    ARDPosterior
        Reference to obs_posterior.alpha (updated in-place).
    """
    n_groups = len(obs_posterior.y_dims)
    alpha = obs_posterior.alpha

    # Initialize alpha arrays if needed
    if alpha.a is None:
        alpha.a = hyperprior.a_alpha + obs_posterior.y_dims / 2
    if alpha.b is None:
        alpha.b = np.zeros((n_groups, obs_posterior.x_dim))
    if alpha.mean is None:
        alpha.mean = np.zeros((n_groups, obs_posterior.x_dim))

    if C_norm is None:
        C_norm = obs_posterior.C.compute_squared_norms(obs_posterior.y_dims)

    # Rate parameters
    alpha.b[:] = hyperprior.b_alpha + 0.5 * C_norm

    # Mean
    alpha.compute_mean(in_place=True)

    return alpha


def infer_obs_mean(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    hyperprior: ObsParamsHyperPrior,
) -> ObsMeanPosterior:
    """Infer observation mean posterior q(d). Updates obs_posterior.d in-place.

    Parameters
    ----------
    Y
        Observed data.
    obs_posterior
        Posterior over observation parameters. Reads C, phi; writes d.
    latents_posterior
        Posterior over latents. Reads mean.
    hyperprior
        Hyperprior parameters (beta_d).

    Returns
    -------
    ObsMeanPosterior
        Reference to obs_posterior.d (updated in-place).
    """
    y_dim, n_samples = Y.data.shape
    d = obs_posterior.d

    # Initialize d arrays if needed
    if d.mean is None:
        d.mean = np.zeros(y_dim)
    if d.cov is None:
        d.cov = np.zeros(y_dim)

    # Covariance (diagonal)
    d.cov[:] = 1 / (hyperprior.beta_d + n_samples * obs_posterior.phi.mean)

    # Mean
    d.mean[:] = (
        d.cov
        * obs_posterior.phi.mean
        * np.sum(Y.data - obs_posterior.C.mean @ latents_posterior.mean, axis=1)
    )

    return d


def infer_obs_prec(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    hyperprior: ObsParamsHyperPrior,
    d_moment: np.ndarray | None = None,
    XY: np.ndarray | None = None,
    Y2: np.ndarray | None = None,
) -> ObsPrecPosterior:
    """Infer precision posterior q(φ). Updates obs_posterior.phi in-place.

    Parameters
    ----------
    Y
        Observed data.
    obs_posterior
        Posterior over observation parameters. Reads C, d; writes phi.
    latents_posterior
        Posterior over latents. Reads mean, moment.
    hyperprior
        Hyperprior parameters (a_phi, b_phi).
    d_moment
        Pre-computed second moment of d, shape (y_dim,).
    XY
        Pre-computed correlation matrix (x_dim, y_dim).
    Y2
        Pre-computed sample second moments, shape (y_dim,).

    Returns
    -------
    ObsPrecPosterior
        Reference to obs_posterior.phi (updated in-place).
    """
    y_dim, n_samples = Y.data.shape
    phi = obs_posterior.phi

    # Initialize phi arrays if needed
    if phi.mean is None:
        phi.mean = np.zeros(y_dim)
    if phi.a is None:
        phi.a = hyperprior.a_phi + n_samples / 2
    if phi.b is None:
        phi.b = np.zeros(y_dim)

    # Pre-computations
    if Y2 is None:
        Y2 = np.sum(Y.data**2, axis=1)
    if d_moment is None:
        d_moment = obs_posterior.d.cov + obs_posterior.d.mean**2
    if XY is None:
        XY = latents_posterior.mean @ (Y.data - obs_posterior.d.mean[:, np.newaxis]).T

    # Rate parameter: expected reconstruction error -> phi.b: (y_dim,)
    phi.b[:] = hyperprior.b_phi + 0.5 * (
        n_samples * d_moment
        + Y2
        - 2 * np.sum(obs_posterior.d.mean[:, np.newaxis] * Y.data, axis=1)
        - 2 * np.sum(obs_posterior.C.mean * XY.T, axis=1)
        + np.sum(obs_posterior.C.moment * latents_posterior.moment, axis=(1, 2))
    )

    # Mean
    phi.compute_mean(in_place=True)

    return phi


# -----------------------------------------------------------------------------
# Lower bound computation
# -----------------------------------------------------------------------------


def compute_lower_bound(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    obs_hyperprior: ObsParamsHyperPrior,
    consts: tuple | None = None,
    logdet_C: float | None = None,
    C_norm: np.ndarray | None = None,
    d_moment: np.ndarray | None = None,
) -> float:
    """Compute the variational lower bound (ELBO) for a GFA model.

    Parameters
    ----------
    Y
        Observed data.
    obs_posterior
        Observation model posterior.
    latents_posterior
        Latent posterior.
    obs_hyperprior
        Hyperparameters of the prior distributions.
    consts
        Constant factors in the lower bound. If None, computed.
    logdet_C
        Log-determinant of loading covariances. If None, computed.
    C_norm
        Expected squared norms of loading columns. If None, computed.
    d_moment
        Second moment of observation mean. If None, computed.

    Returns
    -------
    float
        Variational lower bound.
    """
    y_dims = obs_posterior.y_dims
    y_dim = y_dims.sum()
    x_dim = obs_posterior.x_dim
    n_groups = len(y_dims)
    n_samples = Y.data.shape[1]

    if consts is None:
        consts = compute_lower_bound_constants(n_samples, obs_posterior, obs_hyperprior)

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

    if logdet_C is None:
        logdet_C = np.sum(np.linalg.slogdet(obs_posterior.C.cov)[1])
    if C_norm is None:
        C_norm = obs_posterior.C.compute_squared_norms(y_dims)
    if d_moment is None:
        d_moment = obs_posterior.d.cov + obs_posterior.d.mean**2

    floor = stability_floor(obs_posterior.phi.b.dtype)

    # Likelihood term
    log_phi = digamma_a_phi - np.log(np.maximum(obs_posterior.phi.b, floor))
    lb = (
        const_lik
        + 0.5 * n_samples * np.sum(log_phi)
        - np.sum(obs_posterior.phi.mean * (obs_posterior.phi.b - obs_hyperprior.b_phi))
    )

    # X KL term
    lb += 0.5 * n_samples * (
        x_dim + np.linalg.slogdet(latents_posterior.cov)[1]
    ) - 0.5 * np.trace(latents_posterior.moment)

    # digamma_a_alpha: (n_groups,) -> (n_groups, 1)
    log_alpha = digamma_a_alpha[:, np.newaxis] - np.log(
        np.maximum(obs_posterior.alpha.b, floor)
    )

    # C KL term
    lb += 0.5 * (
        x_dim * y_dim
        + logdet_C
        + np.sum(y_dims[:, np.newaxis] * log_alpha - obs_posterior.alpha.mean * C_norm)
    )

    # alpha KL term
    lb += (
        n_groups * x_dim * (alogb_alpha - loggamma_a_alpha_prior)
        + np.sum(
            -obs_posterior.alpha.a[:, np.newaxis]
            * np.log(np.maximum(obs_posterior.alpha.b, floor))
            - obs_hyperprior.b_alpha * obs_posterior.alpha.mean
            + (obs_hyperprior.a_alpha - obs_posterior.alpha.a)[:, np.newaxis]
            * log_alpha
        )
        + np.sum(x_dim * (loggamma_a_alpha_post + obs_posterior.alpha.a))
    )

    # phi KL term
    lb += y_dim * (
        alogb_phi + loggamma_a_phi_post - loggamma_a_phi_prior + obs_posterior.phi.a
    ) + np.sum(
        -obs_posterior.phi.a * np.log(np.maximum(obs_posterior.phi.b, floor))
        + obs_hyperprior.b_phi * obs_posterior.phi.mean
        + (obs_hyperprior.a_phi - obs_posterior.phi.a) * log_phi
    )

    # d KL term
    lb += const_d + 0.5 * (
        np.sum(np.log(obs_posterior.d.cov)) - obs_hyperprior.beta_d * np.sum(d_moment)
    )

    return lb


def compute_lower_bound_constants(
    n_samples: int,
    obs_posterior: ObsParamsPosterior,
    obs_hyperprior: ObsParamsHyperPrior,
) -> tuple[
    float, float, float, float, float, float, float, float, np.ndarray, np.ndarray
]:
    """Compute constant factors in the variational lower bound.

    Parameters
    ----------
    n_samples
        Number of samples in the observed data.
    obs_posterior
        Observation model posterior.
    obs_hyperprior
        Hyperparameters of the prior distributions.

    Returns
    -------
    tuple
        Constant factors for the lower bound computation.
    """
    y_dim = obs_posterior.y_dims.sum()

    # Related to the likelihood
    const_lik = -(y_dim * n_samples / 2) * np.log(2 * np.pi)
    # Related to observation mean parameters
    const_d = 0.5 * y_dim + 0.5 * y_dim * np.log(obs_hyperprior.beta_d)
    # Related to observation precision parameters
    alogb_phi = obs_hyperprior.a_phi * np.log(obs_hyperprior.b_phi)
    loggamma_a_phi_prior = gammaln(obs_hyperprior.a_phi)
    loggamma_a_phi_post = gammaln(obs_posterior.phi.a)
    digamma_a_phi = psi(obs_posterior.phi.a)
    # Related to ARD parameters
    alogb_alpha = obs_hyperprior.a_alpha * np.log(obs_hyperprior.b_alpha)
    loggamma_a_alpha_prior = gammaln(obs_hyperprior.a_alpha)
    loggamma_a_alpha_post = gammaln(obs_posterior.alpha.a)
    digamma_a_alpha = psi(obs_posterior.alpha.a)

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
