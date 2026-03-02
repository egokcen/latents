"""Inference functions for Group Factor Analysis."""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln, psi
from scipy.stats import gmean

from latents._internal.numerics import stability_floor, validate_tolerance
from latents.callbacks import invoke_callbacks
from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.gfa.tracking import GFAFitContext, GFAFitFlags, GFAFitTracker
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
    callbacks: list | None = None,
) -> tuple[ObsParamsPosterior, LatentsPosteriorStatic, GFAFitTracker, GFAFitFlags]:
    """Fit a GFA model to data via variational inference.

    Parameters
    ----------
    Y : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Observation model posterior (modified in place).
    latents_posterior : LatentsPosteriorStatic
        Latent posterior (modified in place).
    config : GFAFitConfig or None, default None
        Fitting configuration. If None, uses default `GFAFitConfig()`.
    obs_hyperprior : ObsParamsHyperPrior or None, default None
        Prior hyperparameters. If None, uses default `ObsParamsHyperPrior()`.
    tracker : GFAFitTracker or None, default None
        If provided, append to existing tracker (resume). If None, create fresh.
    flags : GFAFitFlags or None, default None
        If provided, preserve existing flags (resume). If None, create fresh.
    max_iter : int or None, default None
        Override `config.max_iter`. Useful for resume with different budget.
    callbacks : list of Callback or None, default None
        List of callback objects. See :mod:`~latents.callbacks` module.

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
        ``obs_posterior.x_dim`` does not match ``config.x_dim_init`` (fresh
        fits only; resumes may have pruned dimensions), or if ``tracker``
        and ``flags`` are not both provided or both `None`.

    Examples
    --------
    Most users should use :meth:`GFAModel.fit` instead of calling this directly.

    **Low-level usage**

    >>> from latents.gfa.inference import fit
    >>> obs_posterior, latents_posterior, tracker, flags = fit(
    ...     Y, obs_posterior, latents_posterior, config=config
    ... )
    """
    if config is None:
        config = GFAFitConfig()
    if obs_hyperprior is None:
        obs_hyperprior = ObsParamsHyperPrior()
    if callbacks is None:
        callbacks = []

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

    # Check that initial latent dimensionality matches (fresh fits only;
    # resumes may have pruned dimensions)
    if not resuming and obs_posterior.x_dim != x_dim_init:
        msg = "obs_posterior.x_dim must match config.x_dim_init."
        raise ValueError(msg)

    # Data size characteristics
    y_dims = Y.dims
    y_dim = y_dims.sum()
    n_samples = Y.data.shape[1]
    x_dim = obs_posterior.x_dim

    # Sample summary statistics (precomputed once, reused every iteration)
    Y2 = np.sum(Y.data**2, axis=1)
    Y_sum = np.sum(Y.data, axis=1)

    # Initialize the posterior covariance of C if needed
    if obs_posterior.C.cov is None:
        obs_posterior.C.cov = np.zeros((y_dim, x_dim, x_dim))

    # Reconstruct latents if cleared (enables seamless resume with save_x=False)
    if not latents_posterior.is_initialized():
        infer_latents(Y, obs_posterior, latents_posterior)

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

    # Create context for callbacks (holds references, reflects current state)
    ctx = GFAFitContext(
        config=config,
        obs_hyperprior=obs_hyperprior,
        obs_posterior=obs_posterior,
        latents_posterior=latents_posterior,
        tracker=tracker,
        flags=flags,
    )

    # Callback: fit started
    invoke_callbacks(callbacks, "on_fit_start", ctx=ctx)

    fit_iter = 0
    for fit_iter in range(max_iter):
        # Check if any latents need to be removed
        if prune_x:
            kept_x_dims = np.nonzero(
                np.mean(latents_posterior.mean**2, axis=1) > prune_tol
            )[0]
            if len(kept_x_dims) < x_dim:
                # Remove inactive latents
                n_removed = x_dim - len(kept_x_dims)
                obs_posterior.get_subset_dims(kept_x_dims, in_place=True)
                latents_posterior.get_subset_dims(kept_x_dims, in_place=True)
                flags.x_dims_removed += n_removed
                x_dim = obs_posterior.x_dim

                # Callback: x_dim pruned
                invoke_callbacks(
                    callbacks,
                    "on_x_dim_pruned",
                    ctx=ctx,
                    n_removed=n_removed,
                    x_dim_remaining=x_dim,
                    iteration=fit_iter,
                )

                if x_dim <= 0:
                    break

        # Start timer for current iteration
        if save_fit_progress:
            start_time = time.time()

        # Observation mean parameter, d
        infer_obs_mean(Y, obs_posterior, latents_posterior, obs_hyperprior)
        # Second moments for phi updates and lower bound
        d_moment = obs_posterior.d.cov + obs_posterior.d.mean**2

        # Mean-centered observations (d is fixed for rest of iteration)
        # Y.data: (y_dim, n_samples), d.mean: (y_dim,) -> (y_dim, n_samples)
        Y_centered = Y.data - obs_posterior.d.mean[:, np.newaxis]

        # Correlation matrix between latents and zero-centered observations
        # X.mean: (x_dim, n_samples) -> XY: (x_dim, y_dim)
        XY = latents_posterior.mean @ Y_centered.T

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
            Y_sum=Y_sum,
        )
        # Set minimum private variance
        obs_posterior.phi.mean[:] = np.minimum(1 / var_floor, obs_posterior.phi.mean)
        obs_posterior.phi.b[:] = obs_posterior.phi.a / obs_posterior.phi.mean

        # Latent variables, X
        infer_latents(Y, obs_posterior, latents_posterior, Y_centered=Y_centered)

        # Recompute reconstruction error with updated latents.
        # phi.b was computed before the X update, so it encodes a stale
        # reconstruction error. One extra XY matmul per iteration ensures
        # the ELBO uses the current X.
        XY = latents_posterior.mean @ Y_centered.T
        recon = _reconstruction_error(
            obs_posterior,
            latents_posterior,
            n_samples=n_samples,
            d_moment=d_moment,
            XY=XY,
            Y2=Y2,
            Y_sum=Y_sum,
        )

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
            recon=recon,
        )

        # Save progress
        if save_fit_progress:
            end_time = time.time()
            tracker.iter_time[iter_offset + fit_iter] = end_time - start_time
            tracker.lb[iter_offset + fit_iter] = lb_curr

        # Callback: iteration end
        invoke_callbacks(
            callbacks,
            "on_iteration_end",
            ctx=ctx,
            iteration=fit_iter,
            lb=lb_curr,
            lb_prev=lb_old,
        )

        # Check stopping conditions
        # Set lb_base during burn-in period (fresh fit only)
        if not resuming and fit_iter <= 1:
            tracker.lb_base = lb_curr
        elif lb_curr < lb_old:
            if not flags.decreasing_lb:
                flags.decreasing_lb = True
                invoke_callbacks(
                    callbacks,
                    "on_flag_changed",
                    ctx=ctx,
                    flag="decreasing_lb",
                    value=True,
                    iteration=fit_iter,
                )
        elif (lb_curr - tracker.lb_base) < (1 + fit_tol) * (lb_old - tracker.lb_base):
            if not flags.converged:
                flags.converged = True
                invoke_callbacks(
                    callbacks,
                    "on_flag_changed",
                    ctx=ctx,
                    flag="converged",
                    value=True,
                    iteration=fit_iter,
                )
            break

    # Truncate pre-allocated arrays to actual iteration count
    if save_fit_progress:
        total_iters = iter_offset + fit_iter + 1
        tracker.lb = tracker.lb[:total_iters]
        tracker.iter_time = tracker.iter_time[:total_iters]

    # Check if the variance floor was reached (post-fit check)
    if np.any(obs_posterior.phi.mean == 1 / var_floor) and not flags.private_var_floor:
        flags.private_var_floor = True
        invoke_callbacks(
            callbacks,
            "on_flag_changed",
            ctx=ctx,
            flag="private_var_floor",
            value=True,
            iteration=None,  # Post-fit check, no specific iteration
        )

    # Determine stop reason
    if flags.converged:
        reason = "converged"
    elif x_dim <= 0:
        reason = "no_latents"
    else:
        reason = "max_iter"

    # Callback: fit ended
    invoke_callbacks(callbacks, "on_fit_end", ctx=ctx, reason=reason)

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
    Y : ObsStatic
        Observed data.
    config : GFAFitConfig or None, default None
        Fitting configuration. If None, uses default `GFAFitConfig()`.
    obs_hyperprior : ObsParamsHyperPrior or None, default None
        Prior hyperparameters. If None, uses default `ObsParamsHyperPrior()`.

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
    # np.cov returns a scalar (0-d) when y_dim=1; ensure always 2-d
    Y_covs = [np.atleast_2d(np.cov(Y_m)) for Y_m in Ys]

    rng = np.random.default_rng(random_seed)

    # Latent variables
    latents_posterior.mean = rng.normal(size=(x_dim, n_samples))
    # Choose a small initial covariance
    latents_posterior.cov = stability_floor(Y.data.dtype) * np.eye(x_dim)
    latents_posterior.compute_moment()

    # Mean parameter d
    obs_posterior.d.mean = np.mean(Y.data, axis=1)
    obs_posterior.d.cov = np.full(y_dim, 1 / obs_hyperprior.beta_d)

    # Noise precisions phi
    obs_posterior.phi.a = obs_hyperprior.a_phi + n_samples / 2
    obs_posterior.phi.mean = np.concatenate(
        [1 / np.diag(Y_cov) for Y_cov in Y_covs], axis=0
    )
    # Moment-matched: b = a / mean ensures Gamma parameters are consistent
    obs_posterior.phi.b = obs_posterior.phi.a / obs_posterior.phi.mean

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
    obs_posterior.alpha.a = obs_hyperprior.a_alpha + y_dims / 2  # (n_groups,)
    obs_posterior.alpha.mean = np.zeros((n_groups, x_dim))
    for group_idx in range(n_groups):
        obs_posterior.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )
    # Moment-matched: b = a / mean ensures Gamma parameters are consistent
    # alpha.a: (n_groups,) -> (n_groups, 1), alpha.mean: (n_groups, x_dim)
    obs_posterior.alpha.b = (
        obs_posterior.alpha.a[:, np.newaxis] / obs_posterior.alpha.mean
    )

    return obs_posterior, latents_posterior


# -----------------------------------------------------------------------------
# Inference functions for individual parameters
# -----------------------------------------------------------------------------


def infer_latents(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic | None = None,
    Y_centered: np.ndarray | None = None,
) -> LatentsPosteriorStatic:
    """Infer latent posterior q(X) given observations and fitted parameters.

    Parameters
    ----------
    Y : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Posterior over observation parameters. Reads `C`, `phi`, `d`.
    latents_posterior : LatentsPosteriorStatic or None, default None
        If provided, update in-place and return.
        If `None`, create and return a new `LatentsPosteriorStatic`.
    Y_centered : ndarray or None, default None
        Pre-computed ``Y.data - d.mean[:, None]``, shape ``(y_dim, n_samples)``.
        If None, computed internally.

    Returns
    -------
    LatentsPosteriorStatic
        Posterior over latent variables.

    Examples
    --------
    Infer latents for new data using a fitted model:

    >>> X_new = infer_latents(Y_new, model.obs_posterior)
    >>> X_new.mean  # Posterior mean, shape (x_dim, n_samples)
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
    # phi.mean: (y_dim,), C.moment: (y_dim, x_dim, x_dim)
    # Weighted sum over y_dim -> (x_dim, x_dim)
    latents_posterior.cov[:] = np.linalg.inv(
        np.eye(x_dim)
        + np.einsum("i,ijk->jk", obs_posterior.phi.mean, obs_posterior.C.moment)
    )
    # Ensure symmetry
    latents_posterior.cov[:] = 0.5 * (latents_posterior.cov + latents_posterior.cov.T)

    # Mean: cov @ C^T diag(phi) @ (Y - d) -> mean: (x_dim, n_samples)
    # phi: (y_dim,) -> (1, y_dim) for broadcast with C.mean.T: (x_dim, y_dim)
    if Y_centered is None:
        Y_centered = Y.data - obs_posterior.d.mean[:, np.newaxis]
    latents_posterior.mean[:] = (
        latents_posterior.cov
        @ (obs_posterior.C.mean.T * obs_posterior.phi.mean[np.newaxis, :])
        @ Y_centered
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
    """Infer loading posterior q(C). Updates `obs_posterior.C` in-place.

    Parameters
    ----------
    Y : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Posterior over observation parameters. Reads `alpha`, `phi`, `d`; writes `C`.
    latents_posterior : LatentsPosteriorStatic
        Posterior over latents. Reads `mean`, `moment`.
    XY : ndarray or None, default None
        Pre-computed correlation matrix, shape `(x_dim, y_dim)`. Computed if not
        provided.

    Returns
    -------
    LoadingPosterior
        Reference to `obs_posterior.C` (updated in-place).
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
    """Infer ARD posterior q(alpha). Updates `obs_posterior.alpha` in-place.

    Parameters
    ----------
    obs_posterior : ObsParamsPosterior
        Posterior over observation parameters. Reads `C`; writes `alpha`.
    hyperprior : ObsParamsHyperPrior
        Hyperprior parameters (`a_alpha`, `b_alpha`).
    C_norm : ndarray or None, default None
        Pre-computed squared column norms of `C` per group, shape `(n_groups, x_dim)`.

    Returns
    -------
    ARDPosterior
        Reference to `obs_posterior.alpha` (updated in-place).
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
    """Infer observation mean posterior q(d). Updates `obs_posterior.d` in-place.

    Parameters
    ----------
    Y : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Posterior over observation parameters. Reads `C`, `phi`; writes `d`.
    latents_posterior : LatentsPosteriorStatic
        Posterior over latents. Reads `mean`.
    hyperprior : ObsParamsHyperPrior
        Hyperprior parameters (`beta_d`).

    Returns
    -------
    ObsMeanPosterior
        Reference to `obs_posterior.d` (updated in-place).
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
    Y_sum: np.ndarray | None = None,
) -> ObsPrecPosterior:
    """Infer precision posterior q(phi). Updates `obs_posterior.phi` in-place.

    Parameters
    ----------
    Y : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Posterior over observation parameters. Reads `C`, `d`; writes `phi`.
    latents_posterior : LatentsPosteriorStatic
        Posterior over latents. Reads `mean`, `moment`.
    hyperprior : ObsParamsHyperPrior
        Hyperprior parameters (`a_phi`, `b_phi`).
    d_moment : ndarray or None, default None
        Pre-computed second moment of `d`, shape `(y_dim,)`.
    XY : ndarray or None, default None
        Pre-computed correlation matrix, shape `(x_dim, y_dim)`.
    Y2 : ndarray or None, default None
        Pre-computed sample second moments, shape `(y_dim,)`.
    Y_sum : ndarray or None, default None
        Pre-computed column sums of `Y`, shape `(y_dim,)`.

    Returns
    -------
    ObsPrecPosterior
        Reference to `obs_posterior.phi` (updated in-place).
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
    if Y_sum is None:
        Y_sum = np.sum(Y.data, axis=1)

    # Rate parameter: expected reconstruction error -> phi.b: (y_dim,)
    recon = _reconstruction_error(
        obs_posterior,
        latents_posterior,
        n_samples=n_samples,
        d_moment=d_moment,
        XY=XY,
        Y2=Y2,
        Y_sum=Y_sum,
    )
    phi.b[:] = hyperprior.b_phi + 0.5 * recon

    # Mean
    phi.compute_mean(in_place=True)

    return phi


# -----------------------------------------------------------------------------
# Reconstruction error and lower bound computation
# -----------------------------------------------------------------------------


def _reconstruction_error(
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    n_samples: int,
    d_moment: np.ndarray,
    XY: np.ndarray,
    Y2: np.ndarray,
    Y_sum: np.ndarray,
) -> np.ndarray:
    """Compute expected reconstruction error per observed dimension.

    Computes E[sum_n (y_jn - d_j - c_j^T X_n)^2] for each dimension j.

    Parameters
    ----------
    obs_posterior : ObsParamsPosterior
        Posterior over observation parameters. Reads `C`, `d`.
    latents_posterior : LatentsPosteriorStatic
        Posterior over latents. Reads `moment`.
    n_samples : int
        Number of samples.
    d_moment : ndarray, shape (y_dim,)
        Second moment of observation mean, ``d.cov + d.mean**2``.
    XY : ndarray, shape (x_dim, y_dim)
        Cross-correlation ``X.mean @ (Y - d.mean).T``.
    Y2 : ndarray, shape (y_dim,)
        Sample second moments ``sum(Y**2, axis=1)``.
    Y_sum : ndarray, shape (y_dim,)
        Column sums ``sum(Y, axis=1)``.

    Returns
    -------
    ndarray, shape (y_dim,)
        Expected reconstruction error per observed dimension.
    """
    return (
        n_samples * d_moment
        + Y2
        - 2 * obs_posterior.d.mean * Y_sum
        - 2 * np.sum(obs_posterior.C.mean * XY.T, axis=1)
        # C.moment: (y_dim, x_dim, x_dim), X.moment: (x_dim, x_dim) -> (y_dim,)
        + np.einsum("ijk,jk->i", obs_posterior.C.moment, latents_posterior.moment)
    )


def compute_lower_bound(
    Y: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    latents_posterior: LatentsPosteriorStatic,
    obs_hyperprior: ObsParamsHyperPrior,
    consts: tuple | None = None,
    logdet_C: float | None = None,
    C_norm: np.ndarray | None = None,
    d_moment: np.ndarray | None = None,
    recon: np.ndarray | None = None,
) -> float:
    """Compute the variational lower bound (ELBO) for a GFA model.

    Parameters
    ----------
    Y : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Observation model posterior.
    latents_posterior : LatentsPosteriorStatic
        Latent posterior.
    obs_hyperprior : ObsParamsHyperPrior
        Hyperparameters of the prior distributions.
    consts : tuple or None, default None
        Constant factors in the lower bound. If `None`, computed.
    logdet_C : float or None, default None
        Log-determinant of loading covariances. If `None`, computed.
    C_norm : ndarray or None, default None
        Expected squared norms of loading columns. If `None`, computed.
    d_moment : ndarray or None, default None
        Second moment of observation mean. If `None`, computed.
    recon : ndarray or None, default None
        Expected reconstruction error per dimension, shape ``(y_dim,)``.
        If `None`, derived from ``obs_posterior.phi.b``.

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

    # Derive reconstruction error from phi.b if not provided.
    # phi.b = b_phi + 0.5 * recon, so recon = 2 * (phi.b - b_phi).
    if recon is None:
        recon = 2.0 * (obs_posterior.phi.b - obs_hyperprior.b_phi)

    # Likelihood term
    log_phi = digamma_a_phi - np.log(np.maximum(obs_posterior.phi.b, floor))
    lb = (
        const_lik
        + 0.5 * n_samples * np.sum(log_phi)
        - 0.5 * np.sum(obs_posterior.phi.mean * recon)
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
        # y_dims: (n_groups,) -> (n_groups, 1) for broadcast with
        # log_alpha: (n_groups, x_dim)
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
    n_samples : int
        Number of samples in the observed data.
    obs_posterior : ObsParamsPosterior
        Observation model posterior.
    obs_hyperprior : ObsParamsHyperPrior
        Hyperparameters of the prior distributions.

    Returns
    -------
    tuple of (float, float, float, float, float, float, float, float, ndarray, ndarray)
        Constant factors: `const_lik`, `const_d`, `alogb_phi`, `loggamma_a_phi_prior`,
        `loggamma_a_phi_post`, `digamma_a_phi`, `alogb_alpha`, `loggamma_a_alpha_prior`,
        `loggamma_a_alpha_post`, `digamma_a_alpha`.
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
