"""
Core utilities to fit a delayed latents across multiple groups (mDLAG) model to data.

**Functions**

- :func:`fit` -- Fit a mDLAG model to data.
- :func:`init` -- Initialize mDLAG model parameters to data prior to fitting.
- :func:`infer_latents` -- Infer latent variables.
- :func:`learn_gp_params` -- Learn Gaussian process parameters using mDLAGGP.
- :func:`infer_loadings` -- Infer loading matrices.
- :func:`infer_ard` -- Infer ARD parameters.
- :func:`infer_obs_mean` -- Infer observation mean parameter.
- :func:`infer_obs_prec` -- Infer observation precision parameters.
- :func:`compute_lower_bound` -- Compute the variational lower bound.
- :func:`compute_lower_bound_constants` -- Compute constants in the lower bound.

**Classes**

- :class:`mDLAGModel` -- A wrapper class to store mDLAG fitting results.

"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.special import gammaln, psi
from scipy.stats import gmean

from latents.mdlag.data_types import (
    mDLAGFitArgs,
    mDLAGFitFlags,
    mDLAGFitTracker,
    mDLAGParams,
)
from latents.mdlag.gp.fit_config import GPFitConfig
from latents.mdlag.gp.gp_model import mDLAGGP
from latents.mdlag.gp.kernels.rbf.rbf_kernel import RBFKernel
from latents.observation_model.observations import ObsTimeSeries
from latents.observation_model.probabilistic import (
    HyperPriorParams,
    PosteriorARD,
    PosteriorLoading,
    PosteriorObsMean,
    PosteriorObsPrec,
)
from latents.state_model.latents import PosteriorLatentDelayed


def fit(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    gp_fit_config: GPFitConfig,
    max_iter: int = int(1e6),
    fit_tol: float = 1e-8,
    prune_X: bool = True,
    prune_tol: float = 1e-7,
    verbose: bool = False,
    random_seed: int | None = 42,
    save_X_cov: bool = False,
    save_C_cov: bool = False,
    save_fit_progress: bool = True,
    checkpoint_interval: int = 0,
    checkpoint_dir: str = "checkpoints",
) -> tuple[mDLAGParams, mDLAGFitTracker, mDLAGFitFlags]:
    """Fit a mDLAG model to data.

    Fit a delayed latents across multiple groups (mDLAG) model using an iterative
    variational inference scheme with mean-field approximation.
    """
    if not params.is_initialized():
        if verbose:
            print("Initializing mDLAG model parameters...")
        gp_init = params.gp
        params = init(
            Y,
            gp_init,
            hyper_priors,
            random_seed=random_seed,
            save_C_cov=save_C_cov,
            save_X_cov=save_X_cov,
        )

    obs_params = params.obs_params
    state_params = params.state_params
    gp = params.gp

    # Initialize hyper_priors if not provided
    if hyper_priors is None:
        hyper_priors = HyperPriorParams()
    elif not isinstance(hyper_priors, HyperPriorParams):
        msg = "hyper_priors must be a HyperPriorParams object."
        raise TypeError(msg)

    if gp_fit_config is None:
        gp_fit_config = GPFitConfig()
    elif not isinstance(gp_fit_config, GPFitConfig):
        msg = "gp_fit_config must be a GPFitConfig object."
        raise TypeError(msg)

    # Check that the observed data dimensions match between the data and the
    # parameters
    if not np.array_equal(obs_params.y_dims, Y.dims):
        msg = "params.obs_params.y_dims must match Y.dims."
        raise ValueError(msg)

    if obs_params.x_dim != gp.params.x_dim:
        msg = "params.obs_params.x_dim must match gp.params.x_dim."
        raise ValueError(msg)

    if state_params.x_dim != gp.params.x_dim:
        msg = "params.state_params.x_dim must match gp.params.x_dim."
        raise ValueError(msg)

    # Get data size characteristics:
    y_dims = Y.dims
    y_dim = y_dims.sum()
    N = Y.data.shape[2]
    T = Y.T
    x_dim = gp.params.x_dim
    num_groups = len(y_dims)

    # Initialize the posterior covariance of C, if needed
    if obs_params.C.cov is None:
        obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))

    # Constant factors in the lower bound
    consts_lb = compute_lower_bound_constants(N, params, hyper_priors)

    # Initialize tracked quantities
    tracker = mDLAGFitTracker()
    if save_fit_progress:
        tracker.lb = np.array([])  # Lower bound
        tracker.iter_time = np.array([])  # Runtime per iteration
    lb_curr = -np.inf  # Initial lower bound

    # Initialize status flags
    flags = mDLAGFitFlags()

    # Initialize checkpointing
    checkpoint_enabled = checkpoint_interval > 0
    if checkpoint_enabled:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if verbose:
            print(
                f"Checkpointing enabled: saving every {checkpoint_interval} iterations to {checkpoint_dir}"
            )

    fit_iter = 0
    for fit_iter in range(max_iter):
        # Check if any latents need to be removed
        if prune_X:
            is_active = np.zeros((x_dim, num_groups), dtype=bool)
            for group_idx in range(num_groups):
                # Calculate mean squared activity for each latent in the current group
                # Xs[groupIdx] has shape (x_dim, T_group), so we average over the time axis.
                Xs = state_params.X.mean[:, group_idx, :, :].reshape(x_dim, -1)
                activity = np.mean(Xs**2, axis=1)
                is_active[:, group_idx] = activity > prune_tol

            # A dimension is kept if it's active in at least one group
            kept_x_dims = np.flatnonzero(np.any(is_active, axis=1))

            if len(kept_x_dims) < x_dim:
                # Remove inactive latents
                if verbose:
                    print(f"\nRemoving {x_dim - len(kept_x_dims)} latents")
                params.get_subset_dims(kept_x_dims, in_place=True)
                flags.x_dims_removed += x_dim - obs_params.x_dim
                x_dim = obs_params.x_dim
                if x_dim <= 0:
                    # Stop fitting if no significant latents remain
                    break

        # Start timer for current iteration
        if save_fit_progress:
            start_time = time.time()

        # Latent variables, X
        infer_latents(Y, params, in_place=True)
        X_moment = np.copy(params.state_params.X.moment_gp)

        # GP parameters
        l_gp = params.gp.fit(X_moment, N, T, gp_fit_config, in_place=True)

        # Mean parameter, d
        infer_obs_mean(Y, params, hyper_priors, in_place=True)

        # Loading matrices, C
        infer_loadings(Y, params, XY=None, in_place=True)

        # ARD parameters, alpha
        infer_ard(params, hyper_priors, in_place=True)

        # Observation precision parameters, phi
        infer_obs_prec(Y, params, hyper_priors, in_place=True)

        # Compute the lower bound
        lb_old = 1.0 * lb_curr
        lb_curr = compute_lower_bound(
            Y, params, hyper_priors, consts=consts_lb, gp_loss=l_gp
        )

        # Save progress
        if save_fit_progress:
            # Compute the runtime of this iteration
            end_time = time.time()
            tracker.iter_time = np.append(tracker.iter_time, end_time - start_time)
            # Record the current lower bound
            tracker.lb = np.append(tracker.lb, lb_curr)

        # Save checkpoint if enabled and at the right interval
        if checkpoint_enabled and (fit_iter + 1) % checkpoint_interval == 0:
            checkpoint_filename = os.path.join(
                checkpoint_dir, f"checkpoint_iter_{fit_iter + 1}.pkl"
            )

            try:
                # Create a temporary mDLAGModel object with current state to leverage existing save method
                from latents.mdlag.data_types import mDLAGFitArgs

                # Create fit_args object with current parameters
                fit_args = mDLAGFitArgs()
                fit_args.set_args(
                    hyper_priors=hyper_priors,
                    gp_fit_config=gp_fit_config,
                    max_iter=max_iter,
                    fit_tol=fit_tol,
                    prune_X=prune_X,
                    prune_tol=prune_tol,
                    verbose=verbose,
                    random_seed=random_seed,
                    save_X_cov=save_X_cov,
                    save_C_cov=save_C_cov,
                    save_fit_progress=save_fit_progress,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                )

                # Create temporary model object and use existing save method
                checkpoint_model = mDLAGModel(
                    params=params, tracker=tracker, flags=flags, fit_args=fit_args
                )
                checkpoint_model.save(checkpoint_filename)

                if verbose:
                    print(
                        f"\nCheckpoint saved at iteration {fit_iter + 1}: {checkpoint_filename}"
                    )
                    print(
                        f"Iteration {fit_iter + 1} of {max_iter}        lb {lb_curr}",
                        end="",
                        flush=True,
                    )
            except Exception as e:
                if verbose:
                    print(f"\nERROR saving checkpoint at iteration {fit_iter + 1}: {e}")
                    print(
                        f"Iteration {fit_iter + 1} of {max_iter}        lb {lb_curr}",
                        end="",
                        flush=True,
                    )

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

    return params, tracker, flags


def init(
    Y: ObsTimeSeries,
    gp_init: mDLAGGP,
    hyper_priors: HyperPriorParams | None = None,
    random_seed: int | None = None,
    save_C_cov: bool = False,
    save_X_cov: bool = False,
) -> mDLAGParams:
    """Initialize mDLAG model parameters for fitting.

    Parameters
    ----------
    Y
        Observed time series data.
    gp_init
        Initial Gaussian process parameters.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions. If not provided,
        default hyperparameters will be used.
    random_seed
        Seed the random number generator for reproducibility. Defaults to
        ``None``.
    save_C_cov
        Set to ``True`` to save posterior covariance of :math:`C`. For large
        ``y_dim`` and ``x_dim``, these structures can use a lot of memory.
        Defaults to ``False``.
    save_X_cov
        Set to ``True`` to save posterior covariance of :math:`X`. For large
        datasets, this matrix can use a lot of memory. Defaults to ``False``.

    Returns
    -------
    mDLAGParams
        Initialized mDLAG model parameters.

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

    # Seed the random number generator for reproducible initialization
    rng = np.random.default_rng(random_seed)

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    num_groups = len(y_dims)  # Number of observed groups
    N = Y.data.shape[2]  # Number of samples
    T = Y.T  # Number of time points
    x_dim = gp_init.params.x_dim  # Number of latent dimensions

    # Get views of the observed data for each group
    Ys = Y.get_groups()
    # Get the variance of each observed group
    Y_covs = [np.cov(Y_m.reshape(Y_m.shape[0], -1)) for Y_m in Ys]

    # Initialize mDLAG parameter object
    params = mDLAGParams(x_dim, y_dims, T, gp_init, save_X_cov, save_C_cov)
    obs_params = params.obs_params
    state_params = params.state_params

    # Mean parameter
    obs_params.d.mean = np.mean(Y.data, axis=(1, 2))
    obs_params.d.cov = np.full(y_dim, 1 / hyper_priors.d_beta)

    # Noise precisions
    obs_params.phi.a = hyper_priors.a_phi + N * T / 2
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
        eigs = eigh(Y_covs[group_idx], eigvals_only=True)
        scale = gmean(eigs[eigs > 0])
        C_means[group_idx][:] = rng.normal(
            scale=np.sqrt(scale / x_dim), size=(y_dims[group_idx], x_dim)
        )

    obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))
    # Second moments
    obs_params.C.compute_moment()
    if not save_C_cov:
        # Delete the loading matrix covariances to save memory
        obs_params.C.cov = None
    # Get views of the loading matrix moments for each group
    _, _, C_moments = obs_params.C.get_groups(y_dims)

    # ARD parameters
    obs_params.alpha.a = hyper_priors.a_alpha + y_dims / 2
    obs_params.alpha.b = np.full((num_groups, x_dim), hyper_priors.b_alpha)
    # Scale ARD parameters to match the data
    obs_params.alpha.mean = np.zeros((num_groups, x_dim))
    for group_idx in range(num_groups):
        obs_params.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    # Initialize posterior mean of the latents:
    infer_latents(Y, params, in_place=True)

    # If we're not saving X.cov, set it to None after initialization
    if not save_X_cov:
        state_params.X.cov = None

    return params


def infer_latents(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    in_place: bool = True,
) -> PosteriorLatentDelayed | None:
    """Infer latent variables given mDLAG model parameters and observed data.

    Parameters
    ----------
    Y
        Observed time series data.
    params
        mDLAG model parameters.
    in_place
        If ``True``, update the posterior latents in place.
        If ``False``, compute the posterior latents and return as a
        new ``PosteriorLatentDelayed`` without modifying ``params``. Defaults to
        ``True``.

    Returns
    -------
    PosteriorLatentDelayed | None
        Posterior estimates of latent variables. If ``in_place=True``, returns
        ``None``. Otherwise, returns the computed posterior latents.
    """
    obs_params = params.obs_params
    state_params = params.state_params
    gp = params.gp

    # Check if gp_params is initialized
    if gp is None:
        error_msg = "GP parameters must be initialized before inferring latents"
        raise ValueError(error_msg)

    x_dim = state_params.x_dim
    y_dims = obs_params.y_dims
    num_groups = len(obs_params.y_dims)
    T = params.T
    N = Y.data.shape[2]

    K_big = gp.build_kernel_matrix(T, return_tensor=False)

    # Initialize X, if needed
    if in_place:
        if state_params.X.mean is None:
            state_params.X.mean = np.zeros((x_dim, num_groups, T, N))
        if state_params.X.cov is None:
            state_params.X.cov = np.zeros((x_dim, num_groups, T, x_dim, num_groups, T))
        if state_params.X.moment is None:
            state_params.X.moment = np.zeros((num_groups, x_dim, x_dim))
        X = state_params.X
    else:
        X = PosteriorLatentDelayed(
            mean=np.zeros((x_dim, num_groups, T, N)),
            cov=np.zeros((x_dim, num_groups, T, x_dim, num_groups, T)),
            moment=np.zeros((num_groups, x_dim, x_dim)),
        )
    # Covariance
    phi_means, _ = obs_params.phi.get_groups(y_dims)
    C_means, _, C_moments = obs_params.C.get_groups(y_dims)
    CPhiC_diag = []
    for g in range(num_groups):
        CPhiC_diag.append((phi_means[g][:, None, None] * C_moments[g]).sum(axis=0))

    CPhiC = block_diag(*CPhiC_diag)
    CPhiC_big = block_diag(*[CPhiC] * T)

    SigX = np.linalg.inv(np.linalg.inv(K_big) + CPhiC_big)

    # Ensure symmetry
    SigX = 0.5 * (SigX + SigX.T)

    # Compute log determinant
    X.logdet_SigX = np.linalg.slogdet(SigX)[1]

    # Covariance
    X.cov[:] = np.reshape(SigX, (x_dim, num_groups, T, x_dim, num_groups, T), order="F")

    # Mean
    CPhi = block_diag(*C_means).T * obs_params.phi.mean
    CPhi_big = block_diag(*[CPhi] * T)
    Y_centered = Y.data - obs_params.d.mean[:, None, None]
    muX = SigX @ CPhi_big @ Y_centered.reshape(-1, N, order="F")
    X.mean[:] = np.reshape(muX, (x_dim, num_groups, T, N), order="F")

    # Compute moment
    X.compute_moment(in_place=True)
    X.compute_moment_gp(in_place=True)

    if not params.save_X_cov:
        X.cov = None

    return None if in_place else X


def infer_loadings(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    in_place: bool = True,
    XY: np.ndarray | None = None,
) -> PosteriorLoading | None:
    """
    Infer loadings :math:`C` given current params and observed data.

    Parameters
    ----------
    Y
        Observed time series data.
    params
        mDLAG model parameters.
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
    y_dims = obs_params.y_dims
    y_dim = y_dims.sum()  # Total number of observed dimensions
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
        Y0 = Y.data - obs_params.d.mean[:, np.newaxis, np.newaxis]
        group_membership = np.repeat(np.arange(num_groups), y_dims)
        XY = np.einsum(
            "xytn,ytn->xy", state_params.X.mean[:, group_membership, :, :], Y0
        )

    # Get views of the loading matrices and precision parameters for each group
    _, C_covs, _ = C.get_groups(obs_params.y_dims)
    phi_means, _ = obs_params.phi.get_groups(obs_params.y_dims)

    for group_idx in range(num_groups):
        C_covs[group_idx][:] = np.linalg.inv(
            np.diag(obs_params.alpha.mean[group_idx, :])
            + phi_means[group_idx][:, np.newaxis, np.newaxis]
            * state_params.X.moment[group_idx, :, :]
        )
    # Mean
    phi_C_cov = obs_params.phi.mean[:, np.newaxis, np.newaxis] * obs_params.C.cov
    C.mean[:] = np.einsum("ijk,ik->ij", phi_C_cov, XY.T)
    # Second moment
    C.compute_moment(in_place=True)

    return None if in_place else C


def infer_ard(
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
    C_norm: np.ndarray | None = None,
) -> PosteriorARD | None:
    """
    Infer ARD parameters alpha given current params.

    Parameters
    ----------
    params
        mDLAG model parameters.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions.
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
    y_dims = obs_params.y_dims
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
        C_norm = obs_params.C.compute_squared_norms(y_dims)

    # Rate parameters
    alpha.b[:] = hyper_priors.b_alpha + 0.5 * C_norm

    # Mean
    alpha.compute_mean(in_place=True)

    return None if in_place else alpha


def infer_obs_mean(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
) -> PosteriorObsMean | None:
    """
    Infer observation mean parameter given current params and observed data.

    Parameters
    ----------
    Y
        Observed time series data.
    params
        mDLAG model parameters.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions.
    in_place
        If ``True``, update the posterior observation mean parameters in place.
        If ``False``, compute the posterior observation mean parameters and
        return as a new ``PosteriorObsMean`` without modifying ``params``.
        Defaults to ``True``.

    Returns
    -------
    PosteriorObsMean | None
        Posterior estimates of observation mean parameters. If ``in_place=True``,
        returns ``None``. Otherwise, returns the computed posterior mean.
    """
    # Extract parameters and dimensions
    obs_params = params.obs_params
    state_params = params.state_params
    y_dim, T, N = Y.data.shape
    y_dims = obs_params.y_dims
    num_groups = len(y_dims)
    x_dim = state_params.x_dim
    T = Y.T

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
    d.cov[:] = 1 / (hyper_priors.d_beta + N * T * obs_params.phi.mean)

    # Calculate posterior mean for each group
    Ys = Y.get_groups()
    C_means, _, _ = obs_params.C.get_groups(y_dims)
    phi_means, _ = obs_params.phi.get_groups(y_dims)
    d_means, d_covs = d.get_groups(y_dims)

    for group_idx in range(num_groups):
        Y_reshaped = Ys[group_idx].reshape(y_dims[group_idx], -1)
        X_reshaped = state_params.X.mean[:, group_idx, :, :].reshape(x_dim, -1)
        residual = Y_reshaped - C_means[group_idx] @ X_reshaped
        d_means[group_idx] = (
            d_covs[group_idx] * phi_means[group_idx] * np.sum(residual, axis=1)
        )

    return None if in_place else d


def infer_obs_prec(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
    d_moment: np.ndarray | None = None,
    XY: np.ndarray | None = None,
    Y2: np.ndarray | None = None,
) -> PosteriorObsPrec | None:
    """
    Infer observation precision parameter given current params and observed data.

    Parameters
    ----------
    Y
        Observed time series data.
    params
        mDLAG model parameters.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions.
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
    y_dims = obs_params.y_dims
    y_dim = y_dims.sum()  # Total number of observed dimensions
    N = Y.data.shape[2]
    T = Y.T
    group_membership = np.repeat(np.arange(len(y_dims)), y_dims)

    # Initialize phi, if needed
    if in_place:
        if obs_params.phi.mean is None:
            obs_params.phi.mean = np.zeros(y_dim)
        if obs_params.phi.a is None:
            obs_params.phi.a = hyper_priors.a_phi + N * T / 2
        if obs_params.phi.b is None:
            obs_params.phi.b = np.zeros(y_dim)
        phi = obs_params.phi
    else:
        phi = PosteriorObsPrec(
            mean=np.zeros(y_dim), a=hyper_priors.a_phi + N * T / 2, b=np.zeros(y_dim)
        )
    # Pre-computations
    # Sample second moments of observed data
    if Y2 is None:
        Y2 = np.sum(Y.data**2, axis=(1, 2))

    # Second moment of the observation mean parameter
    if d_moment is None:
        d_moment = obs_params.d.cov + obs_params.d.mean**2

    # Correlation matrix between latents and zero-centered observations
    if XY is None:
        Y0 = Y.data - obs_params.d.mean[:, np.newaxis, np.newaxis]
        XY = np.einsum(
            "xytn,ytn->xy", state_params.X.mean[:, group_membership, :, :], Y0
        )

    # Rate parameter
    phi.b[:] = hyper_priors.b_phi + 0.5 * (
        N * T * d_moment
        + Y2
        - 2 * np.sum(obs_params.d.mean[:, np.newaxis, np.newaxis] * Y.data, axis=(1, 2))
        - 2 * np.sum(obs_params.C.mean * XY.T, axis=1)
        + np.sum(
            obs_params.C.moment * state_params.X.moment[group_membership, :, :],
            axis=(1, 2),
        )
    )

    # Mean
    phi.compute_mean(in_place=True)

    return None if in_place else phi


def compute_lower_bound(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    consts: float | None = None,
    gp_loss: float | None = None,
    logdet_C: float | None = None,
    C_norm: np.ndarray | None = None,
    d_moment: np.ndarray | None = None,
) -> float:
    """Compute the variational lower bound for a mDLAG model on observed data.

    Parameters
    ----------
    Y
        Observed data.
    params
        mDLAG model parameters.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions.
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
    y_dims = obs_params.y_dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    x_dim = obs_params.x_dim  # Number of latent dimensions
    num_groups = len(obs_params.y_dims)  # Number of observed groups
    N = Y.data.shape[2]  # Number of samples
    T = Y.T

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
        const_gp,
    ) = consts

    # Other pre-computations
    if logdet_C is None:
        logdet_C = np.sum(np.linalg.slogdet(obs_params.C.cov)[1])
    if C_norm is None:
        C_norm = obs_params.C.compute_squared_norms(y_dims)
    if d_moment is None:
        d_moment = obs_params.d.cov + obs_params.d.mean**2
    if gp_loss is None:
        # Use the mDLAGGP's built-in loss computation
        X_moment_GP = params.state_params.X.compute_moment_gp(in_place=False)
        N = params.state_params.X.mean.shape[3]
        T = params.state_params.X.mean.shape[2]
        gp_loss = params.gp.compute_loss(X_moment_GP, N, T)

    # Likelihood term
    log_phi = digamma_a_phi - np.log(obs_params.phi.b)  # (y_dim x 1) array
    lb_lik = (
        const_lik
        + 0.5 * N * T * np.sum(log_phi)
        - np.sum(obs_params.phi.mean * (obs_params.phi.b - hyper_priors.b_phi))
    )

    # X KL term
    lb_x = const_gp + gp_loss + 0.5 * N * params.state_params.X.logdet_SigX

    # C KL term
    log_alpha = np.zeros((num_groups, x_dim))
    for group_idx in range(num_groups):
        log_alpha[group_idx, :] = digamma_a_alpha[group_idx] - np.log(
            obs_params.alpha.b[group_idx, :]
        )

    lb_C = 0.5 * (
        x_dim * y_dim
        + logdet_C
        + np.sum(y_dims[:, np.newaxis] * log_alpha - obs_params.alpha.mean * C_norm)
    )

    # alpha KL term
    val = 0
    temp1 = num_groups * x_dim * (alogb_alpha - loggamma_a_alpha_prior)
    val += temp1
    for group_idx in range(num_groups):
        temp2 = np.sum(
            -obs_params.alpha.a[group_idx] * np.log(obs_params.alpha.b[group_idx, :])
            - hyper_priors.b_alpha * obs_params.alpha.mean[group_idx, :]
            + (hyper_priors.a_alpha - obs_params.alpha.a[group_idx])
            * log_alpha[group_idx]
        )
        val += temp2
        temp3 = x_dim * (
            loggamma_a_alpha_post[group_idx] + obs_params.alpha.a[group_idx]
        )
        val += temp3

    lb_alpha = val

    # phi KL term
    lb_phi = y_dim * (
        alogb_phi + loggamma_a_phi_post - loggamma_a_phi_prior + obs_params.phi.a
    ) + np.sum(
        -obs_params.phi.a * np.log(obs_params.phi.b)
        + hyper_priors.b_phi * obs_params.phi.mean
        + (hyper_priors.a_phi - obs_params.phi.a) * log_phi
    )

    # d KL term
    lb_d = const_d + 0.5 * (
        np.sum(np.log(obs_params.d.cov)) - hyper_priors.d_beta * np.sum(d_moment)
    )

    return lb_lik + lb_x + lb_C + lb_alpha + lb_phi + lb_d


def compute_lower_bound_constants(
    N: int,
    params: mDLAGParams,
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
    obs_params = params.obs_params
    y_dim = obs_params.y_dims.sum()  # Total number of observed dimensions
    num_groups = len(obs_params.y_dims)  # Number of observed groups
    x_dim = params.state_params.x_dim  # Number of latent dimensions
    T = params.T  # Number of time points

    # Constant factors in the lower bound
    # Related to the likelihood
    const_lik = -(y_dim * N * T / 2) * np.log(2 * np.pi)
    # Related to observation mean parameters
    const_d = 0.5 * y_dim + 0.5 * y_dim * np.log(hyper_priors.d_beta)
    # Related to observation precision parameters
    alogb_phi = hyper_priors.a_phi * np.log(hyper_priors.b_phi)
    loggamma_a_phi_prior = gammaln(hyper_priors.a_phi)
    loggamma_a_phi_post = gammaln(obs_params.phi.a)
    digamma_a_phi = psi(obs_params.phi.a)
    # Related to ARD parameters
    alogb_alpha = float(hyper_priors.a_alpha) * np.log(float(hyper_priors.b_alpha))
    loggamma_a_alpha_prior = gammaln(float(hyper_priors.a_alpha))
    loggamma_a_alpha_post = gammaln(obs_params.alpha.a)
    digamma_a_alpha = psi(obs_params.alpha.a)

    # Related to GP:
    const_gp = num_groups * x_dim * N * T / 2

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
        const_gp,
    )


class mDLAGModel:
    """
    Interface with, fit, and store the fitting results of a mDLAG model.

    Parameters
    ----------
    params
        Current mDLAG model parameters.
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
        params: mDLAGParams | None = None,
        tracker: mDLAGFitTracker | None = None,
        flags: mDLAGFitFlags | None = None,
        fit_args: mDLAGFitArgs | None = None,
    ):
        # Estimated parameters
        if params is None:
            self.params = mDLAGParams()
        elif not isinstance(params, mDLAGParams):
            msg = "params must be a mDLAGParams object."
            raise TypeError(msg)
        else:
            self.params = params

        # Fit tracker
        if tracker is None:
            self.tracker = mDLAGFitTracker()
        elif not isinstance(tracker, mDLAGFitTracker):
            msg = "tracker must be a mDLAGFitTracker object."
            raise TypeError(msg)
        else:
            self.tracker = tracker

        # Fit flags
        if flags is None:
            self.flags = mDLAGFitFlags()
        elif not isinstance(flags, mDLAGFitFlags):
            msg = "flags must be a mDLAGFitFlags object."
            raise TypeError(msg)
        else:
            self.flags = flags

        # Fit keyword arguments
        if fit_args is None:
            self.fit_args = mDLAGFitArgs()
        elif not isinstance(fit_args, mDLAGFitArgs):
            msg = "fit_args must be a mDLAGFitArgs object."
            raise TypeError(msg)
        else:
            self.fit_args = fit_args

    def __repr__(self) -> str:
        return (
            f"mDLAGModel(params={self.params}, "
            f"tracker={self.tracker}, "
            f"flags={self.flags}, "
            f"fit_args={self.fit_args})"
        )

    def fit(self, Y: ObsTimeSeries) -> None:
        """
        Fit a mDLAG model to data.

        Fit a mDLAG model to data. Uses the current model parameters as initial
        values, and uses the current keyword arguments.

        Parameters
        ----------
        Y
            Observed time series data.
        """
        # Initialize mDLAG model parameters if they have not been initialized
        if not self.params.is_initialized():
            if self.fit_args.verbose:
                print("mDLAG model parameters not initialized. Initializing...")
            self.init(Y)

        # Fit the model
        self.params, self.tracker, self.flags = fit(
            Y, self.params, **self.fit_args.get_args()
        )

    def init(
        self,
        Y: ObsTimeSeries,
        x_dim_init: int = 1,
        bin_width: int = 1,
        kernel: RBFKernel = RBFKernel(),
        eps: float = 1e-3,
    ) -> None:
        """
        Initialize mDLAG model parameters.

        Parameters
        ----------
        Y
            Observed time series data.
        """
        # Get a subset of keyword arguments to pass to core.init
        init_args = ["hyper_priors", "random_seed", "save_C_cov", "save_X_cov"]
        kwargs = {
            key: value
            for key, value in self.fit_args.get_args().items()
            if key in init_args
        }

        # Create a default GP initialization if none exists
        if self.params.gp is None:
            gp_init = mDLAGGP.initialize_with_defaults(
                T=Y.T,
                x_dim=x_dim_init,
                num_groups=len(Y.dims),
                bin_width=bin_width,
                kernel=kernel,
                eps=eps,
            )
        else:
            gp_init = self.params.gp

        self.params = init(Y, gp_init, **kwargs)

    def save(self, filename: str) -> None:
        """
        Save a mDLAGModel object to a pickle file.

        Parameters
        ----------
        filename
            Name of pickle file to save to.
        """
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> mDLAGModel:
        """
        Load a mDLAGModel object from a pickle file.

        Parameters
        ----------
        filename
            Name of pickle file to load from.

        Returns
        -------
        mDLAGModel
            Loaded mDLAGModel object.
        """
        import pickle

        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_from_checkpoint(checkpoint_filename: str) -> mDLAGModel:
        """
        Load a mDLAGModel object from a checkpoint file.

        Parameters
        ----------
        checkpoint_filename
            Name of checkpoint file to load from.

        Returns
        -------
        mDLAGModel
            Loaded mDLAGModel object with fit state restored from checkpoint.
        """
        # Since checkpoints are now saved using the standard save() method,
        # we can simply use the existing load() method
        return mDLAGModel.load(checkpoint_filename)

    def resume_fit(
        self, Y: ObsTimeSeries, checkpoint_filename: str | None = None
    ) -> None:
        """
        Resume fitting from a checkpoint.

        Parameters
        ----------
        Y
            Observed time series data.
        checkpoint_filename
            Optional checkpoint filename to resume from. If None, will look for
            the most recent checkpoint in the checkpoint directory.
        """
        if checkpoint_filename is None:
            # Find the most recent checkpoint
            checkpoint_dir = self.fit_args.checkpoint_dir
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(
                    f"Checkpoint directory {checkpoint_dir} does not exist"
                )

            checkpoint_files = [
                f
                for f in os.listdir(checkpoint_dir)
                if f.startswith("checkpoint_iter_") and f.endswith(".pkl")
            ]
            if not checkpoint_files:
                raise FileNotFoundError(
                    f"No checkpoint files found in {checkpoint_dir}"
                )

            # Sort by iteration number to get the most recent
            def extract_iter(filename):
                return int(filename.split("_iter_")[1].split(".pkl")[0])

            checkpoint_files.sort(key=extract_iter, reverse=True)
            checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_files[0])

        # Load checkpoint state using existing load method
        model = mDLAGModel.load(checkpoint_filename)

        # Update current model state
        self.params = model.params
        self.tracker = model.tracker
        self.flags = model.flags
        self.fit_args = model.fit_args

        if self.fit_args.verbose:
            print(f"Resuming from checkpoint: {checkpoint_filename}")

        # Continue fitting
        self.fit(Y)

    def infer_latents(
        self,
        Y: ObsTimeSeries,
        in_place: bool = True,
    ) -> PosteriorLatentDelayed | None:
        """
        Infer latent variables X given current params and observed data.

        Parameters
        ----------
        Y
            Observed time series data.
        in_place
            If ``True``, update the posterior latents in place.
            If ``False``, compute the posterior latents and return as a
            new ``PosteriorLatentDelayed`` without modifying ``params``. Defaults to
            ``True``.

        Returns
        -------
        PosteriorLatentDelayed | None
            Posterior estimates of latent variables. If ``in_place=True``, returns
            ``None``. Otherwise, returns the computed posterior latents.
        """
        return infer_latents(Y, self.params, in_place=in_place)

    def infer_loadings(
        self,
        Y: ObsTimeSeries,
        in_place: bool = True,
    ) -> PosteriorLoading | None:
        """
        Infer loadings C given current params and observed data.

        Parameters
        ----------
        Y
            Observed time series data.
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
        Y: ObsTimeSeries,
        in_place: bool = True,
    ) -> PosteriorObsMean | None:
        """
        Infer observation mean parameters given current params and observed data.

        Parameters
        ----------
        Y
            Observed time series data.
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
        Y: ObsTimeSeries,
        in_place: bool = True,
    ) -> PosteriorObsPrec | None:
        """
        Infer observation precision parameters given current params and observed data.

        Parameters
        ----------
        Y
            Observed time series data.
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
        Y: ObsTimeSeries,
    ) -> float:
        """
        Compute the variational lower bound given observed data.

        Parameters
        ----------
        Y
            Observed time series data.

        Returns
        -------
        float
            Variational lower bound.
        """
        return compute_lower_bound(Y, self.params, self.fit_args.hyper_priors)
