"""
Core utilities to fit a delayed latents across multiple groups (mDLAG) model to data.

**Functions**

- :func:`fit` -- Fit a mDLAG model to data.
- :func:`init` -- Initialize mDLAG model parameters to data prior to fitting.
- :func:`infer_latents` -- Infer latent variables.
- :func:`learn_gp_params` -- Learn Gaussian process parameters.
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

import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.optimize import minimize
from scipy.special import gammaln, psi
from scipy.stats import gmean

from latents.mdlag.data_types import mDLAGParams
from latents.observation_model.observations import ObsTimeSeries
from latents.observation_model.probabilistic import (
    HyperPriorParams,
    PosteriorARD,
    PosteriorLoading,
    PosteriorObsMean,
    PosteriorObsPrec,
)
from latents.state_model.gaussian_process import (
    GPParams,
    construct_gp_covariance_matrix,
)
from latents.state_model.latents import PosteriorLatentDelayed


def fit():
    """Fit a mDLAG model to data.

    Fit a delayed latents across multiple groups (mDLAG) model using an iterative
    variational inference scheme with mean-field approximation.
    """
    pass


def init(
    Y: ObsTimeSeries,
    gp_params_init: GPParams,
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
    gp_params_init
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

    # Seed the random number generator for reproducible initialization.
    rng = np.random.default_rng(random_seed)

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    num_groups = len(y_dims)  # Number of observed groups
    N = Y.data.shape[2]  # Number of samples
    T = Y.T  # Number of time point
    x_dim = gp_params_init.x_dim  # Number of latent dimensions

    # Get views of the observed data for each group
    Ys = Y.get_groups()
    # Get the variance of each observed group
    Y_covs = [np.cov(Y_m.reshape(Y_m.shape[0], -1)) for Y_m in Ys]

    # Initialize mDLAG parameter object
    params = mDLAGParams(x_dim, y_dims, T, gp_params_init, save_X_cov, save_C_cov)
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
    gp_params = params.gp_params

    x_dim = state_params.x_dim
    y_dims = obs_params.y_dims
    num_groups = len(obs_params.y_dims)
    T = params.T
    N = Y.data.shape[2]
    K_big = construct_gp_covariance_matrix(gp_params, T, return_tensor=False)

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


def GP_loss_per_latent(X_moment_j, N, D_non_zeros, eps, gamma, T):
    """Calculate the GP loss function for optimization.

    Args:
        X_moment_j: Moment of X for dimension j
        N: Number of trials
        D_non_zeros: Non-zero delay values
        eps: Epsilon parameter
        gamma: Gamma parameter
        T: Time points

    Returns
    -------
        float: Calculated loss value
    """
    num_groups = len(D_non_zeros) + 1
    D = np.zeros(num_groups)
    D[1:] = D_non_zeros

    Kj = np.zeros((num_groups, T, num_groups, T))
    for m1 in range(num_groups):
        for m2 in range(num_groups):
            t = np.arange(T)
            t2_minus_t1 = t[np.newaxis, :] - t[:, np.newaxis]
            diff = t2_minus_t1 - (D[m2] - D[m1])
            Kj[m1, :, m2, :] = (1 - eps) * np.exp(-0.5 * gamma * diff**2)

            if m1 == m2:
                Kj[m1, :, m2, :] += eps * np.eye(T)

    Kj_flat = Kj.reshape(num_groups * T, num_groups * T, order="F")
    # Define Loss using Kj
    f_gp = 0
    f_gp += -N / 2 * np.linalg.slogdet(Kj_flat)[1]
    Kj_inv = np.linalg.inv(Kj_flat)
    f_gp += -1 / 2 * np.trace(Kj_inv @ X_moment_j)
    return f_gp


def GP_loss_wrapper(x, X_moment_j, N, num_groups, T):
    """Optimize GP loss using wrapped parameters.

    Args:
        x: Parameter vector
        X_moment_j: Moment of X for dimension j
        N: Number of trials
        num_groups: Number of groups
        T: Time points

    Returns
    -------
        float: Loss value
    """
    # Unpack parameters from flattened array
    D_non_zeros = x[: num_groups - 1]  # First num_groups elements are delays
    gamma = x[num_groups - 1]  # Next element is gamma
    eps = x[num_groups]  # Last element is eps
    # Compute loss
    loss = GP_loss_per_latent(X_moment_j, N, D_non_zeros, eps, gamma, T)
    # Return negative loss since we want to minimize
    return -loss


def learn_gp_params(state_params, state_params_gp, obs_params):
    """Learn Gaussian process parameters given mDLAG model parameters and latents."""
    x_dim = state_params.x_dim
    num_groups = len(obs_params.y_dims)
    T = state_params.T
    N = state_params.X.mean.shape[-1]
    X_moment_GP = state_params.X.compute_moment_gp(in_place=False)

    D = np.array(state_params_gp.D, dtype=np.float64)
    gamma = np.array(state_params_gp.gamma, dtype=np.float64)
    eps = np.array(state_params_gp.eps, dtype=np.float64)

    l_gp = 0

    for j in range(x_dim):
        x0 = np.zeros(num_groups + 1)
        x0[: num_groups - 1] = D[1:, j]
        x0[num_groups - 1] = gamma[j]
        x0[num_groups] = eps[j]

        bounds = []
        bounds.extend([(-10, 10) for _ in range(num_groups - 1)])
        bounds.append((1e-6, 100))
        bounds.append((1e-6, 1e-2))

        result = minimize(
            GP_loss_wrapper,
            x0,
            args=(X_moment_GP[j], N, num_groups, T),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 10,
                "ftol": 1e-9,
                "gtol": 1e-7,
                "maxcor": 50,
            },
        )

        for i in range(num_groups - 1):
            D[i + 1, j] = result.x[i]

        gamma[j] = result.x[num_groups - 1]
        eps[j] = result.x[num_groups]
        l_gp += result.fun

    return D, gamma, eps, l_gp


def GP_loss(X, gp_params):
    """Compute the total GP loss across all latent dimensions.

    Parameters
    ----------
    X : PosteriorLatentDelayed
        Posterior estimates of time-delayed latent variables.
    gp_params : GPParams
        Gaussian process parameters.

    Returns
    -------
    float
        Total GP loss value.
    """
    # Compute loss:
    x_dim = X.mean.shape[0]
    N = X.mean.shape[3]
    T = X.mean.shape[2]
    f_gp = 0
    for j in range(x_dim):
        f_gp += GP_loss_per_latent(
            X.moment_gp[j],
            N,
            gp_params.D[1:, j],
            gp_params.eps[j],
            gp_params.gamma[j],
            T,
        )
    return f_gp


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
        gp_loss = GP_loss(params.state_params.X, params.gp_params)

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

    return lb_lik, lb_x, lb_C, lb_alpha, lb_phi, lb_d


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
    """Interface with, fit, and store the fitting results of a mDLAG model."""

    def __init__():
        pass

    def __repr__():
        pass

    def fit():
        """Fit a mDLAG model to data."""
        pass

    def init():
        """Initialize mDLAG model parameters."""
        pass

    def save():
        """Save a mDLAGModel object to a JSON file."""
        pass

    @staticmethod
    def load():
        """Load a mDLAGModel object from a JSON file."""
        pass

    def infer_latents():
        """Infer latent variables X given current params and observed data."""
        pass

    def infer_loadings():
        """Infer loadings C given current params and observed data."""
        pass

    def infer_ard():
        """Infer ARD parameters alpha given current params."""
        pass

    def infer_obs_mean():
        """Infer observation mean parameter given current params and observed data."""
        pass

    def infer_obs_prec():
        """Infer observation precision params given current params and observed data."""
        pass

    def compute_lower_bound():
        """Compute the variational lower bound given observed data."""
        pass
