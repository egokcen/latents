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
from scipy.stats import gmean

from latents.mdlag.data_types import mDLAGParams
from latents.observation_model.observations import ObsTimeSeries
from latents.observation_model.probabilistic import (
    HyperPriorParams,
    PosteriorARD,
    PosteriorLoading,
    PosteriorObsMean,
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
    params = mDLAGParams(x_dim, y_dims, T, gp_params_init)
    obs_params = params.obs_params

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
    CPhiC_diag = []
    k = 0
    for group_idx in range(num_groups):
        CmPhimCm = np.zeros((x_dim, x_dim))
        for i in range(y_dims[group_idx]):
            CmPhimCm += obs_params.phi.mean[k] * obs_params.C.moment[k, :, :]
            k += 1
        CPhiC_diag.append(CmPhimCm)

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
    C_means, _, _ = obs_params.C.get_groups(y_dims)
    CPhi = block_diag(*C_means).T @ np.diag(obs_params.phi.mean)
    CPhi_big = block_diag(*[CPhi] * T)
    Y_centered = Y.data - obs_params.d.mean[:, None, None]
    muX = SigX @ CPhi_big @ Y_centered.reshape(-1, N, order="F")
    X.mean[:] = np.reshape(muX, (x_dim, num_groups, T, N), order="F")

    # Compute moment
    X.compute_moment(in_place=True)
    X.compute_moment_gp(in_place=True)

    if not in_place:
        return X
    return None


def learn_gp_params():
    """Learn Gaussian process parameters given mDLAG model parameters and latents."""
    pass


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
        group_membership = np.repeat(np.arange(len(y_dims)), y_dims)
        XY = np.einsum(
            "xytn,ytn->yx", state_params.X.mean[:, group_membership, :, :], Y0
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
    phi_mean_expanded = obs_params.phi.mean[:, np.newaxis, np.newaxis]
    phi_C_cov = phi_mean_expanded * obs_params.C.cov
    C.mean[:] = np.einsum("ijk,ik->ij", phi_C_cov, XY)
    # Second moment
    C.compute_moment(in_place=True)

    if not in_place:
        return C
    return None


def infer_ard(
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
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

    _, _, C_moments = obs_params.C.get_groups(y_dims)

    # Rate parameters
    num_groups = len(y_dims)  # Number of observed groups
    for group_idx in range(num_groups):
        alpha.b[group_idx, :] = hyper_priors.b_alpha + 0.5 * np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    # Mean
    alpha.compute_mean(in_place=True)

    if not in_place:
        return alpha
    return None


def infer_obs_mean(
    Y: ObsTimeSeries,
    params: mDLAGParams,
    hyper_priors: HyperPriorParams,
    in_place: bool = True,
) -> PosteriorObsMean | None:
    """Infer observation mean parameter given current params and observed data.

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
    group_membership = np.repeat(np.arange(len(y_dims)), y_dims)
    for group_idx in range(num_groups):
        # Get indices for current group
        id_m = group_membership == group_idx

        residual = Y.data[id_m, :, :].reshape(
            y_dims[group_idx], -1
        ) - obs_params.C.mean[id_m, :] @ (
            state_params.X.mean[:, group_idx, :, :].reshape(x_dim, -1)
        )
        d.mean[id_m] = (
            d.cov[id_m] * obs_params.phi.mean[id_m] * np.sum(residual, axis=1)
        )

    if not in_place:
        return d
    return None


def infer_obs_prec():
    """Infer observation precision parameters given current params and observed data."""
    pass


def compute_lower_bound():
    """Compute the variational lower bound for a mDLAG model on observed data."""
    pass


def compute_lower_bound_constants():
    """Compute constant factors in the variational lower bound."""
    pass


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
