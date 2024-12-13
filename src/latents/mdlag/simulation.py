"""
Simulate data from the mDLAG generative model.

**Functions**

- :func:`simulate` -- Generate samples from the full mDLAG model.
- :func:`generate_latents` -- Generate latents via the state model.
- :func:`generate_observations` -- Generate observations via the observation model.

"""

from __future__ import annotations

import numpy as np

from latents.observation_model.observations import ObsTimeSeries
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD
from latents.state_model.gaussian_process import (
    GPParams,
    construct_gp_covariance_matrix,
)


def simulate(
    N: int,
    T: int,
    y_dims: np.ndarray,
    x_dim: int,
    hyper_priors: HyperPriorParams,
    snr: np.ndarray,
    gp_params: GPParams | None = None,
    gamma_lim: tuple[float, float] | None = None,
    eps_lim: tuple[float, float] | None = None,
    delay_lim: tuple[float, float] | None = None,
    random_seed: int | None = None,
) -> tuple[ObsTimeSeries, np.ndarray, ObsParamsARD]:
    """
    Generate samples from the full mDLAG model.

    Parameters
    ----------
    N
        Number of sequences to generate.
    T
        Number of time points per sequence.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Number of observed dimensions in each group.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions.
        Note that ``hyper_priors.a_alpha`` and ``hyper_priors.b_alpha`` can be
        abused here, so that they can be used to specify group- and
        column-specific sparsity patterns in the loadings matrices.
        In that case, specify both of them as `ndarray` of shape
        ``(num_groups, x_dim)``.


    Returns
    -------
    ObsTimeSeries
        Generated observed data.
    ndarray
        Generated latent data, shape ``(x_dim, num_groups, T, N)``.
    ObsParamsARD
        Generated observation model parameters.
    """
    rng = np.random.default_rng(seed=random_seed)
    # Generate observation model parameters
    obs_params = ObsParamsARD.generate(y_dims, x_dim, hyper_priors, snr, rng)

    # Generate GP parameters:
    num_groups = len(y_dims)
    if gp_params is None:
        gp_params = GPParams.generate(
            x_dim, num_groups, gamma_lim, eps_lim, delay_lim, rng
        )

    # Generate latent variables
    X = generate_latents(gp_params, T, N, rng)

    # Generate observations

    Y = generate_observations(X, obs_params, rng)

    return Y, X, obs_params


def generate_latents(
    gp_params: GPParams,
    T: int,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate latents via the mDLAG state model.

    Parameters
    ----------
    gp_params
        Parameters of the gp
    T
        Number of time points per sequence.
    N
        Number of sequences to generate.
    rng
        A random number generator object.

    Returns
    -------
    ndarray
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, N)``.
        Latent data.

    Raises
    ------
    ValueError
        If the covariance matrix is not positive-definite.
    """
    num_groups = gp_params.num_groups
    x_dim = gp_params.x_dim

    K = construct_gp_covariance_matrix(gp_params, T, return_tensor=False, order="F")
    latents = rng.multivariate_normal(np.zeros(K.shape[0]), K, size=N)
    latents = latents.reshape((N, x_dim, num_groups, T), order="F")

    return latents.transpose(1, 2, 3, 0)


def generate_observations(
    X: np.ndarray,
    obs_params: ObsParamsARD,
    rng: np.random.Generator,
) -> ObsTimeSeries:
    """
    Generate observations via the mDLAG observation model.

    Parameters
    ----------
    X
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, N)``.
        Latent data.
    obs_params
        mDLAG observation model parameters.
    rng
        A random number generator object.

    Returns
    -------
    ObsTimeSeries
        Generated observed data.
    """
    # Number of data points
    N = X.shape[-1]
    # Dimensionality of each observed group
    y_dims = obs_params.y_dims
    # Number of observed groups
    num_groups = len(y_dims)
    # Number of time points
    T = X.shape[2]

    # Split d, phi, and C according to observed groups
    ds, _ = obs_params.d.get_groups(y_dims)
    phis, _ = obs_params.phi.get_groups(y_dims)
    Cs, _, _ = obs_params.C.get_groups(y_dims)

    # Initialize observed data list
    Y = ObsTimeSeries(data=np.zeros((y_dims.sum(), T, N)), dims=y_dims, T=T)
    Ys = Y.get_groups()

    # Generate observated data group by group
    for group_idx in range(num_groups):
        Ys[group_idx][:] = (
            np.einsum("ij,j...->i...", Cs[group_idx], X[:, group_idx, :, :])
            + ds[group_idx][:, None, None]
            + rng.multivariate_normal(
                np.zeros(y_dims[group_idx]), np.diag(1 / phis[group_idx]), size=(T, N)
            ).transpose(2, 0, 1)
        )

    return Y
