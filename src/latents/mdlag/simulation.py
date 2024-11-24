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
from latents.observation_model.probabilistic import (
    HyperPriorParams,
    ObsParamsARD,
)
from latents.state_model.GP_latents import construct_K_mdlag_fast
from latents.state_model.latents import StateParamsGP


def simulate(
    N: int,
    y_dims: np.ndarray,
    hyper_priors: HyperPriorParams,
    state_params: StateParamsGP,
    snr: np.ndarray,
    random_seed: int | None = None,
) -> tuple[ObsTimeSeries, np.ndarray, ObsParamsARD]:
    """
    Generate samples from the full mDLAG model.

    Parameters
    ----------
    N
        Number of sequences to generate.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Number of observed dimensions in each group.
    hyper_priors
        Hyperparameters of the observation model priors.
    state_params
        Parameters of the state model.
    snr
        `ndarray` of `float`, shape ``(num_groups,)``.
        Signal-to-noise ratio of each group.
    random_seed
        Random seed for reproducibility.

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
    x_dim = state_params.x_dim

    # Generate observation model parameters
    obs_params = ObsParamsARD.generate(y_dims, x_dim, hyper_priors, snr, rng)

    # Generate latent variables
    latents = generate_latents(state_params, N, rng)
    X = np.transpose(latents, (1, 2, 3, 0))

    # Generate observations
    Y = generate_observations(X, obs_params, rng)

    return Y, X, obs_params


def generate_latents(
    state_params: StateParamsGP,
    N: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """
    Generate latents via the mDLAG state model.

    Parameters
    ----------
    state_params
        Parameters of the state model.
    N
        Number of sequences to generate.
    rng
        A random number generator object.

    Returns
    -------
    ndarray
        `ndarray` of `float`, shape ``(N, x_dim, num_groups, T)``.
        Latent data.

    Raises
    ------
    ValueError
        If the covariance matrix is not positive-definite.
    """
    num_groups = state_params.num_groups
    x_dim = state_params.x_dim
    T = state_params.T

    # Construct the covariance matrix K_big for sequences of length T
    K_big = construct_K_mdlag_fast(state_params.gp_params, return_matrix=True)

    # Perform Cholesky decomposition for sampling
    K_big_size = K_big.shape[0]
    try:
        # Adding a small value to the diagonal for numerical stability
        L = np.linalg.cholesky(K_big + 1e-6 * np.eye(K_big_size))
    except np.linalg.LinAlgError:
        msg = "Covariance matrix is not positive-definite."
        raise ValueError(msg)

    mean = np.zeros(K_big_size)
    latents = np.zeros((N, x_dim, num_groups, T))

    for n in range(N):
        # Sample latent variables
        u = rng.standard_normal(K_big_size)
        x_n = L @ u + mean
        latents[n, :, :, :] = x_n.reshape((x_dim, num_groups, T), order="F")

    return latents


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
