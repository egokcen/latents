"""
Simulate data from the mDLAG generative model.

**Functions**

- :func:`simulate` -- Generate samples from the full mDLAG model.
- :func:`generate_latents` -- Generate latents via the state model.
- :func:`generate_observations` -- Generate observations via the observation model.

"""

from __future__ import annotations

import numpy as np

from latents.mdlag.gp.gp_model import mDLAGGP
from latents.observation_model.observations import ObsTimeSeries
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD


def simulate(
    N: int,
    T: int,
    y_dims: np.ndarray,
    x_dim: int,
    hyper_priors: HyperPriorParams,
    snr: np.ndarray,
    gp_params: mDLAGGP | None = None,
    gamma_lim: tuple[float, float] | None = None,
    eps_lim: tuple[float, float] | None = None,
    delay_lim: tuple[float, float] | None = None,
    random_seed: int | None = None,
) -> tuple[ObsTimeSeries, np.ndarray, ObsParamsARD]:
    """
    Generate samples from the full mDLAG model.

    Parameters
    ----------
    N : int
        Number of sequences to generate.
    T : int
        Number of time points per sequence.
    y_dims : ndarray
        `ndarray` of `int`, shape ``(num_groups,)``.
        Number of observed dimensions in each group.
    x_dim : int
        Number of latent dimensions.
    hyper_priors : HyperPriorParams
        Hyperparameters of the mDLAG prior distributions.
        Note that ``hyper_priors.a_alpha`` and ``hyper_priors.b_alpha`` can be
        abused here to specify group- and column-specific sparsity patterns
        in the loadings matrices. In that case, specify both as `ndarray`
        of shape ``(num_groups, x_dim)``.
    snr : ndarray
        Signal-to-noise ratio for each group.
    gp_params : mDLAGGP | None, optional
        Parameters of the Gaussian Process. If None, will be generated.
    gamma_lim : tuple[float, float] | None, optional
        (min, max) limits for generating gamma parameters if gp_params is None.
    eps_lim : tuple[float, float] | None, optional
        (min, max) limits for generating epsilon parameters if gp_params is None.
    delay_lim : tuple[float, float] | None, optional
        (min, max) limits for generating delay parameters if gp_params is None.
    random_seed : int | None, optional
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
    # Generate observation model parameters
    obs_params = ObsParamsARD.generate(y_dims, x_dim, hyper_priors, snr, rng)

    # Generate GP parameters:
    num_groups = len(y_dims)
    if gp_params is None:
        gp_params = mDLAGGP.generate(
            x_dim, num_groups, delay_lim, eps_lim, gamma_lim, rng
        )

    # Generate latent variables
    X = generate_latents(gp_params, T, N, rng)

    # Generate observations

    Y = generate_observations(X, obs_params, rng)

    return Y, X, obs_params


def generate_latents(
    gp_params: mDLAGGP,
    T: int,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate latents via the mDLAG state model.

    Parameters
    ----------
    gp_params : mDLAGGP
        Parameters of the Gaussian Process.
    T : int
        Number of time points per sequence.
    N : int
        Number of sequences to generate.
    rng : np.random.Generator
        A random number generator object.

    Returns
    -------
    ndarray
        Generated latent data, shape ``(x_dim, num_groups, T, N)``.
        The dimensions represent:
        - x_dim: number of latent dimensions
        - num_groups: number of observation groups
        - T: number of time points per sequence
        - N: number of sequences
    """
    num_groups = gp_params.params.num_groups
    x_dim = gp_params.params.x_dim

    K = gp_params.build_kernel_matrix(T, return_tensor=False, order="F")
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
    X : ndarray
        Latent data, shape ``(x_dim, num_groups, T, N)``.
        The dimensions represent:
        - x_dim: number of latent dimensions
        - num_groups: number of observation groups
        - T: number of time points per sequence
        - N: number of sequences
    obs_params : ObsParamsARD
        mDLAG observation model parameters.
    rng : np.random.Generator
        A random number generator object.

    Returns
    -------
    ObsTimeSeries
        Generated observed data with dimensions matching the input parameters.
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
            + ds[group_idx][:, np.newaxis, np.newaxis]
            + rng.multivariate_normal(
                np.zeros(y_dims[group_idx]), np.diag(1 / phis[group_idx]), size=(T, N)
            ).transpose(2, 0, 1)
        )

    return Y
