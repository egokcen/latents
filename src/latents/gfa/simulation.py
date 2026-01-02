"""Simulate data from the group factor analysis (GFA) generative model."""

from __future__ import annotations

import numpy as np

from latents.observation_model.observations import ObsStatic
from latents.observation_model.probabilistic import (
    ObsParamsARD,
    SimulationHyperPriors,
)


def simulate(
    n_samples: int,
    y_dims: np.ndarray,
    x_dim: int,
    hyper_priors: SimulationHyperPriors,
    snr: np.ndarray,
    random_seed: int | None = None,
) -> tuple[ObsStatic, np.ndarray, ObsParamsARD]:
    """Generate samples from the full group factor analysis model.

    Parameters
    ----------
    n_samples
        Number of data points to generate.
    y_dims
        `ndarray` of `int`, shape ``(n_groups,)``.
        Dimensionalities of each observed group.
    x_dim
        Number of latent dimensions.
    hyper_priors
        Simulation hyperparameters. The ``a_alpha`` and ``b_alpha`` arrays
        specify group- and column-specific sparsity patterns in the loading
        matrices. Use ``np.inf`` in ``a_alpha`` to force zero loadings.
    snr
        `ndarray` of `float`, shape ``(n_groups,)``.
        Signal-to-noise ratios of each group.
    random_seed
        Seed the random number generator for reproducible simulations.
        Defaults to ``None``, in which case the generated data will be
        different each run.

    Returns
    -------
    Y : ObsStatic
        Generated observed data.
    X : ndarray
        `ndarray` of `float`, shape ``(x_dim, n_samples)``.
        Latent data.
    obs_params : ObsParamsARD
        Generated GFA observation model parameters.
    """
    # Seed the random number generator for reproducibility
    rng = np.random.default_rng(random_seed)

    # Generate observation model parameters
    obs_params = ObsParamsARD.generate(y_dims, x_dim, hyper_priors, snr, rng)

    # Generate latent data
    X = generate_latents(n_samples, x_dim, rng)

    # Generate observated data
    Y = generate_observations(X, obs_params, rng)

    return Y, X, obs_params


def generate_latents(
    n_samples: int,
    x_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate latents via the GFA state model.

    Parameters
    ----------
    n_samples
        Number of data points to generate.
    x_dim
        Number of latent dimensions.
    rng
        A random number generator object.

    Returns
    -------
    ndarray
        `ndarray` of `float`, shape ``(x_dim, n_samples)``.
        Latent data.
    """
    return rng.normal(size=(x_dim, n_samples))


def generate_observations(
    X: np.ndarray,
    obs_params: ObsParamsARD,
    rng: np.random.Generator,
) -> ObsStatic:
    """
    Generate observed data via the GFA observation model, given latents and parameters.

    Parameters
    ----------
    X
        `ndarray` of `float`, shape ``(x_dim, n_samples)``.
        Latent data.
    obs_params
        GFA observation model parameters.
    rng
        A random number generator object.

    Returns
    -------
    ObsStatic
        Generated observed data.
    """
    # Number of data points
    n_samples = X.shape[1]
    # Dimensionality of each observed group
    y_dims = obs_params.y_dims
    # Number of observed groups
    n_groups = len(y_dims)

    # Split d, phi, and C according to observed groups
    ds, _ = obs_params.d.get_groups(y_dims)
    phis, _ = obs_params.phi.get_groups(y_dims)
    Cs, _, _ = obs_params.C.get_groups(y_dims)

    # Initialize observed data list
    Y = ObsStatic(data=np.zeros((y_dims.sum(), n_samples)), dims=y_dims)
    Ys = Y.get_groups()

    # Generate observated data group by group
    for group_idx in range(n_groups):
        Ys[group_idx][:] = (
            Cs[group_idx] @ X
            + ds[group_idx][:, np.newaxis]
            + rng.multivariate_normal(
                np.zeros(y_dims[group_idx]),
                np.diag(1 / phis[group_idx]),
                size=n_samples,
            ).T
        )

    return Y
