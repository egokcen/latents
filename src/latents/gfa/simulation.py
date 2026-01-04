"""Simulate data from the group factor analysis (GFA) generative model."""

from __future__ import annotations

import numpy as np

from latents.data import ObsStatic
from latents.observation import (
    ObsParamsHyperPrior,
    ObsParamsHyperPriorStructured,
    ObsParamsPoint,
    ObsParamsPrior,
    ObsParamsRealization,
    adjust_snr,
)
from latents.state import LatentsPriorStatic, LatentsRealization


def simulate(
    n_samples: int,
    y_dims: np.ndarray,
    x_dim: int,
    obs_hyperprior: ObsParamsHyperPrior | ObsParamsHyperPriorStructured,
    snr: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[ObsStatic, LatentsRealization, ObsParamsRealization]:
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
    obs_hyperprior
        Simulation hyperparameters. For structured sparsity patterns, use
        ``ObsParamsHyperPriorStructured`` where the ``a_alpha`` and ``b_alpha``
        arrays specify group- and column-specific patterns in the loading
        matrices. Use ``np.inf`` in ``a_alpha`` to force zero loadings.
    snr
        `ndarray` of `float`, shape ``(n_groups,)``.
        Signal-to-noise ratios of each group.
    rng
        NumPy random number generator for reproducible simulations.
        If None, a new default generator is created.

    Returns
    -------
    Y : ObsStatic
        Generated observed data.
    latents : LatentsRealization
        Sampled latent data.
    obs_params : ObsParamsRealization
        Generated GFA observation model parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample from the prior and adjust SNR
    prior = ObsParamsPrior(hyperprior=obs_hyperprior)
    obs_params = prior.sample(y_dims, x_dim, rng)
    obs_params = adjust_snr(obs_params, snr)

    # Sample latent data from the static prior
    latents_prior = LatentsPriorStatic()
    latents = latents_prior.sample(x_dim, n_samples, rng)

    # Generate observed data
    Y = sample_observations(latents, obs_params, rng)

    return Y, latents, obs_params


def sample_observations(
    latents: LatentsRealization,
    obs_params: ObsParamsRealization | ObsParamsPoint,
    rng: np.random.Generator,
) -> ObsStatic:
    """Generate observed data via the GFA observation model.

    Parameters
    ----------
    latents
        Sampled latent data.
    obs_params
        GFA observation model parameters, either as a full realization
        or as point estimates.
    rng
        NumPy random number generator.

    Returns
    -------
    ObsStatic
        Generated observed data.
    """
    # Number of data points
    n_samples = latents.n_samples
    # Dimensionality of each observed group
    y_dims = obs_params.y_dims
    # Number of observed groups
    n_groups = len(y_dims)

    # Split d, phi, and C according to observed groups
    y_boundaries = np.cumsum(y_dims)[:-1]
    ds = np.split(obs_params.d, y_boundaries)
    phis = np.split(obs_params.phi, y_boundaries)
    Cs = np.split(obs_params.C, y_boundaries, axis=0)

    # Initialize observed data list
    Y = ObsStatic(data=np.zeros((y_dims.sum(), n_samples)), dims=y_dims)
    Ys = Y.get_groups()

    # Generate observed data group by group
    for group_idx in range(n_groups):
        Ys[group_idx][:] = (
            Cs[group_idx] @ latents.X
            + ds[group_idx][:, np.newaxis]
            + rng.multivariate_normal(
                np.zeros(y_dims[group_idx]),
                np.diag(1 / phis[group_idx]),
                size=n_samples,
            ).T
        )

    return Y
