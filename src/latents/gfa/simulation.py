"""
Simulate data from the group factor analysis (GFA) generative model.

**Functions**

- :func:`simdata` -- Simulates data according to the GFA model.

"""

from __future__ import annotations

import numpy as np

from latents.gfa.data_types import (
    GFAParams,
    HyperPriorParams,
    ObsData,
)


def simdata(
    N: int,
    y_dims: np.ndarray,
    x_dim: int,
    hyper_priors: HyperPriorParams,
    snr: np.ndarray,
    random_seed: int | None = None,
) -> tuple[ObsData, np.ndarray, GFAParams]:
    """
    Generate simulated data according to a group factor analysis model.

    Parameters
    ----------
    N
        Number of data points to generate.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group.
    x_dim
        Number of latent dimensions.
    hyper_priors
        Hyperparameters of the GFA prior distributions.
        Note that ``hyper_priors.a_alpha`` and ``hyper_priors.b_alpha`` can be
        abused here, so that they can be used to specify group- and
        column-specific sparsity patterns in the loadings matrices.
        In that case, specify both of them as `ndarray` of shape
        ``(num_groups, x_dim)``.
    snr
        `ndarray` of `float`, shape ``(num_groups,)``.
        Signal-to-noise ratios of each group.
    random_seed
        Seed the random number generator for reproducible simulations.
        Defaults to ``None``, in which case the generated data will be
        different each run.

    Returns
    -------
    Y : ObsData
        Generated observed data.
    X : ndarray
        `ndarray` of `float`, shape ``(x_dim, N)``.
        Latent data.
    gfa_params : GFAParams
        Generated GFA model parameters.
    """
    # Seed the random number generator for reproducibility
    rng = np.random.default_rng(random_seed)

    # Number of observed groups
    num_groups = len(y_dims)

    # Initialize observed data list
    Y = ObsData(data=np.zeros((y_dims.sum(), N)), dims=y_dims)
    Ys = Y.get_groups()

    # Generate latent data
    X = rng.normal(size=(x_dim, N))

    # Initialize GFA parameter object
    gfa_params = GFAParams(x_dim=x_dim, y_dims=y_dims)

    # Initialize ARD parameters
    gfa_params.alpha.mean = np.zeros((num_groups, x_dim))
    if isinstance(hyper_priors.a_alpha, float):
        # Repeat the ARD hyperparameters for each group and column
        a_alpha = hyper_priors.a_alpha * np.ones((num_groups, x_dim))
        b_alpha = hyper_priors.b_alpha * np.ones((num_groups, x_dim))
    else:
        # Use the ARD hyperparameters specified by the user
        a_alpha = hyper_priors.a_alpha
        b_alpha = hyper_priors.b_alpha

    # Generate observation mean parameters
    gfa_params.d.mean = rng.normal(
        0, 1 / np.sqrt(hyper_priors.d_beta), size=(y_dims.sum())
    )
    # Split d according to observed groups, so we can use it below
    ds, _ = gfa_params.d.get_groups(y_dims)

    # Generate observation precision parameters
    gfa_params.phi.mean = rng.gamma(
        shape=hyper_priors.a_phi, scale=1 / hyper_priors.b_phi, size=(y_dims.sum())
    )
    # Split phi according to observed groups, so we can use it below
    phis, _ = gfa_params.phi.get_groups(y_dims)

    # Generate group-specific parameters and observed data
    gfa_params.C.mean = np.zeros((y_dims.sum(), x_dim))
    Cs, _, _ = gfa_params.C.get_groups(y_dims)
    for group_idx in range(num_groups):
        # Generate each ARD parameter and the corresponding column of the
        # loadings matrix for the current group
        for x_idx in range(x_dim):
            # Generate ARD parameters
            gfa_params.alpha.mean[group_idx, x_idx] = rng.gamma(
                shape=a_alpha[group_idx, x_idx], scale=1 / b_alpha[group_idx, x_idx]
            )
            Cs[group_idx][:, x_idx] = rng.normal(
                0,
                1 / np.sqrt(gfa_params.alpha.mean[group_idx, x_idx]),
                size=(y_dims[group_idx]),
            )

        # Enforce the desired signal-to-noise ratios
        var_CC = np.sum(Cs[group_idx] ** 2)
        var_noise_desired = var_CC / snr[group_idx]
        var_noise_current = np.sum(1 / phis[group_idx])
        phis[group_idx] *= var_noise_current / var_noise_desired

        # Generate observated data
        Ys[group_idx][:] = (
            Cs[group_idx] @ X
            + ds[group_idx][:, np.newaxis]
            + rng.multivariate_normal(
                np.zeros(y_dims[group_idx]), np.diag(1 / phis[group_idx]), size=N
            ).T
        )

    return Y, X, gfa_params
