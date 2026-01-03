"""Compute descriptive statistics related to group factor analysis (GFA) models."""

from __future__ import annotations

import numpy as np

from latents.data import ObsStatic
from latents.gfa.core import infer_latents
from latents.gfa.data_types import GFAParams


def predictive_performance(
    obs_data: ObsStatic,
    params: GFAParams,
    y_dims: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute the leave-group-out predictive performance of a GFA model.

    Parameters
    ----------
    obs_data
        Observed data.
    params
        GFA model parameters.
    y_dims
        `ndarray` of `int`, shape ``(n_groups,)``.
        Dimensionalities of each observed group. Defaults to ``None``, in which
        case the dimensionalities are inferred from ``obs_data``.

    Returns
    -------
    R2 : float
        Leave-group-out :math:`R^2`.
    MSE : float
        Leave-group-out mean squared error.
    """
    if y_dims is None:
        y_dims = obs_data.dims
    # Create a new view of the observed data in which the groups match y_dims
    Y = ObsStatic(data=obs_data.data, dims=y_dims)
    Ys = Y.get_groups()

    # Initialize predicted data and views of each group
    Y_pred = ObsStatic(data=np.zeros_like(Y.data), dims=y_dims)
    Ys_pred = Y_pred.get_groups()

    # Get views for relevant parameters into each group
    C_means, _, C_moments = params.obs_params.C.get_groups(y_dims)
    phi_means, _ = params.obs_params.phi.get_groups(y_dims)
    d_means, _ = params.obs_params.d.get_groups(y_dims)

    n_groups = len(y_dims)  # Number of observed groups
    for group_idx in range(n_groups):
        # Group to be left out
        target_group = group_idx
        # Groups to be used for prediction
        source_groups = np.nonzero(np.arange(n_groups) != target_group)[0]

        # Construct a new set of parameters that excludes the target group
        source_params = GFAParams(
            x_dim=params.obs_params.x_dim, y_dims=y_dims[source_groups]
        )
        source_params.obs_params.C.mean = np.concatenate(
            [C_means[g] for g in source_groups], axis=0
        )
        source_params.obs_params.C.moment = np.concatenate(
            [C_moments[g] for g in source_groups], axis=0
        )
        source_params.obs_params.phi.mean = np.concatenate(
            [phi_means[g] for g in source_groups]
        )
        source_params.obs_params.d.mean = np.concatenate(
            [d_means[g] for g in source_groups]
        )

        # Construct a new set of observed data that excludes the target group
        Y_source = ObsStatic(
            data=np.concatenate([Ys[g] for g in source_groups], axis=0),
            dims=y_dims[source_groups],
        )

        # Infer latent variables given the source groups
        X = infer_latents(Y_source, source_params, in_place=False)

        # Predict the target group
        Ys_pred[target_group][:] = (
            C_means[target_group] @ X.mean + d_means[target_group][:, np.newaxis]
        )

    # Compute aggregate performance metrics
    MSE = np.mean((Y.data - Y_pred.data) ** 2)
    R2 = 1 - np.sum((Y.data - Y_pred.data) ** 2) / np.sum(
        (Y.data - np.mean(Y.data, axis=1, keepdims=True)) ** 2
    )

    return R2, MSE
