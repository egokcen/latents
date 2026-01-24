"""Compute descriptive statistics related to group factor analysis (GFA) models."""

from __future__ import annotations

import numpy as np

from latents.data import ObsStatic
from latents.gfa.inference import infer_latents
from latents.observation import ObsParamsPosterior


def predictive_performance(
    obs_data: ObsStatic,
    obs_posterior: ObsParamsPosterior,
    y_dims: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute the leave-group-out predictive performance of a GFA model.

    Parameters
    ----------
    obs_data : ObsStatic
        Observed data.
    obs_posterior : ObsParamsPosterior
        Fitted observation model posterior.
    y_dims : ndarray or None, default None
        Dimensionalities of each observed group. If None, inferred from obs_data.

    Returns
    -------
    R2 : float
        Leave-group-out R^2.
    MSE : float
        Leave-group-out mean squared error.

    Examples
    --------
    >>> from latents.gfa import GFAModel
    >>> from latents.gfa.analysis import predictive_performance
    >>> model = GFAModel()
    >>> model.fit(Y)
    >>> R2, MSE = predictive_performance(Y, model.obs_posterior)
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
    C_means, _, C_moments = obs_posterior.C.get_groups(y_dims)
    phi_means, _ = obs_posterior.phi.get_groups(y_dims)
    d_means, _ = obs_posterior.d.get_groups(y_dims)

    n_groups = len(y_dims)
    for group_idx in range(n_groups):
        # Group to be left out
        target_group = group_idx
        # Groups to be used for prediction
        source_groups = np.nonzero(np.arange(n_groups) != target_group)[0]

        # Construct a new posterior that excludes the target group
        source_posterior = ObsParamsPosterior(
            x_dim=obs_posterior.x_dim, y_dims=y_dims[source_groups]
        )
        source_posterior.C.mean = np.concatenate(
            [C_means[g] for g in source_groups], axis=0
        )
        source_posterior.C.moment = np.concatenate(
            [C_moments[g] for g in source_groups], axis=0
        )
        source_posterior.phi.mean = np.concatenate(
            [phi_means[g] for g in source_groups]
        )
        source_posterior.d.mean = np.concatenate([d_means[g] for g in source_groups])

        # Construct a new set of observed data that excludes the target group
        Y_source = ObsStatic(
            data=np.concatenate([Ys[g] for g in source_groups], axis=0),
            dims=y_dims[source_groups],
        )

        # Infer latent variables given the source groups
        X = infer_latents(Y_source, source_posterior)

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
