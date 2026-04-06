"""
Compute descriptive statistics related to mDLAG models.

**Functions**

- :func:`predictive_performance` -- Leave-group-out predictive performance.

"""

from __future__ import annotations

import numpy as np

from latents.mdlag.core import infer_latents
from latents.mdlag.data_types import mDLAGParams
from latents.observation_model.observations import ObsTimeSeries


def predictive_performance(
    obs_data: ObsTimeSeries,
    params: mDLAGParams,
    y_dims: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute the leave-group-out predictive performance of a mDLAG model.

    Parameters
    ----------
    obs_data
        Observed time series data.
    params
        mDLAG model parameters.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group. Defaults to ``None``, in which
        case the dimensionalities are inferred from ``obs_data``.

    Returns
    -------
    R2 : float
        Leave-group-out :math:`R^2`. Values close to 1 indicate good prediction,
        while negative values indicate prediction worse than the mean.
    MSE : float
        Leave-group-out mean squared error.
    """
    if y_dims is None:
        y_dims = obs_data.dims

    # Create a new view of the observed data matching y_dims
    Y = ObsTimeSeries(data=obs_data.data, dims=y_dims, T=obs_data.T)
    Ys = Y.get_groups()

    # Initialize predicted data and views of each group
    Y_pred = ObsTimeSeries(data=np.zeros_like(Y.data), dims=y_dims, T=Y.T)
    Ys_pred = Y_pred.get_groups()

    # Get views for relevant parameters into each group
    C_means, _, C_moments = params.obs_params.C.get_groups(y_dims)
    phi_means, _ = params.obs_params.phi.get_groups(y_dims)
    d_means, _ = params.obs_params.d.get_groups(y_dims)

    num_groups = len(y_dims)  # Number of observed groups

    for group_idx in range(num_groups):
        # Group to be left out
        target_group = group_idx
        # Groups to be used for inference
        source_groups = np.array([g for g in range(num_groups) if g != target_group])

        # Construct a new set of parameters that excludes the target group
        source_params = mDLAGParams(
            x_dim=params.obs_params.x_dim,
            y_dims=y_dims[source_groups],
            T=params.T,
        )

        # Copy observation model parameters for source groups
        source_params.obs_params.C.mean = np.concatenate(
            [C_means[g] for g in source_groups], axis=0
        )
        if C_moments[0] is not None:
            source_params.obs_params.C.moment = np.concatenate(
                [C_moments[g] for g in source_groups], axis=0
            )
        source_params.obs_params.phi.mean = np.concatenate(
            [phi_means[g] for g in source_groups]
        )
        source_params.obs_params.d.mean = np.concatenate(
            [d_means[g] for g in source_groups]
        )

        # Copy other observation parameters (keeping same structure)
        source_params.obs_params.alpha = params.obs_params.alpha.copy()

        # Copy and update state parameters for source groups
        source_params.state_params.num_groups = len(source_groups)
        source_params.state_params.T = params.T
        source_params.num_groups = len(source_groups)

        # Copy and update GP parameters for source groups
        source_params.gp = params.gp.copy()
        if hasattr(source_params.gp.params, "delays"):
            # Subset delays to only include source groups
            source_params.gp.params.delays = source_params.gp.params.delays[
                source_groups, :
            ]
            source_params.gp.params.num_groups = len(source_groups)

        # Construct observed data that excludes the target group
        Y_source = ObsTimeSeries(
            data=np.concatenate([Ys[g] for g in source_groups], axis=0),
            dims=y_dims[source_groups],
            T=Y.T,
        )

        # Infer latent variables given the source groups
        X = infer_latents(Y_source, source_params, in_place=False)

        X_target_mean = np.mean(X.mean, axis=1)  # Average across source groups

        # Predict target group observations: Y = C @ X + d
        # X_target_mean has shape (x_dim, T, N)
        # C_means[target_group] has shape (y_dim_target, x_dim)
        # d_means[target_group] has shape (y_dim_target,)

        for t in range(Y.T):
            for n in range(Y.data.shape[2]):
                Ys_pred[target_group][:, t, n] = (
                    C_means[target_group] @ X_target_mean[:, t, n]
                    + d_means[target_group]
                )

    # Compute aggregate performance metrics
    MSE = np.mean((Y.data - Y_pred.data) ** 2)

    # Compute R² with proper handling of time-series structure
    # For time-series data, we compute variance across all dimensions and time
    Y_mean = np.mean(Y.data, axis=(1, 2), keepdims=True)  # Mean across time and trials
    SS_res = np.sum((Y.data - Y_pred.data) ** 2)
    SS_tot = np.sum((Y.data - Y_mean) ** 2)

    R2 = 1 - SS_res / SS_tot

    return R2, MSE
