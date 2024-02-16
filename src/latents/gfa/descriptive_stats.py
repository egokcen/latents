"""
Compute descriptive statistics related to group factor analysis (GFA) models.

**Functions**

- :func:`compute_snr` -- Compute the signal-to-noise ratio (SNR) of each group.
- :func:`compute_dimensionalities` -- Compute dimensionalities of each type.
- :func:`get_dim_types` -- Determine dimension types for a number of groups.
- :func:`compute_dims_pairs` -- Shared dimensionalities between pairs of groups.
- :func:`predictive_performance` -- Leave-group-out predictive performance.

"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from latents.gfa.core import infer_latents
from latents.gfa.data_types import (
    GFAParams,
    ObsData,
    PosteriorLoading,
    PosteriorObsPrec,
)


def compute_snr(
    C: PosteriorLoading,
    phi: PosteriorObsPrec,
    dims: np.ndarray,
) -> np.ndarray:
    """
    Compute the signal-to-noise ratio (SNR) of each observed group.

    Compute the SNR of each observed group, according to GFA model parameters.

    Parameters
    ----------
    C
        Posterior estimate of the loadings matrix.
    phi
        Posterior estimate of the observation precision parameters.
    dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group.

    Returns
    -------
    ndarray
        `ndarray` of `float`, shape ``(num_groups,)``.
        Signal-to-noise ratio of each observed group.
    """
    # Get views of the loadings matrix and the observation precisions for
    # each group
    _, _, C_moments = C.get_groups(dims)
    phi_means, _ = phi.get_groups(dims)

    # Compute the SNR for each group
    return np.array(
        [
            np.trace(np.sum(C_moments[group_idx], axis=0))
            / np.sum(1 / phi_means[group_idx])
            for group_idx in range(len(dims))
        ]
    )


def compute_dimensionalities(
    params: GFAParams,
    cutoff_shared_var: float = 0.02,
    cutoff_snr: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dimensionalities.

    Compute the number of each possible type of dimension, along with the
    shared variance in each group explained by each type of dimension.

    Parameters
    ----------
    params
        GFA model parameters.
    cutoff_shared_var
        Minimum fraction of shared variance within a group that must be
        explained by a latent to be considered significant. Defaults to
        ``0.02``.
    cutoff_snr
        Minimum signal-to-noise ratio (SNR) that a group must have for ANY
        latents to be considered significant. Defaults to ``0.001``.

    Returns
    -------
    num_dim : ndarray
        `ndarray` of `int`, shape ``(num_dim_types,)``.
        The number of each type of dimension. ``num_dim[i]`` corresponds to
        the dimension type in ``dim_types[:,i]``.
    sig_dims : ndarray
        `ndarray` of `bool`, shape ``(num_groups, x_dim)``.
        ``sig_dims[i,j]`` is ``True`` if latent ``j`` explains a significant
        fraction of the shared variance within group ``i``.
    var_exp : ndarray
        `ndarray` of `float`, shape ``(num_groups, num_dim_types)``.
        ``var_exp[i,j]`` is the fraction of the shared variance within group
        ``i`` that is explained by dimension type ``j``. ``var_exp[:,j]``
        corresponds to the dimension type in ``dim_types[:,j]``.
    dim_types : ndarray
        `ndarray` of `bool`, shape ``(num_groups, num_dim_types)``.
        ``dim_types[:,j]`` is a Boolean vector indicating the structure of
        dimension type ``j``. ``1`` indicates that a group is involved, ``0``
        otherwise.
    """
    num_groups = len(params.y_dims)  # Number of observed groups

    # Determine all dimension types
    dim_types = get_dim_types(num_groups)
    num_dim_types = dim_types.shape[1]

    # Compute signal-to-noise ratios
    snr = compute_snr(params.C, params.phi, params.y_dims)

    # Relative shared variance explained by each dimension
    alpha_inv = 1 / params.alpha.mean
    alpha_inv_rel = alpha_inv / np.sum(alpha_inv, axis=1, keepdims=True)

    # Take dimensions only if shared variance and SNR are above cutoffs
    sig_dims = (alpha_inv_rel > cutoff_shared_var) & (snr > cutoff_snr)[:, np.newaxis]
    num_dim = np.zeros(num_dim_types)
    var_exp = np.zeros((num_groups, num_dim_types))
    for dim_idx in range(num_dim_types):
        dims = np.all(sig_dims == dim_types[:, dim_idx, np.newaxis], axis=0)
        num_dim[dim_idx] = np.sum(dims)
        var_exp[:, dim_idx] = np.sum(alpha_inv_rel[:, dims], axis=1)

    return num_dim, sig_dims, var_exp, dim_types


def get_dim_types(num_groups: int) -> np.ndarray:
    """
    Generate all dimension types for a given number of groups.

    Generate an array with all types of dimensions (singlet, doublet, triplet, global,
    etc.) for a given number of groups.

    Parameters
    ----------
    num_groups
        Number of observed groups.

    Returns
    -------
    dim_types : ndarray
        `ndarray` of `bool`, shape ``(num_groups, num_dim_types)``.
        ``dim_types[:,j]`` is a Boolean vector indicating the structure of
        dimension type ``j``. ``1`` indicates that a group is involved, ``0``
        otherwise.
    """
    num_dim_types = 2**num_groups  # Number of dimension types
    dim_types = np.empty((num_groups, num_dim_types))

    for dim_idx in range(num_dim_types):
        dim_str = format(dim_idx, f"0{num_groups}b")
        dim_types[:, dim_idx] = np.array([int(b) for b in dim_str], dtype=bool)

    return dim_types


def compute_dims_pairs(
    num_dim: np.ndarray,
    dim_types: np.ndarray,
    var_exp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze the shared dimensionalities and variances between pairs of groups.

    Compute the total dimensionality of each group, and the shared
    dimensionality between each pair of groups given the dimensionalities and
    types of dimensions given by ``num_dim`` and ``dim_types``, respectively.
    Compute also the shared variance explained by a pairwise interaction in
    each group.

    Parameters
    ----------
    num_dim
        `ndarray` of `int`, shape ``(num_dim_types,)``.
        The number of each type of dimension. ``num_dim[i]`` corresponds to
        the dimension type in ``dim_types[:,i]``.
    dim_types
        `ndarray` of `bool`, shape ``(num_groups, num_dim_types)``.
        ``dim_types[:,j]`` is a Boolean vector indicating the structure of
        dimension type ``j``. ``1`` indicates that a group is involved, ``0``
        otherwise.
    var_exp
        `ndarray` of `float`, shape ``(num_groups, num_dim_types)``.
        ``var_exp[i,j]`` is the fraction of the shared variance within group
        ``i`` that is explained by dimension type ``j``. ``var_exp[:,j]``
        corresponds to the dimension type in ``dim_types[:,j]``.

    Returns
    -------
    pair_dims : ndarray
        `ndarray` of `int`, shape ``(num_pairs, 3)``.

        ``pair_dims[i,0]`` -- total dimensionality of group ``1`` in pair ``i``.

        ``pair_dims[i,1]`` -- shared dimensionality between pair ``i``.

        ``pair_dims[i,2]`` -- total dimensionality of group ``2`` in pair ``i``.
    pair_var_exp : ndarray
        `ndarray` of `float`, shape ``(num_pairs, 2)``.

        ``pair_var_exp[i,0]`` -- shared variance explained by pairwise
        interaction ``i`` in group ``1``.

        ``pair_var_exp[i,1]`` -- shared variance explained by pairwise
        interaction ``i`` in group ``2``.
    pairs : ndarray
        `ndarray` of `int`, shape ``(num_pairs, 2)``.

        ``pairs[i,0]`` -- index of group ``1`` in pair ``i``.

        ``pairs[i,1]`` -- index of group ``2`` in pair ``i``.
    """
    num_groups = dim_types.shape[0]  # Number of observed groups

    # For each group, create a list of all dimension types that involve that
    # group
    group_idxs = [np.nonzero(dim_types[g, :])[0] for g in range(num_groups)]

    # Create a list of all possible pairs
    pairs = list(combinations(range(num_groups), 2))
    num_pairs = len(pairs)

    # Count the number of each type of dimension
    pair_dims = np.zeros((num_pairs, 3), dtype=int)
    pair_var_exp = np.zeros((num_pairs, 2))
    for pair_idx, pair in enumerate(pairs):
        # Total dimensionality of each group in the pair
        pair_dims[pair_idx, 0] = num_dim[group_idxs[pair[0]]].sum()
        pair_dims[pair_idx, 2] = num_dim[group_idxs[pair[1]]].sum()

        # Shared dimensionality between the two groups
        shared_idxs = np.intersect1d(group_idxs[pair[0]], group_idxs[pair[1]])
        pair_dims[pair_idx, 1] = num_dim[shared_idxs].sum()

        # Shared variance explained by a pairwise interaction in each group
        pair_var_exp[pair_idx, 0] = var_exp[pair[0], shared_idxs].sum()
        pair_var_exp[pair_idx, 1] = var_exp[pair[1], shared_idxs].sum()

    return pair_dims, pair_var_exp, np.array(pairs)


def predictive_performance(
    obs_data: ObsData,
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
        `ndarray` of `int`, shape ``(num_groups,)``.
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
    Y = ObsData(data=obs_data.data, dims=y_dims)
    Ys = Y.get_groups()

    # Initialize predicted data and views of each group
    Y_pred = ObsData(data=np.zeros_like(Y.data), dims=y_dims)
    Ys_pred = Y_pred.get_groups()

    # Get views for relevant parameters into each group
    C_means, _, C_moments = params.C.get_groups(y_dims)
    phi_means, _ = params.phi.get_groups(y_dims)
    d_means, _ = params.d.get_groups(y_dims)

    num_groups = len(y_dims)  # Number of observed groups
    for group_idx in range(num_groups):
        # Group to be left out
        target_group = group_idx
        # Groups to be used for prediction
        source_groups = np.nonzero(np.arange(num_groups) != target_group)[0]

        # Construct a new set of parameters that excludes the target group
        source_params = GFAParams(x_dim=params.x_dim, y_dims=y_dims[source_groups])
        source_params.C.mean = np.concatenate(
            [C_means[g] for g in source_groups], axis=0
        )
        source_params.C.moment = np.concatenate(
            [C_moments[g] for g in source_groups], axis=0
        )
        source_params.phi.mean = np.concatenate([phi_means[g] for g in source_groups])
        source_params.d.mean = np.concatenate([d_means[g] for g in source_groups])

        # Construct a new set of observed data that excludes the target group
        Y_source = ObsData(
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
