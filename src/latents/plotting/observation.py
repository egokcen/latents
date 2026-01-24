"""Visualization functions for observation model results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_dimensionalities(
    num_dim: np.ndarray,
    dim_types: np.ndarray,
    sem_dim: np.ndarray | None = None,
    group_names: list[str] | None = None,
    plot_zero_dim: bool = False,
    ax: Axes | None = None,
) -> None:
    """Plot the number of each dimension type.

    Parameters
    ----------
    num_dim : ndarray of shape (n_dim_types,)
        Number of each dimension type.
    dim_types : ndarray of shape (n_groups, n_dim_types)
        Binary array indicating which groups are involved in each dimension type.
    sem_dim : ndarray or None, default None
        Standard error of the mean for each dimension type, shape (n_dim_types,).
    group_names : list of str or None, default None
        List of group names for labeling. If None, uses "1", "2", etc.
    plot_zero_dim : bool, default False
        Whether to plot dimension types with zero cardinality.
    ax : Axes or None, default None
        Axes on which to draw. If None, uses current axes.

    Examples
    --------
    >>> num_dim, _, _, dim_types = model.obs_posterior.compute_dimensionalities()
    >>> plot_dimensionalities(num_dim, dim_types)
    """
    ax = ax if ax is not None else plt.gca()

    n_groups, n_dim_types = dim_types.shape
    dim_cardinality = dim_types.sum(axis=0)

    if group_names is None:
        group_names = [f"{i + 1}" for i in range(n_groups)]
    xticklbls = ["" for i in range(n_dim_types)]
    for dim_idx in range(n_dim_types):
        if dim_cardinality[dim_idx] == 0:
            xticklbls[dim_idx] = "n.s."
        else:
            involved_groups = np.where(dim_types[:, dim_idx])[0]
            xticklbls[dim_idx] = "-".join([group_names[i] for i in involved_groups])

    sort_idxs = np.argsort(dim_cardinality)
    if not plot_zero_dim:
        sort_idxs = sort_idxs[dim_cardinality[sort_idxs] > 0]
        n_dim_types = len(sort_idxs)

    if sem_dim is None:
        ax.bar(np.arange(1, n_dim_types + 1), num_dim[sort_idxs])
    else:
        ax.bar(
            np.arange(1, n_dim_types + 1),
            num_dim[sort_idxs],
            yerr=sem_dim[sort_idxs],
        )
    ax.set_xlabel("Dimension type")
    ax.set_ylabel("Dimensionality")
    ax.set_xticks(np.arange(1, n_dim_types + 1))
    ax.set_xticklabels([xticklbls[i] for i in sort_idxs])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_var_exp(
    var_exp: np.ndarray,
    dim_types: np.ndarray,
    sem_var_exp: np.ndarray | None = None,
    group_names: list[str] | None = None,
    plot_zero_dim: bool = False,
    fig: Figure | None = None,
) -> None:
    """Plot shared variance explained by each dimension type.

    Parameters
    ----------
    var_exp : ndarray of shape (n_groups, n_dim_types)
        Fraction of shared variance explained by each dimension type in each group.
    dim_types : ndarray of shape (n_groups, n_dim_types)
        Binary array indicating which groups are involved in each dimension type.
    sem_var_exp : ndarray or None, default None
        Standard error of the mean for variance explained,
        shape (n_groups, n_dim_types).
    group_names : list of str or None, default None
        List of group names for labeling. If None, uses "1", "2", etc.
    plot_zero_dim : bool, default False
        Whether to plot dimension types with zero cardinality.
    fig : Figure or None, default None
        Figure on which to draw. If None, uses current figure.

    Examples
    --------
    >>> _, _, var_exp, dim_types = model.obs_posterior.compute_dimensionalities()
    >>> plot_var_exp(var_exp, dim_types)
    """
    fig = fig if fig is not None else plt.gcf()

    n_groups, n_dim_types = dim_types.shape
    dim_cardinality = dim_types.sum(axis=0)

    if group_names is None:
        group_names = [f"{i + 1}" for i in range(n_groups)]
    xticklbls = ["" for i in range(n_dim_types)]
    for dim_idx in range(n_dim_types):
        if dim_cardinality[dim_idx] == 0:
            xticklbls[dim_idx] = "n.s."
        else:
            involved_groups = np.where(dim_types[:, dim_idx])[0]
            xticklbls[dim_idx] = "-".join([group_names[i] for i in involved_groups])

    sort_idxs = np.argsort(dim_cardinality)
    if not plot_zero_dim:
        sort_idxs = sort_idxs[dim_cardinality[sort_idxs] > 0]
        n_dim_types = len(sort_idxs)

    for group_idx in range(n_groups):
        plt.subplot(n_groups, 1, group_idx + 1)
        if sem_var_exp is None:
            plt.bar(np.arange(1, n_dim_types + 1), var_exp[group_idx, sort_idxs])
        else:
            plt.bar(
                np.arange(1, n_dim_types + 1),
                var_exp[group_idx, sort_idxs],
                yerr=sem_var_exp[group_idx, sort_idxs],
            )
        plt.ylim([0, 1])
        plt.xlabel("Dimension type")
        plt.ylabel("Frac. shared var. exp.")
        plt.xticks(np.arange(1, n_dim_types + 1), [xticklbls[i] for i in sort_idxs])
        plt.title(f"Group {group_names[group_idx]}")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    fig.tight_layout()


def plot_dims_pairs(
    pair_dims: np.ndarray,
    pairs: np.ndarray,
    n_groups: int,
    sem_pair_dims: np.ndarray | None = None,
    group_names: list[str] | None = None,
    fig: Figure | None = None,
) -> None:
    """Visualize pairwise dimensionality analysis.

    Parameters
    ----------
    pair_dims : ndarray of shape (n_pairs, 3)
        Dimensionalities for each pair: [total_group1, shared, total_group2].
    pairs : ndarray of shape (n_pairs, 2)
        Indices of groups in each pair.
    n_groups : int
        Total number of groups.
    sem_pair_dims : ndarray or None, default None
        Standard error of the mean for pairwise dimensionalities, shape (n_pairs, 3).
    group_names : list of str or None, default None
        List of group names for labeling. If None, uses "1", "2", etc.
    fig : Figure or None, default None
        Figure on which to draw. If None, uses current figure.

    Examples
    --------
    >>> from latents.observation import ObsParamsPosterior
    >>> num_dim, _, var_exp, dim_types = obs_posterior.compute_dimensionalities()
    >>> pair_dims, _, pairs = ObsParamsPosterior.compute_dims_pairs(
    ...     num_dim, dim_types, var_exp
    ... )
    >>> plot_dims_pairs(pair_dims, pairs, n_groups=len(obs_posterior.y_dims))
    """
    fig = fig if fig is not None else plt.gcf()
    num_pairs = pairs.shape[0]

    if group_names is None:
        group_names = [f"{i + 1}" for i in range(n_groups)]
    xticklbls = np.full((num_pairs, 3), "", dtype=object)
    for pair_idx in range(num_pairs):
        xticklbls[pair_idx, 0] = f"Total, {group_names[pairs[pair_idx, 0]]}"
        xticklbls[pair_idx, 1] = (
            f"{group_names[pairs[pair_idx, 0]]}-{group_names[pairs[pair_idx, 1]]}"
        )
        xticklbls[pair_idx, 2] = f"Total, {group_names[pairs[pair_idx, 1]]}"

    for pair_idx in range(num_pairs):
        plt.subplot(1, num_pairs, pair_idx + 1)
        if sem_pair_dims is None:
            plt.bar(np.arange(1, pair_dims.shape[1] + 1), pair_dims[pair_idx, :])
        else:
            plt.bar(
                np.arange(1, pair_dims.shape[1] + 1),
                pair_dims[pair_idx, :],
                yerr=sem_pair_dims[pair_idx, :],
            )
        plt.xlabel("Dimension type")
        plt.ylabel("Dimensionality")
        plt.xticks(np.arange(1, pair_dims.shape[1] + 1), xticklbls[pair_idx, :])
        plt.title(xticklbls[pair_idx, 1])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    fig.tight_layout()


def plot_var_exp_pairs(
    pair_var_exp: np.ndarray,
    pairs: np.ndarray,
    n_groups: int,
    sem_pair_var_exp: np.ndarray | None = None,
    group_names: list[str] | None = None,
    fig: Figure | None = None,
) -> None:
    """Visualize pairwise shared variance analysis.

    Parameters
    ----------
    pair_var_exp : ndarray of shape (n_pairs, 2)
        Fraction of shared variance explained for each group in each pair.
    pairs : ndarray of shape (n_pairs, 2)
        Indices of groups in each pair.
    n_groups : int
        Total number of groups.
    sem_pair_var_exp : ndarray or None, default None
        Standard error of the mean for pairwise variance explained, shape (n_pairs, 2).
    group_names : list of str or None, default None
        List of group names for labeling. If None, uses "1", "2", etc.
    fig : Figure or None, default None
        Figure on which to draw. If None, uses current figure.

    Examples
    --------
    >>> from latents.observation import ObsParamsPosterior
    >>> num_dim, _, var_exp, dim_types = obs_posterior.compute_dimensionalities()
    >>> _, pair_var_exp, pairs = ObsParamsPosterior.compute_dims_pairs(
    ...     num_dim, dim_types, var_exp
    ... )
    >>> plot_var_exp_pairs(pair_var_exp, pairs, n_groups=len(obs_posterior.y_dims))
    """
    fig = fig if fig is not None else plt.gcf()
    num_pairs = pairs.shape[0]

    if group_names is None:
        group_names = [f"{i + 1}" for i in range(n_groups)]

    pairlbls = np.array(
        [
            f"{group_names[pairs[i, 0]]}" + f"-{group_names[pairs[i, 1]]}"
            for i in range(num_pairs)
        ]
    )

    for pair_idx in range(num_pairs):
        plt.subplot(1, num_pairs, pair_idx + 1)
        if sem_pair_var_exp is None:
            plt.bar(np.arange(1, pair_var_exp.shape[1] + 1), pair_var_exp[pair_idx, :])
        else:
            plt.bar(
                np.arange(1, pair_var_exp.shape[1] + 1),
                pair_var_exp[pair_idx, :],
                yerr=sem_pair_var_exp[pair_idx, :],
            )
        plt.ylim([0, 1])
        plt.xlabel("Group")
        plt.ylabel("Frac. shared var. exp.")
        plt.xticks(
            np.arange(1, pair_var_exp.shape[1] + 1),
            np.array(group_names)[pairs[pair_idx, :]],
        )
        plt.title(pairlbls[pair_idx])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    fig.tight_layout()
