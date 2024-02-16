"""
Plot summaries of group factor analysis (GFA) results.

**Functions**

- :func:`hinton` -- Draw a Hinton diagram of a matrix.
- :func:`plot_dimensionalities` -- Plot the number of dimensions of each type.
- :func:`plot_var_exp` -- Plot shared variance explained by each dimension type.
- :func:`plot_dims_pairs` -- Visualize pairwise analyses of dimensionality.
- :func:`plot_var_exp_pairs` -- Visualize pairwise analyses of shared variance.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def hinton(
    matrix: np.ndarray,
    max_weight: float | None = None,
    ax: Axes | None = None,
) -> None:
    """
    Draw a Hinton diagram of a matrix.

    The color of each square reflects the sign of the corresponding matrix
    element, and the size of each square reflects the magnitude of the
    corresponding element.

    Code adapted from `here
    <https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html>`_.

    Parameters
    ----------
    matrix
        `ndarray` of `float`, shape ``(M, N)``.
        Matrix to visualize.
    max_weight
        Maximum absolute value of matrix elements.
    ax
        Axes on which to draw the diagram. If ``None``, then gets an existing
        axis or creates a new one.

    Example
    -------
    >>> import numpy as np
    >>> from gfa_py.plotting_gfa import hinton
    >>> C = np.random.normal(size=(10, 5))
    >>> hinton(C)
    """
    # If no axis is provided, then get an existing one or create a new one
    ax = ax if ax is not None else plt.gca()

    # Set the default maximum weight if not provided
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    # Set up the figure background
    ax.patch.set_facecolor("white")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Draw each element of the matrix
    if matrix.ndim == 1:
        # For 1D arrays, add a second dimension to create a column vector
        matrix = matrix[:, np.newaxis]
    for (y, x), w in np.ndenumerate(matrix):
        # Color code for positive and negative values
        rect_color = "red" if w > 0 else "blue"
        # Size of each rectangle. Scale down slightly to ensure that there's
        # always some white space between elements.
        rect_size = 0.9 * np.sqrt(np.abs(w) / max_weight)
        # Plot each rectangle
        rect = plt.Rectangle(
            (x - rect_size / 2, y - rect_size / 2),
            rect_size,
            rect_size,
            facecolor=rect_color,
            edgecolor=None,
        )
        ax.add_patch(rect)

    # Make sure we can see all elements of the matrix
    ax.autoscale_view()
    # The matrix will be plotted upside down by default, so flip it
    ax.invert_yaxis()


def plot_dimensionalities(
    num_dim: np.ndarray,
    dim_types: np.ndarray,
    sem_dim: np.ndarray | None = None,
    group_names: list[str] | None = None,
    plot_zero_dim: bool = False,
    ax: Axes | None = None,
) -> None:
    """
    Plot the number of each type of dimension (given by ``dim_types``) in ``num_dim``.

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
    sem_dim
        `ndarray` of `float`, shape ``(num_dim_types,)``.
        Standard error of the number of dimensions of each type. Defaults
        to ``None``.
    group_names
        Names of the groups. Defaults to ``None``.
    plot_zero_dim
        Set ``True`` to plot the number of dimensions that are not significant
        in any group. Defaults to ``False``.
    ax
        Axes on which to draw the diagram. If ``None``, then gets an existing
        axis or creates a new one.
    """
    # If no axis is provided, get an existing one or create a new one
    ax = ax if ax is not None else plt.gca()

    # Determine the number of groups involved in each dimension type
    num_groups, num_dim_types = dim_types.shape
    dim_cardinality = dim_types.sum(axis=0)

    # Set up labels for the x-axis
    if group_names is None:
        group_names = [f"{i+1}" for i in range(num_groups)]
    xticklbls = ["" for i in range(num_dim_types)]
    for dim_idx in range(num_dim_types):
        if dim_cardinality[dim_idx] == 0:
            xticklbls[dim_idx] = "n.s."  # Not significant
        else:
            involved_groups = np.where(dim_types[:, dim_idx])[0]
            xticklbls[dim_idx] = "-".join([group_names[i] for i in involved_groups])

    # Sort dimension types by increasing number of involved groups
    sort_idxs = np.argsort(dim_cardinality)
    if not plot_zero_dim:
        # Remove dimension types that are not significant in any group
        sort_idxs = sort_idxs[dim_cardinality[sort_idxs] > 0]
        num_dim_types = len(sort_idxs)

    # Plot dimensionalities
    if sem_dim is None:
        ax.bar(np.arange(1, num_dim_types + 1), num_dim[sort_idxs])
    else:
        ax.bar(
            np.arange(1, num_dim_types + 1), num_dim[sort_idxs], yerr=sem_dim[sort_idxs]
        )
    ax.set_xlabel("Dimension type")
    ax.set_ylabel("Dimensionality")
    ax.set_xticks(np.arange(1, num_dim_types + 1))
    ax.set_xticklabels([xticklbls[i] for i in sort_idxs])

    # Adjust appearance of axes
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
    """
    Plot the shared variance explained by each type of dimension in each group.

    Parameters
    ----------
    var_exp
        `ndarray` of `float`, shape ``(num_groups, num_dim_types)``.
        ``var_exp[i,j]`` is the fraction of the shared variance within group
        ``i`` that is explained by dimension type ``j``. ``var_exp[:,j]``
        corresponds to the dimension type in ``dim_types[:,j]``.
    dim_types
        `ndarray` of `bool`, shape ``(num_groups, num_dim_types)``.
        ``dim_types[:,j]`` is a Boolean vector indicating the structure of
        dimension type ``j``. ``1`` indicates that a group is involved, ``0``
        otherwise.
    sem_var_exp
        `ndarray` of `float`, shape ``(num_groups, num_dim_types)``.
        Standard error of ``var_exp``. Defaults to ``None``.
    group_names
        Names of the groups. Defaults to ``None``.
    plot_zero_dim
        Set ``True`` to plot shared variance for dimensions that are not
        significant in any group. Defaults to ``False``.
    fig
        Figure on which to draw the diagram. If ``None``, then gets an existing
        figure or creates a new one.
    """
    # If no figure is provided, then get an existing one or create a new one
    fig = fig if fig is not None else plt.gcf()

    # Determine the number of groups involved in each dimension type
    num_groups, num_dim_types = dim_types.shape
    dim_cardinality = dim_types.sum(axis=0)

    # Set up labels for the x-axis
    if group_names is None:
        group_names = [f"{i+1}" for i in range(num_groups)]
    xticklbls = ["" for i in range(num_dim_types)]
    for dim_idx in range(num_dim_types):
        if dim_cardinality[dim_idx] == 0:
            xticklbls[dim_idx] = "n.s."  # Not significant
        else:
            involved_groups = np.where(dim_types[:, dim_idx])[0]
            xticklbls[dim_idx] = "-".join([group_names[i] for i in involved_groups])

    # Sort dimension types by increasing number of involved groups
    sort_idxs = np.argsort(dim_cardinality)
    if not plot_zero_dim:
        # Remove dimension types that are not significant in any group
        sort_idxs = sort_idxs[dim_cardinality[sort_idxs] > 0]
        num_dim_types = len(sort_idxs)

    # Plot shared variance explained by each dimension type in each group
    for group_idx in range(num_groups):
        plt.subplot(num_groups, 1, group_idx + 1)
        if sem_var_exp is None:
            plt.bar(np.arange(1, num_dim_types + 1), var_exp[group_idx, sort_idxs])
        else:
            plt.bar(
                np.arange(1, num_dim_types + 1),
                var_exp[group_idx, sort_idxs],
                yerr=sem_var_exp[group_idx, sort_idxs],
            )
        plt.ylim([0, 1])
        plt.xlabel("Dimension type")
        plt.ylabel("Frac. shared var. exp.")
        plt.xticks(np.arange(1, num_dim_types + 1), [xticklbls[i] for i in sort_idxs])
        plt.title(f"Group {group_names[group_idx]}")

        # Adjust appearance of axes
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    # Make sure the subplots don't overlap
    fig.tight_layout()


def plot_dims_pairs(
    pair_dims: np.ndarray,
    pairs: np.ndarray,
    num_groups: int,
    sem_pair_dims: np.ndarray | None = None,
    group_names: list[str] | None = None,
    fig: Figure | None = None,
) -> None:
    """
    Visualize pairwise analyses of dimensionality.

    Visualize pairwise analyses of dimensionality: the total dimensionality
    of each group in each pair, and the shared dimensionality of each pair.

    Parameters
    ----------
    pair_dims
        `ndarray` of `int`, shape ``(num_pairs, 3)``.
        ``pair_dims[i,0]`` -- total dimensionality of group ``1`` in pair ``i``.
        ``pair_dims[i,1]`` -- shared dimensionality between pair ``i``.
        ``pair_dims[i,2]`` -- total dimensionality of group ``2`` in pair ``i``.
    pairs
        `ndarray` of `int`, shape ``(num_pairs, 2)``.
        ``pairs[i,0]`` -- index of group ``1`` in pair ``i``.
        ``pairs[i,1]`` -- index of group ``2`` in pair ``i``.
    num_groups
        Number of observed groups.
    sem_pair_dims
        `ndarray` of `float`, shape ``(num_pairs, 3)``.
        Standard error of ``pair_dims``. Defaults to ``None``.
    group_names
        Names of the groups. Defaults to ``None``.
    fig
        Figure on which to draw the diagram. If ``None``, then gets an existing
        figure or creates a new one.
    """
    # If no figure is provided, then get an existing one or create a new one
    fig = fig if fig is not None else plt.gcf()

    num_pairs = pairs.shape[0]

    # Set up labels for the x-axis
    if group_names is None:
        group_names = [f"{i+1}" for i in range(num_groups)]
    xticklbls = np.full((num_pairs, 3), "", dtype=object)
    for pair_idx in range(num_pairs):
        # Total in group 1
        xticklbls[pair_idx, 0] = f"Total, {group_names[pairs[pair_idx, 0]]}"
        # Shared between both groups
        xticklbls[pair_idx, 1] = (
            f"{group_names[pairs[pair_idx, 0]]}" + f"-{group_names[pairs[pair_idx, 1]]}"
        )
        # Total in group 2
        xticklbls[pair_idx, 2] = f"Total, {group_names[pairs[pair_idx, 1]]}"

    # Plot dimensionalities
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

        # Adjust appearance of axes
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    # Make sure the subplots don't overlap
    fig.tight_layout()


def plot_var_exp_pairs(
    pair_var_exp: np.ndarray,
    pairs: np.ndarray,
    num_groups: int,
    sem_pair_var_exp: np.ndarray | None = None,
    group_names: list[str] | None = None,
    fig: Figure | None = None,
) -> None:
    """
    Visualize pairwise analyses of shared variance.

    Visualize the shared variance explained in each group by their pairwise
    interaction with another group.

    Parameters
    ----------
    pair_var_exp
        `ndarray` of `float`, shape ``(num_pairs, 2)``.
        ``pair_var_exp[i,0]`` -- shared variance explained by pairwise
        interaction ``i`` in group ``1``.
        ``pair_var_exp[i,1]`` -- shared variance explained by pairwise
        interaction ``i`` in group ``2``.
    pairs
        `ndarray` of `int`, shape ``(num_pairs, 2)``.
        ``pairs[i,0]`` -- index of group ``1`` in pair ``i``.
        ``pairs[i,1]`` -- index of group ``2`` in pair ``i``.
    num_groups
        Number of observed groups.
    sem_pair_var_exp
        `ndarray` of `float`, shape ``(num_pairs, num_groups)``.
        Standard error of ``pair_var_exp``. Defaults to ``None``.
    group_names
        Names of the groups. Defaults to ``None``.
    fig
        Figure on which to draw the diagram. Defaults to ``None``.
    """
    # If no figure is provided, then get an existing one or create a new one
    fig = fig if fig is not None else plt.gcf()

    num_pairs = pairs.shape[0]

    # Set up labels for the x-axis
    if group_names is None:
        group_names = [f"{i+1}" for i in range(num_groups)]

    pairlbls = np.array(
        [
            f"{group_names[pairs[i,0]]}" + f"-{group_names[pairs[i,1]]}"
            for i in range(num_pairs)
        ]
    )

    # Plot pairwise shared variances
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

        # Adjust appearance of axes
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    # Make sure the subplots don't overlap
    fig.tight_layout()
