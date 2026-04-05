"""
General plotting support across the latents package.

**Functions**

- :func:`hinton` -- Draw a Hinton diagram of a matrix.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


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

    Examples
    --------
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


def find_optimal_permutation(C_estimated, C_true):
    """Find the optimal permutation of latent indices to match loadings.

    For each permutation, find the optimal individual sign flips for each latent.

    Parameters
    ----------
    C_estimated : np.ndarray
        Estimated C.mean matrix
    C_true : np.ndarray
        True C.mean matrix

    Returns
    -------
    permutation : np.ndarray
        Optimal permutation of indices
    sign_flips : np.ndarray
        Optimal sign flips (1 for no flip, -1 for flip) for each latent
    cost : float
        Total cost (sum of absolute differences)
    """
    from itertools import permutations
    from math import factorial
    n_latents = C_estimated.shape[1]  # Number of latent dimensions
    if C_estimated.shape[1]!=C_true.shape[1]:
        raise ValueError('The number of latents are different')
    if n_latents > 8:
        print(
            f"Warning: {n_latents} latents would require "
            f"{n_latents}! = {factorial(n_latents)} permutations"
        )
        print("This might take a while...")

    print(f"Exploring all {factorial(n_latents)} possible permutations...")

    best_cost = float("inf")
    best_permutation = None
    best_sign_flips = None

    # Try all possible permutations
    for perm in permutations(range(n_latents)):
        # Apply permutation to estimated C
        C_permuted = C_estimated[:, perm]

        # For each latent dimension (column), find the optimal sign
        sign_flips = np.ones(n_latents)
        for i in range(n_latents):
            # Compare positive vs negative for this specific latent column
            cost_pos = np.sum(np.abs(C_permuted[:, i] - C_true[:, i]))
            cost_neg = np.sum(np.abs(-C_permuted[:, i] - C_true[:, i]))

            if cost_neg < cost_pos:
                sign_flips[i] = -1

        # Apply the optimal sign flips to each column
        C_optimized = C_permuted * sign_flips
        cost = np.sum(np.abs(C_optimized - C_true))

        if cost < best_cost:
            best_cost = cost
            best_permutation = list(perm)
            best_sign_flips = sign_flips.copy()

    return np.array(best_permutation), best_sign_flips, best_cost
