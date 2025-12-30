"""General plotting support across the latents package."""

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
    >>> from latents.plotting import hinton
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
