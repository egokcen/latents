"""
Store and manipulate different types of observed data.

**Classes**

- :class:`ObsStatic` -- Store and manipulate views of observed static data.
- :class:`ObsTimeSeries` -- Store and manipulate views of observed time series data.

"""

from __future__ import annotations

import numpy as np


class ObsStatic:
    """
    Store and manipulate views of observed static data.

    Parameters
    ----------
    data
        `ndarray` of `float`, shape ``(dim, N)``.
        Observed data. Groups are stacked vertically. For example, if there
        are three groups with dimensionalities 2, 3, and 4, then ``data`` is a
        `ndarray` of shape ``(9, N)``, and ``data[:2, :]`` contains the first
        group, ``data[2:5, :]`` contains the second group, and ``data[5:, :]``
        contains the third group.
    dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group.

    Attributes
    ----------
    data
        Same as **data**, above.
    dims
        Same as **dims**, above.

    Raises
    ------
    TypeError
        If ``data`` or ``dims`` is not a `ndarray`.
    ValueError
        If the sum of ``dims`` does not equal the number of rows in ``data``.
    """

    def __init__(
        self,
        data: np.ndarray,
        dims: np.ndarray,
    ):
        # Observed data
        if not isinstance(data, np.ndarray):
            msg = "data must be a numpy.ndarray."
            raise TypeError(msg)

        # Dimensionalities of each group
        if not isinstance(dims, np.ndarray):
            msg = "dims must be a numpy.ndarray."
            raise TypeError(msg)

        # Check that the dimensionalities of each group are consistent with
        # the shape of Y
        if np.sum(dims) != data.shape[0]:
            msg = "The sum of dims must equal the number of rows in data."
            raise ValueError(msg)

        self.dims = dims
        self.data = data

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"data.shape={self.data.shape}, "
            f"dims={self.dims})"
        )

    def get_groups(self) -> list[np.ndarray]:
        """
        Return a list of views of the observed data, one for each group.

        Returns
        -------
        list[ndarray]
            *list* of `ndarray`, length ``num_groups``.
            List of views of the observed data, one for each group.
        """
        return np.split(self.data, np.cumsum(self.dims)[:-1], axis=0)


class ObsTimeSeries:
    """Store and manipulate views of observed time series data."""

    pass
