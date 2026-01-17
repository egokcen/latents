"""Observation data containers for latent variable models."""

from __future__ import annotations

import numpy as np


class ObsStatic:
    """Store and manipulate views of observed static data.

    Parameters
    ----------
    data : ndarray of float, shape (y_dim, n_samples)
        Observed data. Groups are stacked vertically. For example, if there
        are three groups with dimensionalities 2, 3, and 4, then ``data`` is a
        ndarray of shape ``(9, n_samples)``, and ``data[:2, :]`` contains the first
        group, ``data[2:5, :]`` contains the second group, and ``data[5:, :]``
        contains the third group.
    dims : ndarray of int, shape (n_groups,)
        Dimensionalities of each observed group.

    Attributes
    ----------
    data : ndarray of float, shape (y_dim, n_samples)
        Observed data. Groups are stacked vertically.
    dims : ndarray of int, shape (n_groups,)
        Dimensionalities of each observed group.

    Raises
    ------
    TypeError
        If ``data`` or ``dims`` is not a ndarray.
    ValueError
        If the sum of ``dims`` does not equal the number of rows in ``data``.

    Examples
    --------
    Create observation data with two groups (3 and 2 dimensions):

    >>> import numpy as np
    >>> from latents.data import ObsStatic
    >>> data = np.random.randn(5, 100)  # 5 total dims, 100 samples
    >>> dims = np.array([3, 2])  # Group 1 has 3 dims, group 2 has 2
    >>> Y = ObsStatic(data, dims)
    >>> Y
    ObsStatic(data.shape=(5, 100), dims=[3 2])

    Access data for each group separately:

    >>> groups = Y.get_groups()
    >>> groups[0].shape  # First group
    (3, 100)
    >>> groups[1].shape  # Second group
    (2, 100)
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
        return f"{type(self).__name__}(data.shape={self.data.shape}, dims={self.dims})"

    def get_groups(self) -> list[np.ndarray]:
        """Return a list of views of the observed data, one for each group.

        Returns
        -------
        list of ndarray
            Views of the observed data, one per group, length ``n_groups``.
        """
        return np.split(self.data, np.cumsum(self.dims)[:-1], axis=0)


class ObsTimeSeries:
    """Store and manipulate views of observed time series data.

    Stub for future implementation.

    Raises
    ------
    NotImplementedError
        Always raised; this class is a placeholder for future implementation.
    """

    def __init__(self) -> None:
        msg = "ObsTimeSeries not yet implemented"
        raise NotImplementedError(msg)
