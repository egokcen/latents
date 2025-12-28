"""Building blocks for latent state models."""

from __future__ import annotations

import sys

import numpy as np

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from latents.base import ArrayContainer


class PosteriorLatentStatic(ArrayContainer):
    """
    Posterior estimates of static latent variables.

    Parameters
    ----------
    mean
        `ndarray` of `float`, shape ``(x_dim, N)``.
        Posterior mean of the latent variables.
    cov
        `ndarray` of `float`, shape ``(x_dim, x_dim)``.
        Posterior covariance of the latent variables.
    moment
        `ndarray` of `float`, shape ``(x_dim, x_dim)``.
        Posterior second moments of the latent variables.

    Attributes
    ----------
    mean
        Same as **mean**, above.
    cov
        Same as **cov**, above.
    moment
        Same as **moment**, above.

    Raises
    ------
    TypeError
        If ``mean``, ``cov``, or ``moment`` is not a `ndarray`.
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        moment: np.ndarray | None = None,
    ):
        # Mean
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

        # Covariance
        if cov is not None and not isinstance(cov, np.ndarray):
            msg = "cov must be a numpy.ndarray."
            raise TypeError(msg)
        self.cov = cov

        # Second moment
        if moment is not None and not isinstance(moment, np.ndarray):
            msg = "moment must be a numpy.ndarray."
            raise TypeError(msg)
        self.moment = moment

    def compute_moment(
        self,
        in_place: bool = True,
    ) -> np.ndarray:
        """
        Compute the posterior second moments of the latent variables.

        Parameters
        ----------
        in_place
            If ``True``, store result in ``self.moment`` and return reference to it.
            If ``False``, return a new array without modifying self.
            Defaults to ``True``.

        Returns
        -------
        ndarray
            `ndarray` of shape ``(x_dim, x_dim)``.
            Posterior second moments of the latent variables.
        """
        x_dim, N = self.mean.shape
        if in_place:
            if self.moment is None:
                self.moment = np.zeros((x_dim, x_dim))
            self.moment[:] = N * self.cov + self.mean @ self.mean.T
            return self.moment

        return N * self.cov + self.mean @ self.mean.T

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """
        Keep only a subset of the latent dimensions in each attribute.

        Parameters
        ----------
        dims
            1D `ndarray` of `int`, at most length ``x_dim``.
            Indexes into the latent dimensions to keep.
        in_place
            If ``True``, modify self in place and return self.
            If ``False``, return a new instance with the subset.
            Defaults to ``True``.

        Returns
        -------
        Self
            The modified instance (if ``in_place=True``) or a new instance
            with only the specified latent dimensions.
        """
        # Compute sliced arrays once
        new_mean = self.mean[dims, :] if self.mean is not None else None
        new_cov = self.cov[np.ix_(dims, dims)] if self.cov is not None else None
        new_moment = (
            self.moment[np.ix_(dims, dims)] if self.moment is not None else None
        )

        if in_place:
            self.mean = new_mean
            self.cov = new_cov
            self.moment = new_moment
            return self

        return self.__class__(mean=new_mean, cov=new_cov, moment=new_moment)


class PosteriorLatentDelayed(ArrayContainer):
    """Posterior estimates of time-delayed latent variables."""

    pass


class StateParamsStatic:
    """
    A generic state model with static latent variables.

    Parameters
    ----------
    x_dim
        Number of latent dimensions.
    X
        Posterior latents.

    Attributes
    ----------
    x_dim
        Same as **x_dim**, above.
    X
        Same as **X**, above.

    Raises
    ------
    TypeError
        If any attribute is not of the respective type specified above.
    """

    def __init__(
        self,
        x_dim: int | None = None,
        X: PosteriorLatentStatic | None = None,
    ):
        # Latent dimensionality
        if x_dim is not None and not isinstance(x_dim, int):
            msg = "x_dim must be an integer."
            raise TypeError(msg)
        self.x_dim = x_dim

        # Latents
        if X is None:
            self.X = PosteriorLatentStatic()
        elif not isinstance(X, PosteriorLatentStatic):
            msg = "X must be a PosteriorLatentStatic object."
            raise TypeError(msg)
        else:
            self.X = X

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x_dim={self.x_dim}, X={self.X})"

    def is_initialized(self) -> bool:
        """
        Check if observation model parameters have been initialized to data.

        Returns
        -------
        bool
            ``True`` if all parameters have been initialized to data.
        """
        return self.x_dim is not None and self.X.mean is not None

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """
        Keep only a subset of the latent dimensions in each relevant parameter.

        Parameters
        ----------
        dims
            1D `ndarray` of `int`, at most length ``x_dim``.
            Indexes into the latent dimensions to keep.
        in_place
            If ``True``, modify self in place and return self.
            If ``False``, return a new instance with the subset.
            Defaults to ``True``.

        Returns
        -------
        Self
            The modified instance (if ``in_place=True``) or a new instance
            with only the specified latent dimensions.
        """
        if in_place:
            self.x_dim = len(dims)
            self.X.get_subset_dims(dims, in_place=True)
            return self

        return self.__class__(
            x_dim=len(dims),
            X=self.X.get_subset_dims(dims, in_place=False),
        )

    def copy(self) -> Self:
        """
        Return a copy of self.

        Returns
        -------
        Self
            A copy of self.
        """
        return self.__class__(
            x_dim=self.x_dim,
            X=self.X.copy(),
        )
