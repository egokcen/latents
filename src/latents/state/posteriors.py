"""Posterior distributions for state model parameters."""

from __future__ import annotations

import sys

import numpy as np

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from latents._internal.base import ArrayContainer
from latents.state.realizations import LatentsRealization


class LatentsPosteriorStatic(ArrayContainer):
    """Posterior distribution q(X) for static latents.

    Parameters
    ----------
    mean
        Posterior mean, shape (x_dim, n_samples).
    cov
        Posterior covariance (shared across samples), shape (x_dim, x_dim).
    moment
        Posterior second moments, shape (x_dim, x_dim).

    Attributes
    ----------
    mean
        Same as **mean**, above.
    cov
        Same as **cov**, above.
    moment
        Same as **moment**, above.
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        moment: np.ndarray | None = None,
    ):
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

        if cov is not None and not isinstance(cov, np.ndarray):
            msg = "cov must be a numpy.ndarray."
            raise TypeError(msg)
        self.cov = cov

        if moment is not None and not isinstance(moment, np.ndarray):
            msg = "moment must be a numpy.ndarray."
            raise TypeError(msg)
        self.moment = moment

    @property
    def x_dim(self) -> int | None:
        """Number of latent dimensions."""
        if self.mean is None:
            return None
        return self.mean.shape[0]

    @property
    def n_samples(self) -> int | None:
        """Number of samples."""
        if self.mean is None:
            return None
        return self.mean.shape[1]

    def is_initialized(self) -> bool:
        """Check if posterior has been initialized.

        Returns
        -------
        bool
            True if mean is not None.
        """
        return self.mean is not None

    @property
    def posterior_mean(self) -> LatentsRealization:
        """Return posterior mean as a realization.

        Returns
        -------
        LatentsRealization
            Posterior mean wrapped as a realization.
        """
        return LatentsRealization(X=self.mean.copy())

    def sample(self, rng: np.random.Generator) -> LatentsRealization:
        """Draw X from the posterior distribution.

        Parameters
        ----------
        rng
            Random number generator.

        Returns
        -------
        LatentsRealization
            Sampled latent values.
        """
        # Sample deviations from mean using shared covariance
        # X_sample = mean + chol(cov) @ z, where z ~ N(0, I)
        samples = (
            rng.multivariate_normal(
                np.zeros(self.x_dim),
                self.cov,
                size=self.n_samples,
            ).T
            + self.mean
        )
        return LatentsRealization(X=samples)

    def compute_moment(self, in_place: bool = True) -> np.ndarray:
        """Compute the posterior second moments.

        E[X X^T] = n_samples * cov + mean @ mean^T

        Parameters
        ----------
        in_place
            If True, store result in self.moment and return reference to it.
            If False, return a new array without modifying self.
            Defaults to True.

        Returns
        -------
        ndarray
            Shape (x_dim, x_dim). Posterior second moments.
        """
        x_dim, n_samples = self.mean.shape
        if in_place:
            if self.moment is None:
                self.moment = np.zeros((x_dim, x_dim))
            self.moment[:] = n_samples * self.cov + self.mean @ self.mean.T
            return self.moment

        return n_samples * self.cov + self.mean @ self.mean.T

    def get_subset_dims(
        self,
        x_indices: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """Keep only a subset of the latent dimensions.

        Parameters
        ----------
        x_indices
            1D array of int, at most length x_dim.
            Indices of the latent dimensions to keep.
        in_place
            If True, modify self in place and return self.
            If False, return a new instance with the subset.
            Defaults to True.

        Returns
        -------
        Self
            The modified instance (if in_place=True) or a new instance
            with only the specified latent dimensions.
        """
        new_mean = self.mean[x_indices, :] if self.mean is not None else None
        new_cov = (
            self.cov[np.ix_(x_indices, x_indices)] if self.cov is not None else None
        )
        new_moment = (
            self.moment[np.ix_(x_indices, x_indices)]
            if self.moment is not None
            else None
        )

        if in_place:
            self.mean = new_mean
            self.cov = new_cov
            self.moment = new_moment
            return self

        return self.__class__(mean=new_mean, cov=new_cov, moment=new_moment)


class LatentsPosteriorTimeSeries(ArrayContainer):
    """Posterior distribution q(X) for time series latents.

    Stub for GPFA.
    """

    def __init__(self) -> None:
        msg = "LatentsPosteriorTimeSeries not yet implemented"
        raise NotImplementedError(msg)


class LatentsPosteriorDelayed(ArrayContainer):
    """Posterior distribution q(X) for time-delayed latents.

    Stub for mDLAG.
    """

    def __init__(self) -> None:
        msg = "LatentsPosteriorDelayed not yet implemented"
        raise NotImplementedError(msg)
