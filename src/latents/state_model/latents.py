"""
Building blocks for latent state models.

**Classes**

- :class:`PosteriorLatentStatic` -- Posterior estimates of static latents.
- :class:`PosteriorLatentDelayed` -- Posterior estimates of time-delayed latents.
- :class:`StateParamsStatic` -- A generic state model with static latents.

"""

from __future__ import annotations

import numpy as np

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
    ) -> np.ndarray | None:
        """
        Compute the posterior second moments of the latent variables.

        Parameters
        ----------
        in_place
            If ``True``, compute the posterior second moments in place.
            If ``False``, compute the posterior second moments and return as a
            new `ndarray` without modifying self. Defaults to ``True``.

        Returns
        -------
        ndarray | None
            `ndarray` of shape ``(x_dim, x_dim)``.
            Posterior second moments of the latent variables.
        """
        x_dim, N = self.mean.shape
        if in_place:
            if self.moment is None:
                self.moment = np.zeros((x_dim, x_dim))
            self.moment[:] = N * self.cov + self.mean @ self.mean.T

        return N * self.cov + self.mean @ self.mean.T

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> PosteriorLatentStatic | None:
        """
        Keep only a subset of the latent dimensions in each attribute.

        Parameters
        ----------
        dims
            `ndarray` of `int`, at most length ``x_dim``.
            Indexes into the latent dimensions to keep.
        in_place
            If ``True``, modify self in place.
            If ``False``, copy over the relevant subsets of dimensions to
            a new ``PosteriorLatentStatic``, and return that new
            ``PosteriorLatentStatic``. Defaults to ``True``.

        Returns
        -------
        PosteriorLatentStatic | None
            A new ``PosteriorLatentStatic`` object with only the specified
            latent dimensions.
        """
        if in_place:
            # Keep only the specified dimensions
            self.mean = self.mean[dims, :] if self.mean is not None else None
            self.cov = self.cov[np.ix_(dims, dims)] if self.cov is not None else None
            self.moment = (
                self.moment[np.ix_(dims, dims)] if self.moment is not None else None
            )
            return None

        # Copy over the relevant subsets of dimensions to a new
        # PosteriorLatentStatic, and return that new PosteriorLatentStatic
        new_mean = self.mean[dims, :] if self.mean is not None else None
        new_cov = self.cov[np.ix_(dims, dims)] if self.cov is not None else None
        new_moment = (
            self.moment[np.ix_(dims, dims)] if self.moment is not None else None
        )
        return self.__class__(mean=new_mean, cov=new_cov, moment=new_moment)


class PosteriorLatentDelayed(ArrayContainer):
    """Posterior estimates of time-delayed latent variables.

    Parameters
    ----------
    mean
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, N)``.
        Posterior mean of the latent variables.
    cov
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, x_dim, num_groups, T)``.
        Posterior covariance of the latent variables.
    moment
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, x_dim, num_groups, T)``.
        Posterior second moments of the latent variables.
    moment_GP
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, x_dim, num_groups, T)``.
        Posterior second moments of the latent variables for the GP case.

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

    def compute_moment_GP(self, in_place: bool = True) -> np.ndarray | None:
        """
        Compute the posterior second moments of the latent variables for GP case.

        This function is written to be differentiable with respect to the parameters x.

        Parameters
        ----------
        in_place
            If ``True``, compute the posterior second moments in place.
            If ``False``, compute the posterior second moments and return as a
            new `ndarray` without modifying self. Defaults to ``True``.

        Returns
        -------
        ndarray | None
            `ndarray` of shape ``(x_dim, num_groups*T, num_groups*T, N)``.
            Posterior second moments of the latent variables.
        """
        x_dim, num_groups, T, N = self.mean.shape
        moment_GP = np.zeros((x_dim, num_groups * T, num_groups * T, N))

        for j in range(x_dim):
            for n in range(N):
                Sig_jj = self.cov[j, :, :, j, :, :].reshape(
                    num_groups * T, num_groups * T, order="F"
                )
                mu_j = self.mean[j, :, :, n].reshape(num_groups * T, 1, order="F")
                moment_GP[j, :, :, n] = Sig_jj + mu_j @ mu_j.T

        if in_place:
            self.moment = moment_GP
            return None

        return moment_GP

    def compute_moment(self, in_place: bool = True) -> np.ndarray | None:
        """Compute the posterior second moments of the latent variables."""
        raise NotImplementedError

    def get_subset_dims(
        self, dims: np.ndarray, in_place: bool = True
    ) -> PosteriorLatentDelayed | None:
        """Keep only a subset of the latent dimensions in each attribute."""
        raise NotImplementedError


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
    ) -> StateParamsStatic | None:
        """
        Keep only a subset of the latent dimensions in each relevant parameter.

        Parameters
        ----------
        dims
            1D `ndarray` of `int`, at most length ``x_dim``.
            Indexes into the latent dimensions to keep.
        in_place
            If ``True``, modify self in place.
            If ``False``, copy over parameters with the relevant subsets of
            dimensions to a new ``StateParamsStatic``, and return that new
            ``StateParamsStatic``. Defaults to ``True``.

        Returns
        -------
        StateParamsStatic | None
            A new ``StateParamsStatic`` object whose parameters have only the specified
            latent dimensions.
        """
        if in_place:
            # Keep only the specified dimensions
            self.x_dim = len(dims)
            self.X.get_subset_dims(dims, in_place=True)
            return None

        # Copy over parameters with the relevant subsets of dimensions to a
        # new StateParamsStatic object, and return that new StateParamsStatic object.
        return self.__class__(
            x_dim=len(dims),
            X=self.X.get_subset_dims(dims, in_place=False),
        )

    def copy(self) -> StateParamsStatic:
        """
        Return a copy of self.

        Returns
        -------
        StateParamsStatic
            A copy of self.
        """
        return self.__class__(
            x_dim=self.x_dim,
            X=self.X.copy(),
        )


class StateParamsDelayed:
    """A Gaussian Process state model."""

    def __init__(
        self,
        x_dim: int | None = None,
        num_groups: int | None = None,
        T: int | None = None,
        X: PosteriorLatentDelayed | None = None,
    ):
        # Latent dimensionality
        if x_dim is not None and not isinstance(x_dim, int):
            msg = "x_dim must be an integer."
            raise TypeError(msg)
        self.x_dim = x_dim

        # Latents
        if X is None:
            self.X = PosteriorLatentDelayed()
        elif not isinstance(X, PosteriorLatentDelayed):
            msg = "X must be a PosteriorLatentDelayed object."
            raise TypeError(msg)
        else:
            self.X = X

        # Number of groups
        if num_groups is not None and not isinstance(num_groups, int):
            msg = "num_groups must be an integer."
            raise TypeError(msg)
        self.num_groups = num_groups

        # Time points
        if T is not None and not isinstance(T, int):
            msg = "T must be an integer."
            raise TypeError(msg)
        self.T = T

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"x_dim={self.x_dim}, "
            f"num_groups={self.num_groups}, "
            f"T={self.T}, "
            f"X={self.X}, "
        )

    def is_initialized(self) -> bool:
        """Check if observation model parameters have been initialized to data."""
        raise NotImplementedError

    def get_subset_dims(
        self, dims: np.ndarray, in_place: bool = True
    ) -> StateParamsDelayed | None:
        """Keep only a subset of the latent dimensions in each relevant parameter."""
        raise NotImplementedError

    def copy(self) -> StateParamsDelayed:
        """Return a copy of self."""
        raise NotImplementedError
