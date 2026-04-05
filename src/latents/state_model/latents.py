"""
Building blocks for latent state models.

**Classes**

- :class:`PosteriorLatentStatic` -- Posterior estimates of static latents.
- :class:`PosteriorLatentDelayed` -- Posterior estimates of time-delayed latents.
- :class:`StateParamsStatic` -- A generic state model with static latents.
- :class:`StateParamsDelayed` -- A state model for mdlag latent variables.
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
        Posterior mean of the latent variables, where:
        - ``x_dim`` is the number of latent dimensions
        - ``num_groups`` is the number of groups
        - ``T`` is the number of timepoints
        - ``N`` is the number of trials
    cov
        `ndarray` of `float`, shape ``(x_dim, num_groups, T, x_dim, num_groups, T)``.
        Posterior covariance of the latent variables.
    moment
        `ndarray` of `float`, shape ``(num_groups, x_dim, x_dim)``.
        Posterior second moments of the latent variables.

    Attributes
    ----------
    mean
        Same as **mean**, above.
    cov
        Same as **cov**, above.
    moment
        Same as **moment**, above.
    moment_gp
        `ndarray` of `float`, shape ``(x_dim, num_groups * T, num_groups * T)``.
        Posterior second moments for each latent GP.
    logdet_SigX
        `float`.
        Log determinant of the posterior covariance of :math:`X`.

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

        # GP second moment
        self.moment_gp = None

        # determinant of the GP covariance matrix
        self.logdet_SigX = None

    def compute_moment_gp(self, in_place: bool = True) -> np.ndarray | None:
        """Compute a subset of the posterior second moments.

        Corresponding to each latent GP across time points.

        The full posterior second moment of the latent variables is a
        (num_groups * x_dim * T, num_groups * x_dim * T) matrix.
        Here, compute a (num_groups * T, num_groups * T) subset of that
        larger matrix for each latent variable.

        Parameters
        ----------
        in_place : bool
            If ``True``, store the computed moments in self.moment_gp.
            If ``False``, return the computed moments without modifying self.
            Defaults to ``True``.

        Returns
        -------
        ndarray | None
            If ``in_place=False``, returns `ndarray` of shape
            ``(x_dim, num_groups * T, num_groups * T)``.
            Posterior second moments for each latent GP.
            If ``in_place=True``, returns ``None``.

        Raises
        ------
        ValueError
            If covariance is not available (self.cov is None).
        """
        if self.cov is None:
            msg = (
                "Covariance is required to compute GP moments. "
                "Set save_X_cov=True in infer_latents."
            )
            raise ValueError(msg)

        x_dim, num_groups, T, N = self.mean.shape
        moment_gp = np.zeros((x_dim, num_groups * T, num_groups * T))
        mean_reshaped = self.mean.reshape(x_dim, num_groups * T, N, order="F")

        for j in range(x_dim):
            Sig_jj = self.cov[j, :, :, j, :, :].reshape(
                num_groups * T, num_groups * T, order="F"
            )
            mu_j = mean_reshaped[j]
            moment_gp[j] = N * Sig_jj + mu_j @ mu_j.T

        if in_place:
            self.moment_gp = moment_gp
            return None
        return moment_gp

    def compute_moment(self, in_place: bool = True) -> np.ndarray | None:
        """Compute a subset of the posterior second moments.

        For each time point across latents and groups.

        The full posterior second moment of the latent variables is a
        (num_groups * x_dim * T, num_groups * x_dim * T) matrix.
        Here, compute a (x_dim, x_dim) subset of that
        larger matrix for each group.

        Parameters
        ----------
        in_place : bool
            If ``True``, store the computed moments in self.moment.
            If ``False``, return the computed moments without modifying self.
            Defaults to ``True``.

        Returns
        -------
        ndarray | None
            If ``in_place=False``, returns `ndarray` of shape
            ``(num_groups, x_dim, x_dim)``.
            Posterior second moments for each time point.
            If ``in_place=True``, returns ``None``.

        Raises
        ------
        ValueError
            If covariance is not available (self.cov is None).
        """
        if self.cov is None:
            msg = (
                "Covariance is required to compute moments. "
                "Set save_X_cov=True in infer_latents."
            )
            raise ValueError(msg)

        x_dim, num_groups, T, N = self.mean.shape
        moment = np.zeros((num_groups, x_dim, x_dim))

        cov_reshaped = self.cov.reshape(
            x_dim * num_groups, T, x_dim * num_groups, T, order="F"
        )
        cov_time_blocks = np.empty((x_dim * num_groups, x_dim * num_groups, T))
        for t in range(T):
            cov_time_blocks[:, :, t] = cov_reshaped[:, t, :, t]

        for group_idx in range(num_groups):
            lat_idxs = slice(group_idx * x_dim, (group_idx + 1) * x_dim)
            cov_term = N * np.sum(cov_time_blocks[lat_idxs, lat_idxs, :], axis=2)
            group_mean_slice = self.mean[:, group_idx, :, :]
            mean_term = np.einsum("itn,jtn->ij", group_mean_slice, group_mean_slice)
            moment[group_idx] = cov_term + mean_term

        if in_place:
            self.moment = moment
            return None
        return moment

    def get_subset_dims(
        self, dims: np.ndarray, in_place: bool = True
    ) -> PosteriorLatentDelayed | None:
        """Keep only a subset of the latent dimensions in each attribute."""
        if in_place:
            self.mean = self.mean[dims, :, :, :] if self.mean is not None else None
            if self.cov is not None:
                self.cov = self.cov[dims, :, :, :, :, :][:, :, :, dims, :, :]
            if self.moment is not None:
                self.moment = self.moment[:, dims, :][:, :, dims]
            return None
        else:
            new_cov = None
            if self.cov is not None:
                new_cov = self.cov[dims, :, :, :, :, :][:, :, :, dims, :, :]
            new_moment = None
            if self.moment is not None:
                new_moment = self.moment[:, dims, :][:, :, dims]
            return self.__class__(
                mean=self.mean[dims, :, :, :] if self.mean is not None else None,
                cov=new_cov,
                moment=new_moment,
            )


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
        Check if state model parameters have been initialized.

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
    """
    A Gaussian Process state model with time-delayed latent variables.

    Parameters
    ----------
    x_dim : int | None, optional
        Number of latent dimensions.
    num_groups : int | None, optional
        Number of groups in the data.
    T : int | None, optional
        Number of time points.
    X : PosteriorLatentDelayed | None, optional
        Posterior latents.

    Attributes
    ----------
    x_dim : int | None
        Same as **x_dim**, above.
    num_groups : int | None
        Same as **num_groups**, above.
    T : int | None
        Same as **T**, above.
    X : PosteriorLatentDelayed
        Same as **X**, above.

    Raises
    ------
    TypeError
        If any attribute is not of the respective type specified above.
    """

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

        # Latents
        if X is None:
            self.X = PosteriorLatentDelayed()
        elif not isinstance(X, PosteriorLatentDelayed):
            msg = "X must be a PosteriorLatentDelayed object."
            raise TypeError(msg)
        else:
            self.X = X

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"x_dim={self.x_dim}, "
            f"num_groups={self.num_groups}, "
            f"T={self.T}, "
            f"X={self.X}, "
        )

    def is_initialized(self) -> bool:
        """Check if state model parameters have been initialized."""
        return self.x_dim is not None and self.X.mean is not None

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> StateParamsDelayed | None:
        """Keep only a subset of the latent dimensions in each relevant parameter."""
        if in_place:
            self.x_dim = len(dims)
            self.X.get_subset_dims(dims, in_place=True)
            return None
        else:
            return self.__class__(
                x_dim=len(dims),
                num_groups=self.num_groups,
                T=self.T,
                X=self.X.get_subset_dims(dims, in_place=False),
            )

    def copy(self) -> StateParamsDelayed:
        """Return a copy of self."""
        raise NotImplementedError
