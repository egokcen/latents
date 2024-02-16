"""
Custom data types used throughout the GFA package.

**Classes**

- :class:`ObsData` -- Store and manipulate views of observed data.
- :class:`ArrayContainer` -- A parent class for classes with ndarray attributes.
- :class:`PosteriorLatent` -- Posterior estimates of latent variables.
- :class:`PosteriorLoading` -- Posterior estimates of loading matrices.
- :class:`PosteriorARD` -- Posterior estimates of ARD parameters.
- :class:`PosteriorObsMean` -- Posterior estimates of mean parameters.
- :class:`PosteriorObsPrec` -- Posterior estimates of precision parameters.
- :class:`GFAParams` -- GFA model parameters.
- :class:`HyperPriorParams` -- GFA prior hyperparameters.
- :class:`GFAFitTracker` -- Contains quantities tracked during a GFA model fit.
- :class:`GFAFitFlags` -- Contains status messages during a GFA model fit.
- :class:`GFAFitArgs` -- Contains keyword arguments used to fit a GFA model.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class ObsData:
    """
    A class to store and manipulate views of observed data.

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


class ArrayContainer:
    """A parent class for classes with numpy.ndarray attributes."""

    def __repr__(self) -> str:
        # If array attributes are specified, then display their shapes, rather
        # than their full values
        attr_reprs = []
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray):
                attr_reprs.append(f"{attr_name}.shape={attr_value.shape}")
            else:
                attr_reprs.append(f"{attr_name}={attr_value}")

        return type(self).__name__ + "(" + ", ".join(attr_reprs) + ")"

    def copy(self) -> ArrayContainer:
        """
        Return a deep copy of self.

        Returns
        -------
        ArrayContainer
            A deep copy of self.
        """
        # If array attributes are specified, then create copies of them, and
        # create a new class instance with those copies
        new_attrs = {}
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray):
                new_attrs[attr_name] = attr_value.copy()
            else:
                new_attrs[attr_name] = attr_value
        return self.__class__(**new_attrs)

    def clear(self) -> None:
        """Set all attributes to None."""
        for attr_name in vars(self):
            setattr(self, attr_name, None)


class PosteriorLatent(ArrayContainer):
    """
    A class for posterior estimates of latent variables.

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
    ) -> PosteriorLatent | None:
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
            a new ``PosteriorLatent``, and return that new ``PosteriorLatent``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorLatent | None
            A new ``PosteriorLatent`` object with only the specified
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
        # PosteriorLatent, and return that new PosteriorLatent
        new_mean = self.mean[dims, :] if self.mean is not None else None
        new_cov = self.cov[np.ix_(dims, dims)] if self.cov is not None else None
        new_moment = (
            self.moment[np.ix_(dims, dims)] if self.moment is not None else None
        )
        return self.__class__(mean=new_mean, cov=new_cov, moment=new_moment)


class PosteriorLoading(ArrayContainer):
    """
    A class for posterior estimates of loading matrices.

    Parameters
    ----------
    mean
        `ndarray`, shape ``(y_dim, x_dim)``.
        Posterior mean of the loading matrices.
    cov
        `ndarray`, shape ``(y_dim, x_dim, x_dim)``.
        Posterior covariances of the loading matrices.
    moment
        `ndarray`, shape ``(y_dim, x_dim, x_dim)``.
        Posterior second moments of the loading matrices.

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

    def get_groups(
        self,
        dims: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Get a list of views into the posterior loadings parameters, one for each group.

        For the mean (mean), covariance (cov), and second moments (moment),
        return a list of views into each parameter, one view for each group.

        Parameters
        ----------
        dims
            `ndarray` of `int`, shape ``(num_groups,)``
            Dimensionalities of each observed group.

        Returns
        -------
        group_means: List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior mean of the loading matrices,
            one view for each group. If ``self.mean`` is ``None``, returns
            ``None``.
        group_covs : List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior covariance of the loading
            matrices, one view for each group. If ``self.cov`` is ``None``,
            returns ``None``.
        group_moments : List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior second moments of the loading
            matrices, one view for each group. If ``self.moment`` is ``None``,
            returns ``None``.

        Raises
        ------
        ValueError
            If the sum of ``dims`` does not equal the number of rows in
            ``mean``, ``cov``, or ``moment``.
        """
        # Split mean into groups
        group_means = None
        if self.mean is not None:
            if np.sum(dims) != self.mean.shape[0]:
                msg = "The sum of dims must equal the number of rows in mean."
                raise ValueError(msg)
            group_means = np.split(self.mean, np.cumsum(dims)[:-1], axis=0)

        # Split cov into groups
        group_covs = None
        if self.cov is not None:
            if np.sum(dims) != self.cov.shape[0]:
                msg = "The sum of dims must equal the size of the first axis of cov."
                raise ValueError(msg)
            group_covs = np.split(self.cov, np.cumsum(dims)[:-1], axis=0)

        # Split moment into groups
        group_moments = None
        if self.moment is not None:
            if np.sum(dims) != self.moment.shape[0]:
                msg = "The sum of dims must equal the size of the first axis of moment."
                raise ValueError(msg)
            group_moments = np.split(self.moment, np.cumsum(dims)[:-1], axis=0)

        return group_means, group_covs, group_moments

    def compute_moment(
        self,
        in_place: bool = True,
    ) -> np.ndarray | None:
        """
        Compute the posterior second moments of the loading matrices.

        Parameters
        ----------
        in_place
            If ``True``, compute the posterior second moments in place.
            If ``False``, compute the posterior second moments and return as a
            new `ndarray` without modifying self. Defaults to ``True``.

        Returns
        -------
        ndarray | None
            `ndarray`, shape ``(y_dim, x_dim, x_dim)``.
            Posterior second moments of the loading matrices.
        """
        if in_place:
            if self.moment is None:
                self.moment = np.zeros_like(self.cov)
            self.moment[:] = np.einsum("ij,ik->ijk", self.mean, self.mean) + self.cov
            return None

        return np.einsum("ij,ik->ijk", self.mean, self.mean) + self.cov

    def compute_squared_norms(self, dims: np.ndarray) -> np.ndarray:
        """
        Compute the expected squared norm of each column of the loadings for each group.

        Parameters
        ----------
        dims
            `ndarray` of `int`, shape ``(num_groups,)``.
            Dimensionalities of each observed group.

        Returns
        -------
        squared_norms : ndarray
            `ndarray` of `float`, shape ``(num_groups, x_dim)``.
            ``squared_norms[i,j]`` is the expected squared norm of column ``j``
            of the loading matrix for group ``i``.
        """
        num_groups = len(dims)
        x_dim = self.moment.shape[1]

        # Get views of the moment matrix for each group
        _, _, moments = self.get_groups(dims)
        squared_norms = np.zeros((num_groups, x_dim))
        for group_idx in range(num_groups):
            # We need only the diagonal of each x_dim x x_dim moment matrix
            squared_norms[group_idx, :] = np.sum(
                moments[group_idx].diagonal(offset=0, axis1=1, axis2=2), axis=0
            )

        return squared_norms

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> PosteriorLoading | None:
        """
        Keep only a subset of the latent dimensions in each attribute.

        Parameters
        ----------
        dims
            1D `ndarray` of `int`, at most length ``x_dim``.
            Indexes into the latent dimensions to keep.
        in_place
            If ``True``, modify self in place.
            If ``False``, copy over the relevant subsets of dimensions to
            a new ``PosteriorLoading``, and return that new
            ``PosteriorLoading``. Defaults to ``True``.

        Returns
        -------
        PosteriorLoading | None
            A new ``PosteriorLoading`` object with only the specified
            latent dimensions.
        """
        if in_place:
            # Keep only the specified dimensions
            self.mean = self.mean[:, dims] if self.mean is not None else None
            self.cov = (
                self.cov[np.ix_(np.arange(self.cov.shape[0]), dims, dims)]
                if self.cov is not None
                else None
            )
            self.moment = (
                self.moment[np.ix_(np.arange(self.moment.shape[0]), dims, dims)]
                if self.moment is not None
                else None
            )
            return None

        # Copy over the relevant subsets of dimensions to a new
        # PosteriorLoading, and return that new PosteriorLoading
        new_mean = self.mean[:, dims] if self.mean is not None else None
        new_cov = (
            self.cov[np.ix_(np.arange(self.cov.shape[0]), dims, dims)]
            if self.cov is not None
            else None
        )
        new_moment = (
            self.moment[np.ix_(np.arange(self.moment.shape[0]), dims, dims)]
            if self.moment is not None
            else None
        )
        return self.__class__(mean=new_mean, cov=new_cov, moment=new_moment)


class PosteriorARD(ArrayContainer):
    """
    A class for posterior estimates of ARD parameters.

    Parameters
    ----------
    a
        `ndarray` of `float`, shape ``(num_groups,)``.
        Shape parameters of the ARD posterior.
    b
        `ndarray` of `float`, shape ``(num_groups, x_dim)``.
        Rate parameters of the ARD posterior.
    mean
        `ndarray` of `float`, shape ``(num_groups, x_dim)``.
        Posterior mean of the ARD parameters.

    Attributes
    ----------
    a
        Same as **a**, above.
    b
        Same as **b**, above.
    mean
        Same as **mean**, above.

    Raises
    ------
    TypeError
        If ``a``, ``b``, or ``mean`` is not a `ndarray`.
    """

    def __init__(
        self,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
        mean: np.ndarray | None = None,
    ):
        # Shape parameter
        if a is not None and not isinstance(a, np.ndarray):
            msg = "a must be a numpy.ndarray."
            raise TypeError(msg)
        self.a = a

        # Rate parameter
        if b is not None and not isinstance(b, np.ndarray):
            msg = "b must be a numpy.ndarray."
            raise TypeError(msg)
        self.b = b

        # Mean
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

    def compute_mean(
        self,
        in_place: bool = True,
    ) -> np.ndarray | None:
        """
        Compute the posterior mean of the ARD parameters, a / b.

        Parameters
        ----------
        in_place
            If ``True``, compute the posterior mean in place.
            If ``False``, compute the posterior mean and return as a new
            `ndarray` without modifying self. Defaults to ``True``.

        Returns
        -------
        ndarray | None
            `ndarray` of `float`, shape ``(num_groups, x_dim)``.
            Posterior mean of the ARD parameters.
        """
        if in_place:
            if self.mean is None:
                self.mean = np.zeros_like(self.b)
            self.mean[:] = self.a[:, np.newaxis] / self.b
            return None

        return self.a[:, np.newaxis] / self.b

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> PosteriorARD | None:
        """
        Keep only a subset of the latent dimensions in each attribute.

        Parameters
        ----------
        dims
            1D `ndarray` of `int`, at most length ``x_dim``.
            Indexes into the latent dimensions to keep.
        in_place
            If ``True``, modify self in place.
            If ``False``, copy over the relevant subsets of dimensions to
            a new ``PosteriorARD``, and return that new ``PosteriorARD``.
            Defaults to ``True``.

        Returns
        -------
        PosteriorARD | None
            A new ``PosteriorARD`` object with only the specified
            latent dimensions.
        """
        if in_place:
            # Keep only the specified dimensions
            self.b = self.b[:, dims] if self.b is not None else None
            self.mean = self.mean[:, dims] if self.mean is not None else None
            return None

        # Copy over the relevant subsets of dimensions to a new
        # PosteriorARD, and return that new PosteriorARD
        new_a = self.a.copy() if self.a is not None else None
        new_b = self.b[:, dims] if self.b is not None else None
        new_mean = self.mean[:, dims] if self.mean is not None else None
        return self.__class__(a=new_a, b=new_b, mean=new_mean)


class PosteriorObsMean(ArrayContainer):
    """
    A class for posterior estimates of observation mean parameters.

    Parameters
    ----------
    mean
        `ndarray` of `float`, shape ``(y_dim,)``.
        Posterior mean of the observation mean parameters.
    cov
        `ndarray` of `float`, shape ``(y_dim,)``.
        Posterior covariance of the observation mean parameters.

    Attributes
    ----------
    mean
        Same as **mean**, above.
    cov
        Same as **cov**, above.

    Raises
    ------
    TypeError
        If ``mean`` or ``cov`` is not a `ndarray`.
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        cov: np.ndarray | None = None,
    ):
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

        if cov is not None and not isinstance(cov, np.ndarray):
            msg = "cov must be a numpy.ndarray."
            raise TypeError(msg)
        self.cov = cov

    def get_groups(
        self,
        dims: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Get a list of views into the posterior mean parameters, one for each group.

        For the mean (mean) and covariance (cov), return a list of views into
        each parameter, one view for each group.

        Parameters
        ----------
        dims
            `ndarray` of `int`, shape ``(num_groups,)``.
            Dimensionalities of each observed group.

        Returns
        -------
        group_means : List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior mean of the observation mean
            parameters, one view for each group. If ``self.mean`` is ``None``,
            returns ``None``.
        group_covs : List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior covariance of the observation
            mean parameters, one view for each group. If ``self.cov`` is
            ``None``, returns ``None``.

        Raises
        ------
        ValueError
            If the sum of ``dims`` does not equal the length of ``mean`` or
            ``cov``.
        """
        # Split mean into groups
        group_means = None
        if self.mean is not None:
            if np.sum(dims) != len(self.mean):
                msg = "The sum of dims must equal the length of mean."
                raise ValueError(msg)
            group_means = np.split(self.mean, np.cumsum(dims)[:-1], axis=0)

        # Split cov into groups
        group_covs = None
        if self.cov is not None:
            if np.sum(dims) != len(self.cov):
                msg = "The sum of dims must equal the length of cov."
                raise ValueError(msg)
            group_covs = np.split(self.cov, np.cumsum(dims)[:-1], axis=0)

        return group_means, group_covs


class PosteriorObsPrec(ArrayContainer):
    """
    A class for posterior estimates of observation precision parameters.

    Parameters
    ----------
    a
        Shape parameter of the observation precision posterior.
    b
        `ndarray` of `float`, shape ``(y_dim,)``.
        Rate parameters of the observation precision posterior.
    mean
        `ndarray` of `float`, shape ``(y_dim,)``.
        Posterior mean of the observation precision parameters.

    Attributes
    ----------
    a
        Same as **a**, above.
    b
        Same as **b**, above.
    mean
        Same as **mean**, above.

    Raises
    ------
    TypeError
        If ``a`` is not a `float`, or if ``b`` or ``mean`` is not a `ndarray`.
    """

    def __init__(
        self,
        a: float | None = None,
        b: np.ndarray | None = None,
        mean: np.ndarray | None = None,
    ):
        # Shape parameter
        if a is not None and not isinstance(a, float):
            msg = "a must be a float."
            raise TypeError(msg)
        self.a = a

        # Rate parameter
        if b is not None and not isinstance(b, np.ndarray):
            msg = "b must be a numpy.ndarray."
            raise TypeError(msg)
        self.b = b

        # Mean
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

    def get_groups(
        self,
        dims: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Get a list of views into the posterior precision parameters, one for each group.

        For the mean (mean) and rate parameters (b), return a list of views into
        each parameter, one view for each group.

        Parameters
        ----------
        dims
            `ndarray` of `int`, shape ``(num_groups,)``.
            Dimensionalities of each observed group.

        Returns
        -------
        group_means : List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior mean of the observation
            precision parameters, one view for each group. If ``self.mean`` is
            ``None``, returns ``None``.
        group_bs : List[ndarray] | None
            list of `ndarray`, length ``num_groups``.
            List of views into the posterior rate parameters of the
            observation precision parameters, one view for each group.
            If ``self.b`` is ``None``, returns ``None``.

        Raises
        ------
        ValueError
            If the sum of ``dims`` does not equal the length of ``mean`` or
            ``b``.
        """
        # Split mean into groups
        group_means = None
        if self.mean is not None:
            if np.sum(dims) != len(self.mean):
                msg = "The sum of dims must equal the length of mean."
                raise ValueError(msg)
            group_means = np.split(self.mean, np.cumsum(dims)[:-1], axis=0)

        # Split b into groups
        group_bs = None
        if self.b is not None:
            if np.sum(dims) != len(self.b):
                msg = "The sum of dims must equal the length of b."
                raise ValueError(msg)
            group_bs = np.split(self.b, np.cumsum(dims)[:-1], axis=0)

        return group_means, group_bs

    def compute_mean(
        self,
        in_place: bool = True,
    ) -> np.ndarray | None:
        """
        Compute the posterior mean of the observation precision parameters, a / b.

        Parameters
        ----------
        in_place
            If ``True``, compute the posterior mean in place.
            If ``False``, compute the posterior mean and return as a new
            `ndarray` without modifying self. Defaults to ``True``.

        Returns
        -------
        ndarray | None
            `ndarray` of `float`, shape ``(y_dim,)``.
            Posterior mean of the observation precision parameters.
        """
        if in_place:
            if self.mean is None:
                self.mean = np.zeros_like(self.b)
            self.mean[:] = self.a / self.b
            return None

        return self.a / self.b


class GFAParams:
    """
    A class for group factor analysis (GFA) model parameters.

    Parameters
    ----------
    x_dim
        Number of latent dimensions.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group.
    X
        Posterior latents.
    C
        Posterior loadings.
    alpha
        Posterior ARD parameters.
    d
        Posterior observation mean parameters.
    phi
        Posterior observation precision parameters.

    Attributes
    ----------
    x_dim
        Same as **x_dim**, above.
    y_dims
        Same as **y_dims**, above.
    X
        Same as **X**, above.
    C
        Same as **C**, above.
    alpha
        Same as **alpha**, above.
    d
        Same as **d**, above.
    phi
        Same as **phi**, above.

    Raises
    ------
    TypeError
        If any attribute is not of the respective type specified above.
    """

    def __init__(
        self,
        x_dim: int | None = None,
        y_dims: np.ndarray | None = None,
        X: PosteriorLatent | None = None,
        C: PosteriorLoading | None = None,
        alpha: PosteriorARD | None = None,
        d: PosteriorObsMean | None = None,
        phi: PosteriorObsPrec | None = None,
    ):
        # Latent dimensionality
        if x_dim is not None and not isinstance(x_dim, int):
            msg = "x_dim must be an integer."
            raise TypeError(msg)
        self.x_dim = x_dim

        # Observed dimensionalities
        if y_dims is not None and not isinstance(y_dims, np.ndarray):
            msg = "y_dims must be a numpy.ndarray of integers."
            raise TypeError(msg)
        self.y_dims = y_dims

        # Latents
        if X is None:
            self.X = PosteriorLatent()
        elif not isinstance(X, PosteriorLatent):
            msg = "X must be a PosteriorLatent object."
            raise TypeError(msg)
        else:
            self.X = X

        # Loadings
        if C is None:
            self.C = PosteriorLoading()
        elif not isinstance(C, PosteriorLoading):
            msg = "C must be a PosteriorLoading object."
            raise TypeError(msg)
        else:
            self.C = C

        # ARD parameters
        if alpha is None:
            self.alpha = PosteriorARD()
        elif not isinstance(alpha, PosteriorARD):
            msg = "alpha must be a PosteriorARD object."
            raise TypeError(msg)
        else:
            self.alpha = alpha

        # Observation mean parameters
        if d is None:
            self.d = PosteriorObsMean()
        elif not isinstance(d, PosteriorObsMean):
            msg = "d must be a PosteriorObsMean object."
            raise TypeError(msg)
        else:
            self.d = d

        # Observation precision parameters
        if phi is None:
            self.phi = PosteriorObsPrec()
        elif not isinstance(phi, PosteriorObsPrec):
            msg = "phi must be a PosteriorObsPrec object."
            raise TypeError(msg)
        else:
            self.phi = phi

    def __repr__(self) -> str:
        return (
            f"GFAParams(x_dim={self.x_dim}, "
            f"y_dims={self.y_dims}, "
            f"X={self.X}, "
            f"C={self.C}, "
            f"alpha={self.alpha}, "
            f"d={self.d}, "
            f"phi={self.phi})"
        )

    def is_initialized(self) -> bool:
        """
        Check if GFA parameters have been initialized to data.

        Returns
        -------
        bool
            ``True`` if all GFA parameters have been initialized to data.
        """
        return (
            self.x_dim is not None
            and self.y_dims is not None
            and self.X.mean is not None
            and self.X.cov is not None
            and self.C.mean is not None
            and self.C.moment is not None
            and self.alpha.a is not None
            and self.alpha.b is not None
            and self.alpha.mean is not None
            and self.d.mean is not None
            and self.d.cov is not None
            and self.phi.a is not None
            and self.phi.b is not None
            and self.phi.mean is not None
        )

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> GFAParams | None:
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
            dimensions to a new ``GFAParams``, and return that new
            ``GFAParams``. Defaults to ``True``.

        Returns
        -------
        GFAParams | None
            A new ``GFAParams`` object whose parameters have only the specified
            latent dimensions.
        """
        if in_place:
            # Keep only the specified dimensions
            self.x_dim = len(dims)
            self.X.get_subset_dims(dims, in_place=True)
            self.C.get_subset_dims(dims, in_place=True)
            self.alpha.get_subset_dims(dims, in_place=True)
            return None

        # Copy over parameters with the relevant subsets of dimensions to a
        # new GFAParams object, and return that new GFAParams object.
        return self.__class__(
            x_dim=len(dims),
            y_dims=self.y_dims.copy(),
            X=self.X.get_subset_dims(dims, in_place=False),
            C=self.C.get_subset_dims(dims, in_place=False),
            alpha=self.alpha.get_subset_dims(dims, in_place=False),
            d=self.d.copy(),
            phi=self.phi.copy(),
        )

    def copy(self) -> GFAParams:
        """
        Return a copy of self.

        Returns
        -------
        GFAParams
            A copy of self.
        """
        return self.__class__(
            x_dim=self.x_dim,
            y_dims=self.y_dims.copy(),
            X=self.X.copy(),
            C=self.C.copy(),
            alpha=self.alpha.copy(),
            d=self.d.copy(),
            phi=self.phi.copy(),
        )


class HyperPriorParams(ArrayContainer):
    """
    A class for GFA prior hyperparameters.

    Parameters
    ----------
    d_beta : float > 0
        Precision of the observation mean prior.
    a_alpha : float > 0
        Shape parameter of the ARD prior.
    b_alpha : float > 0
        Rate parameter of the ARD prior.
    a_phi : float > 0
        Shape parameter of the observation precision prior.
    b_phi : float > 0
        Rate parameter of the observation precision prior.

    Attributes
    ----------
    d_beta
        Same as **d_beta**, above.
    a_alpha
        Same as **a_alpha**, above.
    b_alpha
        Same as **b_alpha**, above.
    a_phi
        Same as **a_phi**, above.
    b_phi
        Same as **b_phi**, above.
    """

    def __init__(
        self,
        d_beta: float = 1e-12,
        a_alpha: float = 1e-12,
        b_alpha: float = 1e-12,
        a_phi: float = 1e-12,
        b_phi: float = 1e-12,
    ):
        self.d_beta = d_beta
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_phi = a_phi
        self.b_phi = b_phi


class GFAFitTracker(ArrayContainer):
    """
    A class for quantities tracked during a GFA model fit.

    Parameters
    ----------
    lb
        `ndarray` of `float`, shape ``(num_iter,)``.
        Variational lower bound at each iteration.
    iter_time
        `ndarray` of `float`, shape ``(num_iter,)``.
        Runtime on each iteration.

    Attributes
    ----------
    lb
        Same as **lb**, above.
    iter_time
        Same as **iter_time**, above.
    """

    def __init__(
        self,
        lb: np.ndarray | None = None,
        iter_time: np.ndarray | None = None,
    ):
        self.lb = lb
        self.iter_time = iter_time

    def plot_fit_progress(self) -> None:
        """Plot the variational lower bound and runtime at each iteration."""
        if self.lb is not None and self.iter_time is not None:
            # create figure
            fig, (ax_lb, ax_rt) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

            # Plot the lower bound. It should be monotonically increasing.
            ax_lb.plot(self.lb, color="black", linestyle="solid", linewidth=1.0)
            ax_lb.set_xlabel("Iteration")
            ax_lb.set_ylabel("Lower bound")
            ax_lb.spines["top"].set_visible(False)
            ax_lb.spines["right"].set_visible(False)

            # Plot cumulative runtime.
            ax_rt.plot(
                np.cumsum(self.iter_time),
                color="black",
                linestyle="solid",
                linewidth=1.0,
            )
            ax_rt.set_xlabel("Iteration")
            ax_rt.set_ylabel("Cumulative runtime (s)")
            ax_rt.spines["top"].set_visible(False)
            ax_rt.spines["right"].set_visible(False)

            fig.tight_layout()
            plt.show()
        else:
            print("No fitting progress to plot.")


class GFAFitFlags:
    """
    A class for status messages during a GFA model fit.

    Parameters
    ----------
    converged
        ``True`` if the lower bound converged before reaching ``max_iter``
        iterations.
    decreasing_lb
        ``True`` if lower bound decreased during fitting.
    private_var_floor
        ``True`` if the private variance floor was used on any values of
        ``phi``.
    x_dims_removed
        Number latent dimensions removed (if ``prune_X`` was ``True``) due to
        low variance.

    Attributes
    ----------
    converged
        Same as **converged**, above.
    decreasing_lb
        Same as **decreasing_lb**, above.
    private_var_floor
        Same as **private_var_floor**, above.
    x_dims_removed
        Same as **x_dims_removed**, above.
    """

    def __init__(
        self,
        converged: bool = False,
        decreasing_lb: bool = False,
        private_var_floor: bool = False,
        x_dims_removed: int = 0,
    ):
        self.converged = converged
        self.decreasing_lb = decreasing_lb
        self.private_var_floor = private_var_floor
        self.x_dims_removed = x_dims_removed

    def __repr__(self) -> str:
        return (
            f"GFAFitFlags("
            f"converged={self.converged}, "
            f"decreasing_lb={self.decreasing_lb}, "
            f"private_var_floor={self.private_var_floor}, "
            f"x_dims_removed={self.x_dims_removed})"
        )

    def display(self) -> None:
        """Print out the fit flags."""
        print(f"Converged: {self.converged}")
        print(f"Decreasing lower bound: {self.decreasing_lb}")
        print(f"Private variance floor: {self.private_var_floor}")
        print(f"Latent dimensions removed: {self.x_dims_removed}")


class GFAFitArgs:
    """
    A class for keyword arguments used to fit a GFA model.

    Parameters
    ----------
    **kwargs
        Keyword arguments that match any or all of the attributes below.

    Attributes
    ----------
    x_dim_init : int
        Initial number of latent dimensions to fit (before pruning).
        Defaults to ``1``.
    hyper_priors : HyperPriorParams
        Hyperparameters of the GFA prior distributions.
    fit_tol : float
        Tolerance for convergence. Defaults to ``1e-8``.
    max_iter : int
        Maximum number of iterations. Defaults to ``1e6``.
    verbose : bool
        Specifies whether to display progress information. Defaults to
        ``False``.
    random_seed : int
        Seed the random number generator for reproducibility. Defaults to
        ``None``.
    min_var_frac : float
        Fraction of overall data variance for each observed dimension to set
        as the private variance floor. Defaults to ``0.001``.
    prune_X : bool
        Set to ``True`` to remove latents that become inactive. Can speed up
        runtime and improve memory efficiency. Defaults to ``True``.
    prune_tol : float
        Tolerance for pruning. Sample second moment of each latent must
        remain larger than this value to remain in the model.
        Defaults to ``1e-7``.
    save_X : bool
        Set to ``True`` to save posterior estimates of latent variables
        :math:`X`. For large datasets, ``X.mean`` may be very large. Defaults
        to ``False``.
    save_C_cov : bool
        Set to true to save posterior covariance of :math:`C`. For large
        ``y_dim`` and ``x_dim``, these structures can use a lot of memory.
        Defaults to ``False``.
    save_fit_progress : bool
        Set to ``True`` to save the lower bound and runtime at each iteration.
        Defaults to ``True``.
    """

    DEFAULT_ARGS = {
        "x_dim_init": 1,
        "hyper_priors": HyperPriorParams(),
        "fit_tol": 1e-8,
        "max_iter": int(1e6),
        "verbose": False,
        "random_seed": None,
        "min_var_frac": 0.001,
        "prune_X": True,
        "prune_tol": 1e-7,
        "save_X": False,
        "save_C_cov": False,
        "save_fit_progress": True,
    }
    """
    Valid keyword arguments and their defaults.
    """

    def __init__(self, **kwargs):
        # Always initialize with default arguments
        self.set_default_args()
        # Then, the user can override any or all of them
        if kwargs is not None:
            self.set_args(**kwargs)

    def __repr__(self) -> str:
        return (
            f"GFAFitArgs(x_dim_init={self.x_dim_init}, "
            f"hyper_priors={self.hyper_priors}, "
            f"fit_tol={self.fit_tol}, "
            f"max_iter={self.max_iter}, "
            f"verbose={self.verbose}, "
            f"random_seed={self.random_seed}, "
            f"min_var_frac={self.min_var_frac}, "
            f"prune_X={self.prune_X}, "
            f"prune_tol={self.prune_tol}, "
            f"save_X={self.save_X}, "
            f"save_C_cov={self.save_C_cov}, "
            f"save_fit_progress={self.save_fit_progress})"
        )

    def get_args(self) -> dict:
        """
        Return a dictionary containing current keyword arguments.

        Returns
        -------
        dict
            Dictionary containing current keyword arguments.
        """
        return vars(self)

    def set_args(self, **kwargs) -> None:
        """
        Set keyword arguments from a dictionary.

        Parameters
        ----------
        **kwargs
            User-specified keyword arguments.

        Raises
        ------
        ValueError
            If any keyword argument is not valid.
        """
        for key, value in kwargs.items():
            if key in self.DEFAULT_ARGS:
                setattr(self, key, value)
            else:
                msg = f"Invalid keyword argument: {key}"
                raise ValueError(msg)

    @classmethod
    def get_default_args(cls) -> dict:
        """
        Return a dictionary containing default keyword arguments.

        Returns
        -------
        dict
            Dictionary containing default keyword arguments.
        """
        return cls.DEFAULT_ARGS

    def set_default_args(self) -> None:
        """Set keyword arguments to default values."""
        self.set_args(**self.DEFAULT_ARGS)

    def display(self) -> None:
        """Print out the current keyword arguments."""
        for key, value in self.get_args().items():
            print(f"{key}: {value}")
