"""
Building blocks for observation models with probabilistic model components.

Build probabilistic observation models and compute descriptive statistics from their
parameters.

**Classes**

- :class:`PosteriorLoading` -- Posterior estimates of loading matrices.
- :class:`PosteriorARD` -- Posterior estimates of ARD parameters.
- :class:`PosteriorObsMean` -- Posterior estimates of mean parameters.
- :class:`PosteriorObsPrec` -- Posterior estimates of precision parameters.
- :class:`HyperPriorParams` -- Prior hyperparameters.
- :class:`ObsParamsARD` -- A generic observation model with ARD.

"""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from latents.base import ArrayContainer


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


class HyperPriorParams(ArrayContainer):
    """
    A class for prior hyperparameters.

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


class ObsParamsARD:
    """
    A generic observation model with automatic relevance determination (ARD).

    Parameters
    ----------
    x_dim
        Number of latent dimensions.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group.
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
            f"{type(self).__name__}(x_dim={self.x_dim}, "
            f"y_dims={self.y_dims}, "
            f"C={self.C}, "
            f"alpha={self.alpha}, "
            f"d={self.d}, "
            f"phi={self.phi})"
        )

    @classmethod
    def generate(
        cls,
        y_dims: np.ndarray,
        x_dim: int,
        hyper_priors: HyperPriorParams,
        snr: np.ndarray,
        rng: np.random.Generator,
    ) -> ObsParamsARD:
        """
        Randomly generate a set of observation model parameters.

        Parameters
        ----------
        y_dims
            `ndarray` of `int`, shape ``(num_groups,)``.
            Dimensionalities of each observed group.
        x_dim
            Number of latent dimensions.
        hyper_priors
            Hyperparameters of the GFA prior distributions.
            Note that ``hyper_priors.a_alpha`` and ``hyper_priors.b_alpha`` can be
            abused here, so that they can be used to specify group- and
            column-specific sparsity patterns in the loadings matrices.
            In that case, specify both of them as `ndarray` of shape
            ``(num_groups, x_dim)``.
        snr
            `ndarray` of `float`, shape ``(num_groups,)``.
            Signal-to-noise ratios of each group.
        rng
            A random number generator object.

        Returns
        -------
        ObsParamsARD
            Generated observation model parameters.
        """
        # Number of observed groups
        num_groups = len(y_dims)

        # Initialize the observation model parameters
        obs_params = cls(x_dim=x_dim, y_dims=y_dims)

        # Initialize ARD parameters
        obs_params.alpha.mean = np.zeros((num_groups, x_dim))
        if isinstance(hyper_priors.a_alpha, float):
            # Repeat the ARD hyperparameters for each group and column
            a_alpha = hyper_priors.a_alpha * np.ones((num_groups, x_dim))
            b_alpha = hyper_priors.b_alpha * np.ones((num_groups, x_dim))
        else:
            # Use the ARD hyperparameters specified by the user
            a_alpha = hyper_priors.a_alpha
            b_alpha = hyper_priors.b_alpha

        # Generate observation mean parameters
        obs_params.d.mean = rng.normal(
            0, 1 / np.sqrt(hyper_priors.d_beta), size=(y_dims.sum())
        )

        # Generate observation precision parameters
        obs_params.phi.mean = rng.gamma(
            shape=hyper_priors.a_phi, scale=1 / hyper_priors.b_phi, size=(y_dims.sum())
        )
        # Split phi according to observed groups, so we can use it below
        phis, _ = obs_params.phi.get_groups(y_dims)

        # Generate group-specific parameters and observed data
        obs_params.C.mean = np.zeros((y_dims.sum(), x_dim))
        Cs, _, _ = obs_params.C.get_groups(y_dims)
        for group_idx in range(num_groups):
            # Generate each ARD parameter and the corresponding column of the
            # loadings matrix for the current group
            for x_idx in range(x_dim):
                # Generate ARD parameters
                obs_params.alpha.mean[group_idx, x_idx] = rng.gamma(
                    shape=a_alpha[group_idx, x_idx], scale=1 / b_alpha[group_idx, x_idx]
                )
                Cs[group_idx][:, x_idx] = rng.normal(
                    0,
                    1 / np.sqrt(obs_params.alpha.mean[group_idx, x_idx]),
                    size=(y_dims[group_idx]),
                )

            # Enforce the desired signal-to-noise ratios
            var_CC = np.sum(Cs[group_idx] ** 2)
            var_noise_desired = var_CC / snr[group_idx]
            var_noise_current = np.sum(1 / phis[group_idx])
            phis[group_idx] *= var_noise_current / var_noise_desired

        return obs_params

    def is_initialized(self) -> bool:
        """
        Check if observation model parameters have been initialized to data.

        Returns
        -------
        bool
            ``True`` if all parameters have been initialized to data.
        """
        return (
            self.x_dim is not None
            and self.y_dims is not None
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
    ) -> ObsParamsARD | None:
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
            dimensions to a new ``ObsParamsARD``, and return that new
            ``ObsParamsARD``. Defaults to ``True``.

        Returns
        -------
        ObsParamsARD | None
            A new ``ObsParamsARD`` object whose parameters have only the specified
            latent dimensions.
        """
        if in_place:
            # Keep only the specified dimensions
            self.x_dim = len(dims)
            self.C.get_subset_dims(dims, in_place=True)
            self.alpha.get_subset_dims(dims, in_place=True)
            return None

        # Copy over parameters with the relevant subsets of dimensions to a
        # new ObsParamsARD object, and return that new ObsParamsARD object.
        return self.__class__(
            x_dim=len(dims),
            y_dims=self.y_dims.copy(),
            C=self.C.get_subset_dims(dims, in_place=False),
            alpha=self.alpha.get_subset_dims(dims, in_place=False),
            d=self.d.copy(),
            phi=self.phi.copy(),
        )

    def copy(self) -> ObsParamsARD:
        """
        Return a copy of self.

        Returns
        -------
        ObsParamsARD
            A copy of self.
        """
        return self.__class__(
            x_dim=self.x_dim,
            y_dims=self.y_dims.copy(),
            C=self.C.copy(),
            alpha=self.alpha.copy(),
            d=self.d.copy(),
            phi=self.phi.copy(),
        )

    def compute_snr(
        self,
        dims: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the signal-to-noise ratio (SNR) of specified observation groups.

        Compute the SNR of each observed group, according to fitted model parameters.

        Parameters
        ----------
        dims
            `ndarray` of `int`, shape ``(num_groups,)``.
            Dimensionalities of the observed groups of interest.
            Defaults to ``self.y_dims``.

        Returns
        -------
        ndarray
            `ndarray` of `float`, shape ``(num_groups,)``.
            Signal-to-noise ratio of each observed group.
        """
        if dims is None:
            dims = self.y_dims

        # Get views of the loadings matrix and the observation precisions for
        # each group
        _, _, C_moments = self.C.get_groups(dims)
        phi_means, _ = self.phi.get_groups(dims)

        # Compute the SNR for each group
        return np.array(
            [
                np.trace(np.sum(C_moments[group_idx], axis=0))
                / np.sum(1 / phi_means[group_idx])
                for group_idx in range(len(dims))
            ]
        )

    @staticmethod
    def get_dim_types(num_groups: int) -> np.ndarray:
        """
        Generate all dimension types for a given number of groups.

        Generate an array with all types of dimensions (singlet, doublet, triplet,
        global, etc.) for a given number of groups.

        Parameters
        ----------
        num_groups
            Number of observed groups.

        Returns
        -------
        dim_types : ndarray
            `ndarray` of `bool`, shape ``(num_groups, num_dim_types)``.
            ``dim_types[:,j]`` is a Boolean vector indicating the structure of
            dimension type ``j``. ``1`` indicates that a group is involved, ``0``
            otherwise.
        """
        num_dim_types = 2**num_groups  # Number of dimension types
        dim_types = np.empty((num_groups, num_dim_types))

        for dim_idx in range(num_dim_types):
            dim_str = format(dim_idx, f"0{num_groups}b")
            dim_types[:, dim_idx] = np.array([int(b) for b in dim_str], dtype=bool)

        return dim_types

    def compute_dimensionalities(
        self,
        cutoff_shared_var: float = 0.02,
        cutoff_snr: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute dimensionalities.

        Compute the number of each possible type of dimension, along with the
        shared variance in each group explained by each type of dimension.

        Parameters
        ----------
        cutoff_shared_var
            Minimum fraction of shared variance within a group that must be
            explained by a latent to be considered significant. Defaults to
            ``0.02``.
        cutoff_snr
            Minimum signal-to-noise ratio (SNR) that a group must have for ANY
            latents to be considered significant. Defaults to ``0.001``.

        Returns
        -------
        num_dim : ndarray
            `ndarray` of `int`, shape ``(num_dim_types,)``.
            The number of each type of dimension. ``num_dim[i]`` corresponds to
            the dimension type in ``dim_types[:,i]``.
        sig_dims : ndarray
            `ndarray` of `bool`, shape ``(num_groups, x_dim)``.
            ``sig_dims[i,j]`` is ``True`` if latent ``j`` explains a significant
            fraction of the shared variance within group ``i``.
        var_exp : ndarray
            `ndarray` of `float`, shape ``(num_groups, num_dim_types)``.
            ``var_exp[i,j]`` is the fraction of the shared variance within group
            ``i`` that is explained by dimension type ``j``. ``var_exp[:,j]``
            corresponds to the dimension type in ``dim_types[:,j]``.
        dim_types : ndarray
            `ndarray` of `bool`, shape ``(num_groups, num_dim_types)``.
            ``dim_types[:,j]`` is a Boolean vector indicating the structure of
            dimension type ``j``. ``1`` indicates that a group is involved, ``0``
            otherwise.
        """
        num_groups = len(self.y_dims)  # Number of observed groups

        # Determine all dimension types
        dim_types = self.get_dim_types(num_groups)
        num_dim_types = dim_types.shape[1]

        # Compute signal-to-noise ratios
        snr = self.compute_snr()

        # Relative shared variance explained by each dimension
        alpha_inv = 1 / self.alpha.mean
        alpha_inv_rel = alpha_inv / np.sum(alpha_inv, axis=1, keepdims=True)

        # Take dimensions only if shared variance and SNR are above cutoffs
        sig_dims = (alpha_inv_rel > cutoff_shared_var) & (snr > cutoff_snr)[
            :, np.newaxis
        ]
        num_dim = np.zeros(num_dim_types)
        var_exp = np.zeros((num_groups, num_dim_types))
        for dim_idx in range(num_dim_types):
            dims = np.all(sig_dims == dim_types[:, dim_idx, np.newaxis], axis=0)
            num_dim[dim_idx] = np.sum(dims)
            var_exp[:, dim_idx] = np.sum(alpha_inv_rel[:, dims], axis=1)

        return num_dim, sig_dims, var_exp, dim_types

    @staticmethod
    def compute_dims_pairs(
        num_dim: np.ndarray,
        dim_types: np.ndarray,
        var_exp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze the shared dimensionalities and variances between pairs of groups.

        Compute the total dimensionality of each group, and the shared
        dimensionality between each pair of groups given the dimensionalities and
        types of dimensions given by ``num_dim`` and ``dim_types``, respectively.
        Compute also the shared variance explained by a pairwise interaction in
        each group.

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
        var_exp
            `ndarray` of `float`, shape ``(num_groups, num_dim_types)``.
            ``var_exp[i,j]`` is the fraction of the shared variance within group
            ``i`` that is explained by dimension type ``j``. ``var_exp[:,j]``
            corresponds to the dimension type in ``dim_types[:,j]``.

        Returns
        -------
        pair_dims : ndarray
            `ndarray` of `int`, shape ``(num_pairs, 3)``.

            ``pair_dims[i,0]`` -- total dimensionality of group ``1`` in pair ``i``.

            ``pair_dims[i,1]`` -- shared dimensionality between pair ``i``.

            ``pair_dims[i,2]`` -- total dimensionality of group ``2`` in pair ``i``.
        pair_var_exp : ndarray
            `ndarray` of `float`, shape ``(num_pairs, 2)``.

            ``pair_var_exp[i,0]`` -- shared variance explained by pairwise
            interaction ``i`` in group ``1``.

            ``pair_var_exp[i,1]`` -- shared variance explained by pairwise
            interaction ``i`` in group ``2``.
        pairs : ndarray
            `ndarray` of `int`, shape ``(num_pairs, 2)``.

            ``pairs[i,0]`` -- index of group ``1`` in pair ``i``.

            ``pairs[i,1]`` -- index of group ``2`` in pair ``i``.
        """
        num_groups = dim_types.shape[0]  # Number of observed groups

        # For each group, create a list of all dimension types that involve that
        # group
        group_idxs = [np.nonzero(dim_types[g, :])[0] for g in range(num_groups)]

        # Create a list of all possible pairs
        pairs = list(combinations(range(num_groups), 2))
        num_pairs = len(pairs)

        # Count the number of each type of dimension
        pair_dims = np.zeros((num_pairs, 3), dtype=int)
        pair_var_exp = np.zeros((num_pairs, 2))
        for pair_idx, pair in enumerate(pairs):
            # Total dimensionality of each group in the pair
            pair_dims[pair_idx, 0] = num_dim[group_idxs[pair[0]]].sum()
            pair_dims[pair_idx, 2] = num_dim[group_idxs[pair[1]]].sum()

            # Shared dimensionality between the two groups
            shared_idxs = np.intersect1d(group_idxs[pair[0]], group_idxs[pair[1]])
            pair_dims[pair_idx, 1] = num_dim[shared_idxs].sum()

            # Shared variance explained by a pairwise interaction in each group
            pair_var_exp[pair_idx, 0] = var_exp[pair[0], shared_idxs].sum()
            pair_var_exp[pair_idx, 1] = var_exp[pair[1], shared_idxs].sum()

        return pair_dims, pair_var_exp, np.array(pairs)

    @staticmethod
    def plot_dimensionalities(
        num_dim: np.ndarray,
        dim_types: np.ndarray,
        sem_dim: np.ndarray | None = None,
        group_names: list[str] | None = None,
        plot_zero_dim: bool = False,
        ax: Axes | None = None,
    ) -> None:
        """
        Plot the number of each dimension type (given by ``dim_types``) in ``num_dim``.

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
            group_names = [f"{i + 1}" for i in range(num_groups)]
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
                np.arange(1, num_dim_types + 1),
                num_dim[sort_idxs],
                yerr=sem_dim[sort_idxs],
            )
        ax.set_xlabel("Dimension type")
        ax.set_ylabel("Dimensionality")
        ax.set_xticks(np.arange(1, num_dim_types + 1))
        ax.set_xticklabels([xticklbls[i] for i in sort_idxs])

        # Adjust appearance of axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @staticmethod
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
            group_names = [f"{i + 1}" for i in range(num_groups)]
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
            plt.xticks(
                np.arange(1, num_dim_types + 1), [xticklbls[i] for i in sort_idxs]
            )
            plt.title(f"Group {group_names[group_idx]}")

            # Adjust appearance of axes
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

        # Make sure the subplots don't overlap
        fig.tight_layout()

    @staticmethod
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
            group_names = [f"{i + 1}" for i in range(num_groups)]
        xticklbls = np.full((num_pairs, 3), "", dtype=object)
        for pair_idx in range(num_pairs):
            # Total in group 1
            xticklbls[pair_idx, 0] = f"Total, {group_names[pairs[pair_idx, 0]]}"
            # Shared between both groups
            xticklbls[pair_idx, 1] = (
                f"{group_names[pairs[pair_idx, 0]]}-{group_names[pairs[pair_idx, 1]]}"
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

    @staticmethod
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
            group_names = [f"{i + 1}" for i in range(num_groups)]

        pairlbls = np.array(
            [
                f"{group_names[pairs[i, 0]]}" + f"-{group_names[pairs[i, 1]]}"
                for i in range(num_pairs)
            ]
        )

        # Plot pairwise shared variances
        for pair_idx in range(num_pairs):
            plt.subplot(1, num_pairs, pair_idx + 1)
            if sem_pair_var_exp is None:
                plt.bar(
                    np.arange(1, pair_var_exp.shape[1] + 1), pair_var_exp[pair_idx, :]
                )
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
