"""Custom data types used throughout the GFA subpackage."""

from __future__ import annotations

import numpy as np

from latents.base import (
    FitFlags,
    FitTracker,
)
from latents.observation_model.probabilistic import (
    HyperPriorParams,
    ObsParamsARD,
)
from latents.state_model.latents import (
    StateParamsStatic,
)


class GFAParams:
    """
    Group factor analysis (GFA) model parameters.

    Parameters
    ----------
    x_dim
        Number of latent dimensions.
    y_dims
        `ndarray` of `int`, shape ``(num_groups,)``.
        Dimensionalities of each observed group.

    Attributes
    ----------
    obs_params : ObsParamsARD
        Posterior observation parameters.
    state_params : StateParamsStatic
        Posterior state parameters.
    """

    def __init__(
        self,
        x_dim: int | None = None,
        y_dims: np.ndarray | None = None,
    ):
        # Latent dimensionality
        if x_dim is not None and not isinstance(x_dim, int):
            msg = "x_dim must be an integer."
            raise TypeError(msg)

        # Observed dimensionalities
        if y_dims is not None and not isinstance(y_dims, np.ndarray):
            msg = "y_dims must be a numpy.ndarray of integers."
            raise TypeError(msg)

        # Observation model parameters
        self.obs_params = ObsParamsARD(x_dim=x_dim, y_dims=y_dims)

        # State model parameters
        self.state_params = StateParamsStatic(x_dim=x_dim)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(obs_params={self.obs_params}, "
            f"state_params={self.state_params})"
        )

    def is_initialized(self) -> bool:
        """
        Check if GFA parameters have been initialized to data.

        Returns
        -------
        bool
            ``True`` if all GFA parameters have been initialized to data.
        """
        return self.obs_params.is_initialized() and self.state_params.is_initialized()

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
            self.obs_params.get_subset_dims(dims, in_place=True)
            self.state_params.get_subset_dims(dims, in_place=True)
            return None

        # Copy over parameters with the relevant subsets of dimensions to a
        # new GFAParams object, and return that new GFAParams object.
        return self.__class__(
            obs_params=self.obs_params.get_subset_dims(dims, in_place=False),
            state_params=self.state_params.get_subset_dims(dims, in_place=False),
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
            obs_params=self.obs_params.copy(),
            state_params=self.state_params.copy(),
        )


class GFAFitTracker(FitTracker):
    """
    Quantities tracked during a GFA model fit.

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

    pass


class GFAFitFlags(FitFlags):
    """
    Status messages during a GFA model fit.

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
        super().__init__(converged, decreasing_lb, private_var_floor)
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
        super().display()
        print(f"Latent dimensions removed: {self.x_dims_removed}")


class GFAFitArgs:
    """
    Keyword arguments used to fit a GFA model.

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
