"""
Custom data types used throughout the mDLAG subpackage.

**Classes**

- :class:`mDLAGParams` -- mDLAG model parameters.
- :class:`mDLAGFitTracker` -- Quantities tracked during a mDLAG model fit.
- :class:`mDLAGFitFlags` -- Status messages during a mDLAG model fit.
- :class:`mDLAGFitArgs` -- Keyword arguments used to fit a mDLAG model.

"""

from __future__ import annotations

import numpy as np

from latents.base import FitFlags, FitTracker
from latents.mdlag.gp.fit_config import GPFitConfig
from latents.mdlag.gp.gp_model import mDLAGGP
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD
from latents.state_model.latents import StateParamsDelayed


class mDLAGParams:
    """Delayed latents across multiple groups (mDLAG) model parameters.

    Parameters
    ----------
    x_dim
        Number of latent dimensions. Defaults to ``None``.
    y_dims
        1D array of integers specifying the dimensionality of each group.
        Defaults to ``None``.
    T
        Number of timepoints. Defaults to ``None``.
    gp_params_init
        Initial Gaussian process parameters. If not provided, an empty mDLAGGP
        object will be created. Defaults to ``None``.
    save_X_cov
        Whether to save the covariance of the latent variables. Defaults to ``False``.
    save_C_cov
        Whether to save the covariance of the observation model parameters.
        Defaults to ``False``.

    Attributes
    ----------
    obs_params
        Observation model parameters.
    state_params
        State model parameters.
    gp
        Gaussian process.
    T
        Number of timepoints.
    save_X_cov
        Whether to save the covariance of the latent variables.
    save_C_cov
        Whether to save the covariance of the observation model parameters.

    Raises
    ------
    TypeError
        If ``y_dims`` is not a numpy.ndarray.
    """

    def __init__(
        self,
        x_dim: int | None = None,
        y_dims: np.ndarray | None = None,
        T: int | None = None,
        gp_init: mDLAGGP | None = None,
        save_X_cov: bool = False,
        save_C_cov: bool = False,
    ):
        # Latent dimensionality
        if x_dim is not None and not isinstance(x_dim, int):
            msg = "x_dim must be an integer."
            raise TypeError(msg)

        # Observed dimensionalities
        if y_dims is not None and not isinstance(y_dims, np.ndarray):
            msg = "y_dims must be a numpy.ndarray of integers."
            raise TypeError(msg)

        # Calculate number of groups
        num_groups = len(y_dims) if y_dims is not None else 0

        # Observation model parameters:
        self.obs_params = ObsParamsARD(x_dim=x_dim, y_dims=y_dims)

        # State model parameters:
        self.state_params = StateParamsDelayed(x_dim, num_groups, T, X=None)

        # GP parameters:
        if gp_init is None:
            # Create empty mDLAGGP with default parameters
            # We'll create a placeholder that will be properly initialized later
            self.gp = None
        else:
            self.gp = gp_init

        self.T = T
        self.save_X_cov = save_X_cov
        self.save_C_cov = save_C_cov

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(obs_params={self.obs_params}, "
            f"state_params={self.state_params},"
            f"gp_params={self.gp})"
        )

    def is_initialized(self) -> bool:
        """Check if all model parameters have been initialized to data.

        Returns
        -------
        bool
            ``True`` if all model parameters have been initialized to data.
        """
        return (
            self.obs_params.is_initialized()
            and self.state_params.is_initialized()
            and (self.gp is not None)
        )

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> mDLAGParams | None:
        """Keep only a subset of the latent dimensions in each relevant parameter."""
        if in_place:
            self.obs_params.get_subset_dims(dims, in_place=True)
            self.state_params.get_subset_dims(dims, in_place=True)
            self.gp.get_subset_dims(dims, in_place=True)
            return None

        subset_obs_params = self.obs_params.get_subset_dims(dims, in_place=False)
        subset_state_params = self.state_params.get_subset_dims(dims, in_place=False)
        subset_gp = self.gp.get_subset_dims(dims, in_place=False)
        subset_params = self.__class__(
            x_dim=len(dims),
            y_dims=self.obs_params.y_dims.copy(),
            T=self.T,
            gp_init=subset_gp,
            save_X_cov=self.save_X_cov,
            save_C_cov=self.save_C_cov,
        )
        subset_params.obs_params = subset_obs_params
        subset_params.state_params = subset_state_params
        return subset_params

    def copy(self) -> mDLAGParams:
        """Return a copy of self."""
        raise NotImplementedError


class mDLAGFitTracker(FitTracker):
    """
    Quantities tracked during a mDLAG model fit.

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


class mDLAGFitFlags(FitFlags):
    """
    Status messages during a mDLAG model fit.

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


class mDLAGFitArgs:
    """
    Keyword arguments used to fit a mDLAG model.

    Parameters
    ----------
    **kwargs
        Keyword arguments that match any or all of the attributes below.

    Attributes
    ----------
    gp_fit_config : GPFitConfig
        Configuration for Gaussian process fitting. Defaults to ``GPFitConfig()``.
    hyper_priors : HyperPriorParams
        Hyperparameters of the mDLAG prior distributions.
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
    prune_X : bool
        Set to ``True`` to remove latents that become inactive. Can speed up
        runtime and improve memory efficiency. Defaults to ``True``.
    prune_tol : float
        Tolerance for pruning inactive latents. Defaults to ``1e-7``.
    save_X_cov : bool
        Set to ``True`` to save posterior covariance of :math:`X`. For large
        datasets, this matrix can use a lot of memory. Defaults to ``False``.
    save_C_cov : bool
        Set to ``True`` to save posterior covariance of :math:`C`. For large
        ``y_dim`` and ``x_dim``, these structures can use a lot of memory.
        Defaults to ``False``.
    save_fit_progress : bool
        Set to ``True`` to save the lower bound and runtime at each iteration.
        Defaults to ``True``.
    checkpoint_interval : int
        Number of iterations between checkpoint saves. Set to ``0`` to disable
        checkpointing. Defaults to ``0``.
    checkpoint_dir : str
        Directory path where checkpoint files will be saved. Defaults to
        ``"checkpoints"``.
    """

    DEFAULT_ARGS = {
        "gp_fit_config": GPFitConfig(),
        "hyper_priors": HyperPriorParams(),
        "fit_tol": 1e-8,
        "max_iter": int(1e6),
        "verbose": False,
        "random_seed": None,
        "prune_X": True,
        "prune_tol": 1e-7,
        "save_X_cov": False,
        "save_C_cov": False,
        "save_fit_progress": True,
        "checkpoint_interval": 0,
        "checkpoint_dir": "checkpoints",
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
            f"mDLAGFitArgs(gp_fit_config={self.gp_fit_config}, "
            f"hyper_priors={self.hyper_priors}, "
            f"fit_tol={self.fit_tol}, "
            f"max_iter={self.max_iter}, "
            f"verbose={self.verbose}, "
            f"random_seed={self.random_seed}, "
            f"prune_X={self.prune_X}, "
            f"prune_tol={self.prune_tol}, "
            f"save_X_cov={self.save_X_cov}, "
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
        return {
            "gp_fit_config": self.gp_fit_config,
            "hyper_priors": self.hyper_priors,
            "fit_tol": self.fit_tol,
            "max_iter": self.max_iter,
            "verbose": self.verbose,
            "random_seed": self.random_seed,
            "prune_X": self.prune_X,
            "prune_tol": self.prune_tol,
            "save_X_cov": self.save_X_cov,
            "save_C_cov": self.save_C_cov,
            "save_fit_progress": self.save_fit_progress,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
        }

    def set_args(self, **kwargs) -> None:
        """
        Set keyword arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments to set. Must be valid arguments as specified in
            the class docstring.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                msg = f"Invalid argument: {key}"
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
        return cls.DEFAULT_ARGS.copy()

    def set_default_args(self) -> None:
        """Set all arguments to their default values."""
        for key, value in self.DEFAULT_ARGS.items():
            setattr(self, key, value)

    def display(self) -> None:
        """Print out the fit arguments."""
        print("mDLAG Fit Arguments:")
        for key, value in self.get_args().items():
            print(f"  {key}: {value}")
