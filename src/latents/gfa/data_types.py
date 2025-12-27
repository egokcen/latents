"""Custom data types used throughout the GFA subpackage."""

from __future__ import annotations

import numpy as np

from latents.base import (
    FitFlags,
    FitTracker,
)
from latents.observation_model.probabilistic import ObsParamsARD
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
