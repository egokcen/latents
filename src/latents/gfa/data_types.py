"""Custom data types used throughout the GFA subpackage."""

from __future__ import annotations

import sys

import numpy as np

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from latents._core.fitting import (
    FitFlags,
    FitTracker,
)
from latents.observation import ObsParamsPosterior
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
        `ndarray` of `int`, shape ``(n_groups,)``.
        Dimensionalities of each observed group.

    Attributes
    ----------
    obs_params : ObsParamsPosterior
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
        self.obs_params = ObsParamsPosterior(x_dim=x_dim, y_dims=y_dims)

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
        x_indices: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """
        Keep only a subset of the latent dimensions in each relevant parameter.

        Parameters
        ----------
        x_indices
            1D `ndarray` of `int`, at most length ``x_dim``.
            Indices of the latent dimensions to keep.
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
            self.obs_params.get_subset_dims(x_indices, in_place=True)
            self.state_params.get_subset_dims(x_indices, in_place=True)
            return self

        return self.__class__(
            obs_params=self.obs_params.get_subset_dims(x_indices, in_place=False),
            state_params=self.state_params.get_subset_dims(x_indices, in_place=False),
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
