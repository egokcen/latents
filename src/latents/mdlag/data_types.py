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

from latents.observation_model.probabilistic import ObsParamsARD
from latents.state_model.gaussian_process import GPParams
from latents.state_model.latents import StateParamsDelayed


class mDLAGParams:
    """Delayed latents across multiple groups (mDLAG)  model parameters."""

    def __init__(
        self,
        x_dim: int | None = None,
        y_dims: np.ndarray | None = None,
        T: int | None = None,
        gp_params_init: GPParams | None = None,
    ):
        num_groups = len(y_dims)

        # Observed dimensionalities
        if y_dims is not None and not isinstance(y_dims, np.ndarray):
            msg = "y_dims must be a numpy.ndarray of integers."
            raise TypeError(msg)

        # Observation model parameters:
        self.obs_params = ObsParamsARD(x_dim=x_dim, y_dims=y_dims)

        # State model parameters:
        self.state_params_delayed = StateParamsDelayed(x_dim, num_groups, T, X=None)

        # GP parameters:
        if gp_params_init is None:
            self.gp_params = GPParams.generate(x_dim=x_dim, num_groups=num_groups)
        else:
            self.gp_params = gp_params_init

        self.T = T

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(obs_params={self.obs_params}, "
            f"state_params={self.state_params_delayed},"
            f"state_params_gp={self.state_params_gp})"
        )

    def is_initialized(self) -> bool:
        """Check if observation model parameters have been initialized to data."""
        raise NotImplementedError

    def get_subset_dims(
        self,
        dims: np.ndarray,
        in_place: bool = True,
    ) -> mDLAGParams | None:
        """Keep only a subset of the latent dimensions in each relevant parameter."""
        raise NotImplementedError

    def copy(self) -> mDLAGParams:
        """Return a copy of self."""
        raise NotImplementedError


class mDLAGFitTracker:
    """Quantities tracked during a mDLAG model fit."""

    pass


class mDLAGFitFlags:
    """Status messages during a mDLAG model fit."""

    pass


class mDLAGFitArgs:
    """Keyword arguments used to fit a mDLAG model."""

    pass
