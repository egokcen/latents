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

from latents.mdlag.gp.gp_model import mDLAGGP
from latents.observation_model.probabilistic import ObsParamsARD
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
            and (self.gp is not None and self.gp.params.is_initialized())
        )

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
