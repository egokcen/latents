"""
Core utilities to fit a delayed latents across multiple groups (mDLAG) model to data.

**Functions**

- :func:`fit` -- Fit a mDLAG model to data.
- :func:`init` -- Initialize mDLAG model parameters to data prior to fitting.
- :func:`infer_latents` -- Infer latent variables.
- :func:`learn_gp_params` -- Learn Gaussian process parameters.
- :func:`infer_loadings` -- Infer loading matrices.
- :func:`infer_ard` -- Infer ARD parameters.
- :func:`infer_obs_mean` -- Infer observation mean parameter.
- :func:`infer_obs_prec` -- Infer observation precision parameters.
- :func:`compute_lower_bound` -- Compute the variational lower bound.
- :func:`compute_lower_bound_constants` -- Compute constants in the lower bound.

**Classes**

- :class:`mDLAGModel` -- A wrapper class to store mDLAG fitting results.

"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.stats import gmean

from latents.mdlag.data_types import mDLAGParams
from latents.observation_model.observations import ObsTimeSeries
from latents.observation_model.probabilistic import (
    HyperPriorParams,
)
from latents.state_model.gaussian_process import GPParams


def fit():
    """Fit a mDLAG model to data.

    Fit a delayed latents across multiple groups (mDLAG) model using an iterative
    variational inference scheme with mean-field approximation.
    """
    pass


def init(
    Y: ObsTimeSeries,
    gp_params_init: GPParams,
    hyper_priors: HyperPriorParams | None = None,
    random_seed: int | None = None,
    save_C_cov: bool = False,
) -> mDLAGParams:
    """Initialize mDLAG model parameters for fitting.

    Parameters
    ----------
    Y
        Observed time series data.
    gp_params_init
        Initial Gaussian process parameters.
    hyper_priors
        Hyperparameters of the mDLAG prior distributions. If not provided,
        default hyperparameters will be used.
    random_seed
        Seed the random number generator for reproducibility. Defaults to
        ``None``.
    save_C_cov
        Set to ``True`` to save posterior covariance of :math:`C`. For large
        ``y_dim`` and ``x_dim``, these structures can use a lot of memory.
        Defaults to ``False``.

    Returns
    -------
    mDLAGParams
        Initialized mDLAG model parameters.

    Raises
    ------
    TypeError
        If ``hyper_priors`` is not a ``HyperPriorParams`` object.
    """
    # Initialize hyper_priors if not provided
    if hyper_priors is None:
        hyper_priors = HyperPriorParams()
    elif not isinstance(hyper_priors, HyperPriorParams):
        msg = "hyper_priors must be a HyperPriorParams object."
        raise TypeError(msg)

    # Seed the random number generator for reproducible initialization.
    rng = np.random.default_rng(random_seed)

    # Get data size characteristics
    y_dims = Y.dims  # Dimensionality of each group
    y_dim = y_dims.sum()  # Total number of observed dimensions
    num_groups = len(y_dims)  # Number of observed groups
    N = Y.data.shape[2]  # Number of samples
    T = Y.T  # Number of time point
    x_dim = gp_params_init.x_dim  # Number of latent dimensions

    # Get views of the observed data for each group
    Ys = Y.get_groups()
    # Get the variance of each observed group
    Y_covs = [np.cov(Y_m.reshape(Y_m.shape[0], -1)) for Y_m in Ys]

    # Initialize mDLAG parameter object
    params = mDLAGParams(x_dim, y_dims, T, gp_params_init)
    obs_params = params.obs_params

    # Mean parameter
    obs_params.d.mean = np.mean(Y.data, axis=(1, 2))
    obs_params.d.cov = np.full(y_dim, 1 / hyper_priors.d_beta)

    # Noise precisions
    obs_params.phi.a = hyper_priors.a_phi + N * T / 2
    obs_params.phi.b = np.full(y_dim, hyper_priors.b_phi)
    obs_params.phi.mean = np.concatenate(
        [1 / np.diag(Y_cov) for Y_cov in Y_covs], axis=0
    )

    # Loading matrices
    # Mean
    obs_params.C.mean = np.zeros((y_dim, x_dim))
    # Get views of the loading matrices for each group
    C_means, _, _ = obs_params.C.get_groups(y_dims)
    for group_idx in range(num_groups):
        eigs = eigh(Y_covs[group_idx], eigvals_only=True)
        scale = gmean(eigs[eigs > 0])
        C_means[group_idx][:] = rng.normal(
            scale=np.sqrt(scale / x_dim), size=(y_dims[group_idx], x_dim)
        )

    obs_params.C.cov = np.zeros((y_dim, x_dim, x_dim))
    # Second moments
    obs_params.C.compute_moment()
    if not save_C_cov:
        # Delete the loading matrix covariances to save memory
        obs_params.C.cov = None
    # Get views of the loading matrix moments for each group
    _, _, C_moments = obs_params.C.get_groups(y_dims)

    # ARD parameters
    obs_params.alpha.a = hyper_priors.a_alpha + y_dims / 2
    obs_params.alpha.b = np.full((num_groups, x_dim), hyper_priors.b_alpha)
    # Scale ARD parameters to match the data
    obs_params.alpha.mean = np.zeros((num_groups, x_dim))
    for group_idx in range(num_groups):
        obs_params.alpha.mean[group_idx, :] = y_dims[group_idx] / np.diag(
            np.sum(C_moments[group_idx], axis=0)
        )

    return params


def infer_latents():
    """Infer latent variables given mDLAG model parameters and observed data."""
    pass


def learn_gp_params():
    """Learn Gaussian process parameters given mDLAG model parameters and latents."""
    pass


def infer_loadings():
    """Infer loadings :math:`C` given current params and observed data."""
    pass


def infer_ard():
    """Infer ARD parameters alpha given current params."""
    pass


def infer_obs_mean():
    """Infer observation mean parameter given current params and observed data."""
    pass


def infer_obs_prec():
    """Infer observation precision parameters given current params and observed data."""
    pass


def compute_lower_bound():
    """Compute the variational lower bound for a mDLAG model on observed data."""
    pass


def compute_lower_bound_constants():
    """Compute constant factors in the variational lower bound."""
    pass


class mDLAGModel:
    """Interface with, fit, and store the fitting results of a mDLAG model."""

    def __init__():
        pass

    def __repr__():
        pass

    def fit():
        """Fit a mDLAG model to data."""
        pass

    def init():
        """Initialize mDLAG model parameters."""
        pass

    def save():
        """Save a mDLAGModel object to a JSON file."""
        pass

    @staticmethod
    def load():
        """Load a mDLAGModel object from a JSON file."""
        pass

    def infer_latents():
        """Infer latent variables X given current params and observed data."""
        pass

    def infer_loadings():
        """Infer loadings C given current params and observed data."""
        pass

    def infer_ard():
        """Infer ARD parameters alpha given current params."""
        pass

    def infer_obs_mean():
        """Infer observation mean parameter given current params and observed data."""
        pass

    def infer_obs_prec():
        """Infer observation precision params given current params and observed data."""
        pass

    def compute_lower_bound():
        """Compute the variational lower bound given observed data."""
        pass
