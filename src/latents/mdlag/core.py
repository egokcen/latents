"""Core utilities to fit an mDLAG model to data."""

from __future__ import annotations

_NOT_IMPLEMENTED_MSG = "mDLAG not yet implemented"


def fit():
    """Fit a mDLAG model to data.

    Fit a delayed latents across multiple groups (mDLAG) model using an iterative
    variational inference scheme with mean-field approximation.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def init():
    """Initialize mDLAG model parameters for fitting."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_latents():
    """Infer latent variables given mDLAG model parameters and observed data."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def learn_gp_params():
    """Learn Gaussian process parameters given mDLAG model parameters and latents."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_loadings():
    """Infer loadings :math:`C` given current params and observed data."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_ard():
    """Infer ARD parameters alpha given current params."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_obs_mean():
    """Infer observation mean parameter given current params and observed data."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_obs_prec():
    """Infer observation precision parameters given current params and observed data."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def compute_lower_bound():
    """Compute the variational lower bound for a mDLAG model on observed data."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def compute_lower_bound_constants():
    """Compute constant factors in the variational lower bound."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


class mDLAGModel:
    """Interface with, fit, and store the fitting results of a mDLAG model."""

    def __init__():
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def __repr__():
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def fit():
        """Fit a mDLAG model to data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def init():
        """Initialize mDLAG model parameters."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def save():
        """Save a mDLAGModel object to a JSON file."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    @staticmethod
    def load():
        """Load a mDLAGModel object from a JSON file."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_latents():
        """Infer latent variables X given current params and observed data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_loadings():
        """Infer loadings C given current params and observed data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_ard():
        """Infer ARD parameters alpha given current params."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_obs_mean():
        """Infer observation mean parameter given current params and observed data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_obs_prec():
        """Infer observation precision params given current params and observed data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def compute_lower_bound():
        """Compute the variational lower bound given observed data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
