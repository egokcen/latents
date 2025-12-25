"""Core utilities to fit an mDLAG model to data."""

from __future__ import annotations


def fit():
    """Fit a mDLAG model to data.

    Fit a delayed latents across multiple groups (mDLAG) model using an iterative
    variational inference scheme with mean-field approximation.
    """
    pass


def init():
    """Initialize mDLAG model parameters for fitting."""
    pass


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
