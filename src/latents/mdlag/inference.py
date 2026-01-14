"""Inference functions for mDLAG."""

from __future__ import annotations

from latents.tracking import FitFlags, FitTracker

_NOT_IMPLEMENTED_MSG = "mDLAG not yet implemented"


class mDLAGFitTracker(FitTracker):
    """Quantities tracked during a mDLAG model fit."""

    pass


class mDLAGFitFlags(FitFlags):
    """Status messages during a mDLAG model fit."""

    pass


def fit():
    """Fit a mDLAG model to data via variational inference."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def init_posteriors():
    """Initialize mDLAG model posteriors for fitting."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_latents():
    """Infer latent posterior given observations and fitted parameters."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_loadings():
    """Infer loading posterior."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_ard():
    """Infer ARD posterior."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_obs_mean():
    """Infer observation mean posterior."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_obs_prec():
    """Infer observation precision posterior."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def learn_gp_params():
    """Learn Gaussian process kernel hyperparameters."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def compute_lower_bound():
    """Compute the variational lower bound (ELBO)."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
