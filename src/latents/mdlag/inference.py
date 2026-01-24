"""Inference functions for mDLAG."""

from __future__ import annotations

_NOT_IMPLEMENTED_MSG = "mDLAG not yet implemented"


def fit():
    """Fit a mDLAG model to data via variational inference.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def init_posteriors():
    """Initialize mDLAG model posteriors for fitting.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_latents():
    """Infer latent posterior given observations and fitted parameters.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_loadings():
    """Infer loading posterior.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_ard():
    """Infer ARD posterior.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_obs_mean():
    """Infer observation mean posterior.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_obs_prec():
    """Infer observation precision posterior.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def learn_gp_params():
    """Learn Gaussian process kernel hyperparameters.

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def compute_lower_bound():
    """Compute the variational lower bound (ELBO).

    Raises
    ------
    NotImplementedError
        mDLAG is not yet implemented.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
