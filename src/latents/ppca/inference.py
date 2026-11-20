"""Inference functions for probabilistic Principal Component Analysis."""

from __future__ import annotations

from latents.data import ObsStatic
from latents.observation import ObsParamsPoint
from latents.ppca.config import pPCAFitConfig
from latents.state import LatentsPosteriorStatic

_NOT_IMPLEMENTED_MSG = "pPCA not yet implemented"

# -----------------------------------------------------------------------
# Main fit function
# -----------------------------------------------------------------------


def fit(Y):
    """Fit."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def init_posteriors(
    Y: ObsStatic,
    config: pPCAFitConfig | None = None,
    obs_params: ObsParamsPoint | None = None,
) -> tuple[ObsParamsPoint, LatentsPosteriorStatic]:
    """Initialize params."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_latents():
    """Infer latents."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def infer_loadings():
    """Infer loadings."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
