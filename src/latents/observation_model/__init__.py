"""Observation model subpackage."""

from latents.observation_model import (
    observations,
    probabilistic,
)
from latents.observation_model.observations import (
    ObsStatic,
    ObsTimeSeries,
)

__all__ = [
    "observations",
    "ObsStatic",
    "ObsTimeSeries",
    "probabilistic",
]
