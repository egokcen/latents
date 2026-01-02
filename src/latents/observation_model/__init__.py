"""Observation model subpackage (data containers)."""

from latents.observation_model import observations
from latents.observation_model.observations import (
    ObsStatic,
    ObsTimeSeries,
)

__all__ = [
    "ObsStatic",
    "ObsTimeSeries",
    "observations",
]
