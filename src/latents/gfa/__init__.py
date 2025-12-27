"""Group factor analysis (GFA) subpackage."""

from latents.gfa import (
    config,
    core,
    data_types,
    descriptive_stats,
    simulation,
)
from latents.gfa.config import GFAFitConfig
from latents.gfa.core import GFAModel

__all__ = [
    "GFAFitConfig",
    "GFAModel",
    "config",
    "core",
    "data_types",
    "descriptive_stats",
    "simulation",
]
