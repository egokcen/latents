"""Group factor analysis (GFA) subpackage."""

from latents.gfa import (
    core,
    data_types,
    descriptive_stats,
    simulation,
)
from latents.gfa.core import GFAModel

__all__ = [
    "core",
    "data_types",
    "descriptive_stats",
    "GFAModel",
    "simulation",
]
