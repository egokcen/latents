"""Internal utilities: base classes and numerical helpers."""

from latents._core.base import ArrayContainer
from latents._core.fitting import FitFlags, FitTracker
from latents._core.numerics import stability_floor, validate_tolerance

__all__ = [
    "ArrayContainer",
    "FitFlags",
    "FitTracker",
    "stability_floor",
    "validate_tolerance",
]
