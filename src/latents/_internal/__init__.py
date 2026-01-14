"""Internal utilities: base classes and numerical helpers."""

from latents._internal.base import ArrayContainer
from latents._internal.numerics import stability_floor, validate_tolerance

__all__ = [
    "ArrayContainer",
    "stability_floor",
    "validate_tolerance",
]
