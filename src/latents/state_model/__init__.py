"""State model subpackage."""

from latents.state_model import (
    latents,
)
from latents.state_model.latents import (
    PosteriorLatentDelayed,
    PosteriorLatentStatic,
)

__all__ = [
    "latents",
    "PosteriorLatentStatic",
    "PosteriorLatentDelayed",
]
