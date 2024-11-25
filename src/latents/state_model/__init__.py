"""State model subpackage."""

from latents.state_model import GP_latents, latents
from latents.state_model.latents import PosteriorLatentDelayed, PosteriorLatentStatic

__all__ = [
    "latents",
    "GP_latents",
    "PosteriorLatentStatic",
    "PosteriorLatentDelayed",
]
