"""State model subpackage."""

from latents.state_model import GP_latents, latents
from latents.state_model.latents import PosteriorLatentDelayed, PosteriorLatentStatic

__all__ = [
    "GP_latents",
    "PosteriorLatentDelayed",
    "PosteriorLatentStatic",
    "latents",
]
