"""State model subpackage."""

from latents.state_model import gaussian_process, latents
from latents.state_model.latents import PosteriorLatentDelayed, PosteriorLatentStatic

__all__ = [
    "PosteriorLatentDelayed",
    "PosteriorLatentStatic",
    "gaussian_process",
    "latents",
]
