"""latents package."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from latents import (
    gfa,
    mdlag,
    observation,
    observation_model,
    plotting,
    state,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("latents")

__all__ = [
    "gfa",
    "mdlag",
    "observation",
    "observation_model",
    "plotting",
    "state",
]
