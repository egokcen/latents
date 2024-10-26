"""latents package."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from latents import (
    base,
    gfa,
    mdlag,
    observation_model,
    plotting,
    state_model,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("latents")

__all__ = [
    "base",
    "gfa",
    "mdlag",
    "observation_model",
    "plotting",
    "state_model",
]
