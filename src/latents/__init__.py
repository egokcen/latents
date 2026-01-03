"""latents package."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from latents import (
    data,
    gfa,
    mdlag,
    observation,
    plotting,
    state,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("latents")

__all__ = [
    "data",
    "gfa",
    "mdlag",
    "observation",
    "plotting",
    "state",
]
