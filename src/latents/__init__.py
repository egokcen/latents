"""latents package."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from latents import (
    base,
    callbacks,
    data,
    gfa,
    mdlag,
    observation,
    plotting,
    state,
    tracking,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("latents")

__all__ = [
    "base",
    "callbacks",
    "data",
    "gfa",
    "mdlag",
    "observation",
    "plotting",
    "state",
    "tracking",
]
