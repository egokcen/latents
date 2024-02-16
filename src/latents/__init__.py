"""latents package."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from latents import (
    gfa,
    mdlag,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("latents")

__all__ = [
    "gfa",
    "mdlag",
]
