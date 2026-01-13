"""Configuration classes for mDLAG model fitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class mDLAGFitConfig:
    """Configuration for mDLAG model fitting."""

    pass
