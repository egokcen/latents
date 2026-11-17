"""Configuration classes for pPCA model fitting and simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class pPCASimConfig:
    """Configuration for pPCA simulation fitting."""

    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class pPCAFitConfig:
    """Configuration for pPCA model fitting."""

    pass
