"""Visualization utilities for model components and analysis results."""

from __future__ import annotations

from latents.plotting.hinton import hinton_diagram
from latents.plotting.observation import (
    plot_dimensionalities,
    plot_dims_pairs,
    plot_var_exp,
    plot_var_exp_pairs,
)

__all__ = [
    "hinton_diagram",
    "plot_dimensionalities",
    "plot_dims_pairs",
    "plot_var_exp",
    "plot_var_exp_pairs",
]
