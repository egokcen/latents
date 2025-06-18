"""GP fit configuration."""

from dataclasses import dataclass


@dataclass
class GPFitConfig:
    """Configuration settings for the GP parameter optimizer."""

    grad_mode: str = "autodiff"  # "autodiff" or "manual"
    max_iter: int = 50
    tol: float = 1e-6
