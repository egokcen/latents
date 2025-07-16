"""GP fit configuration."""

from dataclasses import dataclass


@dataclass
class GPFitConfig:
    """Configuration for the GP optimizer."""

    max_iter: int = 10
    tol: float = 1e-6
    grad_mode: str = "autodiff"  # "autodiff" | "manual"
