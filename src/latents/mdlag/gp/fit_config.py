"""GP fit configuration."""

from dataclasses import dataclass


@dataclass
class GPFitConfig:
    """Configuration for the GP optimizer."""

    max_iter: int = 20
    tol: float = 1e-8
    grad_mode: str = "autodiff"  # "autodiff" | "manual"
    verbose: bool = True
