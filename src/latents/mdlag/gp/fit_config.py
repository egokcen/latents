"""GP fit configuration."""

from dataclasses import dataclass


@dataclass
class GPFitConfig:
    """Configuration for the GP optimizer."""

    max_iter: int = 20  # Increased for better convergence
    tol: float = 1e-8  # Tighter tolerance
    grad_mode: str = "autodiff"  # "autodiff" | "manual"
    verbose: bool = True  # Enable detailed output
