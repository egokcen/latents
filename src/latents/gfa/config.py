"""Configuration classes for GFA model fitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class GFAFitConfig:
    """Configuration for GFA model fitting.

    All parameters have sensible defaults. Create with keyword arguments only.
    Instances are immutable (frozen).

    Parameters
    ----------
    x_dim_init : int
        Initial number of latent dimensions (before pruning). Must be >= 1.
    fit_tol : float
        Convergence tolerance for ELBO relative change. Must be > 0.
    max_iter : int
        Maximum EM iterations. Must be >= 1.
    prune_x : bool
        If True, remove latent dimensions that become inactive during fitting.
        Improves speed and memory for high initial `x_dim_init`.
    prune_tol : float
        Variance threshold for pruning. Latents with mean squared value below
        this are removed. Must be > 0.
    save_x : bool
        If True, save posterior latent estimates. Can be memory-intensive
        for large N.
    save_c_cov : bool
        If True, save loading covariances. Can be memory-intensive for large
        y_dim and x_dim.
    save_fit_progress : bool
        If True, track ELBO and runtime per iteration.
    random_seed : int | None
        RNG seed for reproducibility. None for random initialization.
    verbose : bool
        If True, display fitting progress.
    min_var_frac : float
        Private variance floor as fraction of data variance. Must be in (0, 1).

    Examples
    --------
    >>> config = GFAFitConfig(x_dim_init=10, verbose=True)
    >>> config.x_dim_init
    10

    >>> # Configs are immutable
    >>> config.x_dim_init = 20  # Raises FrozenInstanceError
    Traceback (most recent call last):
        ...
    dataclasses.FrozenInstanceError: cannot assign to field 'x_dim_init'
    """

    # Model structure
    x_dim_init: int = 1

    # Convergence
    fit_tol: float = 1e-8
    max_iter: int = 1_000_000

    # Pruning
    prune_x: bool = True
    prune_tol: float = 1e-7

    # Output control
    save_x: bool = False
    save_c_cov: bool = False
    save_fit_progress: bool = True

    # Reproducibility
    random_seed: int | None = None

    # Runtime
    verbose: bool = False
    min_var_frac: float = 0.001

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate x_dim_init
        if not isinstance(self.x_dim_init, int) or self.x_dim_init < 1:
            msg = f"x_dim_init must be an integer >= 1, got {self.x_dim_init!r}"
            raise ValueError(msg)

        # Validate fit_tol
        if self.fit_tol <= 0:
            msg = f"fit_tol must be > 0, got {self.fit_tol}"
            raise ValueError(msg)

        # Validate max_iter
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            msg = f"max_iter must be an integer >= 1, got {self.max_iter!r}"
            raise ValueError(msg)

        # Validate prune_tol
        if self.prune_tol <= 0:
            msg = f"prune_tol must be > 0, got {self.prune_tol}"
            raise ValueError(msg)

        # Validate min_var_frac
        if not 0 < self.min_var_frac < 1:
            msg = f"min_var_frac must be in (0, 1), got {self.min_var_frac}"
            raise ValueError(msg)

        # Validate random_seed
        if self.random_seed is not None and (
            not isinstance(self.random_seed, int) or self.random_seed < 0
        ):
            msg = (
                f"random_seed must be a non-negative integer or None, "
                f"got {self.random_seed!r}"
            )
            raise ValueError(msg)
