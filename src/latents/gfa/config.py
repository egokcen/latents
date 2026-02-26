"""Configuration classes for GFA model fitting and simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _validate_random_seed(seed: int | Sequence[int] | None) -> None:
    """Validate random_seed field value.

    Parameters
    ----------
    seed : int, sequence of int, or None
        The seed value to validate.

    Raises
    ------
    TypeError
        If seed is not an int, sequence of ints, or None.
    ValueError
        If seed is negative, sequence is empty, or sequence contains negatives.
    """
    if seed is None:
        return

    if isinstance(seed, int):
        if seed < 0:
            msg = f"random_seed must be non-negative, got {seed}"
            raise ValueError(msg)
    elif isinstance(seed, str):
        # Strings are Sequences but not valid seeds
        msg = f"random_seed must be an int, sequence of ints, or None, got {seed!r}"
        raise TypeError(msg)
    elif isinstance(seed, Sequence):
        if len(seed) == 0:
            msg = "random_seed sequence must not be empty"
            raise ValueError(msg)
        for i, val in enumerate(seed):
            if not isinstance(val, int):
                msg = (
                    f"random_seed sequence must contain integers, "
                    f"got {type(val).__name__} at index {i}"
                )
                raise TypeError(msg)
            if val < 0:
                msg = (
                    f"random_seed sequence values must be non-negative, "
                    f"got {val} at index {i}"
                )
                raise ValueError(msg)
    else:
        msg = (
            f"random_seed must be an int, sequence of ints, or None, "
            f"got {type(seed).__name__}"
        )
        raise TypeError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class GFASimConfig:
    """Experimental design parameters for GFA simulation.

    Defines the structure and reproducibility settings for generating synthetic
    GFA data. The probabilistic model specification (hyperprior) is provided
    separately to `simulate()`.

    All parameters except `random_seed` are required. Instances are immutable
    (frozen).

    Parameters
    ----------
    n_samples : int
        Number of data points to generate. Must be >= 1.
    y_dims : np.ndarray
        Dimensionalities of each observed group, shape ``(n_groups,)``.
        Must be 1D array of positive integers.
    x_dim : int
        Number of latent dimensions. Must be >= 1.
    snr : float or ndarray
        Signal-to-noise ratio. Either a scalar (broadcast to all groups) or
        per-group array of shape ``(n_groups,)``. Must be > 0.
    random_seed : int, sequence of int, or None
        RNG seed for reproducibility. Accepts a single non-negative integer,
        a non-empty sequence of non-negative integers (for structured seeding
        in parallel experiments), or None for random initialization. Required
        for reproducible recipe saves.

    Examples
    --------
    >>> config = GFASimConfig(
    ...     n_samples=100,
    ...     y_dims=np.array([10, 10, 10]),
    ...     x_dim=5,
    ...     snr=1.0,
    ...     random_seed=42,
    ... )
    >>> config.n_groups
    3
    >>> config.y_dim
    30
    """

    n_samples: int
    y_dims: np.ndarray
    x_dim: int
    snr: float | np.ndarray = 1.0
    random_seed: int | Sequence[int] | None = None

    @property
    def n_groups(self) -> int:
        """Number of observed groups."""
        return len(self.y_dims)

    @property
    def y_dim(self) -> int:
        """Total observed dimensionality (sum of y_dims)."""
        return int(self.y_dims.sum())

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate n_samples
        if not isinstance(self.n_samples, int) or self.n_samples < 1:
            msg = f"n_samples must be an integer >= 1, got {self.n_samples!r}"
            raise ValueError(msg)

        # Validate y_dims: must be 1D array of positive integers
        if not isinstance(self.y_dims, np.ndarray):
            msg = f"y_dims must be a numpy array, got {type(self.y_dims).__name__}"
            raise TypeError(msg)
        if self.y_dims.ndim != 1:
            msg = f"y_dims must be 1D, got shape {self.y_dims.shape}"
            raise ValueError(msg)
        if len(self.y_dims) == 0:
            msg = "y_dims must have at least one element"
            raise ValueError(msg)
        if not np.issubdtype(self.y_dims.dtype, np.integer):
            msg = f"y_dims must have integer dtype, got {self.y_dims.dtype}"
            raise TypeError(msg)
        if not np.all(self.y_dims >= 1):
            msg = f"y_dims values must be >= 1, got {self.y_dims}"
            raise ValueError(msg)

        # Validate x_dim
        if not isinstance(self.x_dim, int) or self.x_dim < 1:
            msg = f"x_dim must be an integer >= 1, got {self.x_dim!r}"
            raise ValueError(msg)

        # Validate snr: scalar or array, must be > 0
        snr = self.snr
        if isinstance(snr, np.ndarray):
            if snr.ndim != 1:
                msg = f"snr array must be 1D, got shape {snr.shape}"
                raise ValueError(msg)
            if len(snr) != 1 and len(snr) != len(self.y_dims):
                msg = (
                    f"snr array must have length 1 or n_groups={len(self.y_dims)}, "
                    f"got {len(snr)}"
                )
                raise ValueError(msg)
            if not np.all(snr > 0):
                msg = f"snr values must be > 0, got {snr}"
                raise ValueError(msg)
        else:
            # Scalar
            if not isinstance(snr, (int, float)) or snr <= 0:
                msg = f"snr must be > 0, got {snr}"
                raise ValueError(msg)

        _validate_random_seed(self.random_seed)


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
    random_seed : int, sequence of int, or None
        RNG seed for reproducibility. Accepts a single non-negative integer,
        a non-empty sequence of non-negative integers (for structured seeding
        in parallel experiments), or None for random initialization.
    min_var_frac : float
        Private variance floor as fraction of data variance. Must be in (0, 1).

    Examples
    --------
    >>> config = GFAFitConfig(x_dim_init=10)
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
    random_seed: int | Sequence[int] | None = None

    # Numerical stability
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

        _validate_random_seed(self.random_seed)
