"""Configuration classes for pPCA model fitting and simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
class pPCASimConfig:
    """Experimental design parameters for pPCA simulation.

    Defines the structure and reproducibility settings for generating synthetic
    pPCA data.

    All parameters except 'random_seed' are required. Instances are immutable
    (frozen).

    Parameters
    ----------
    n_samples : int
        Number of data points to generate. Must be >= 1.
    x_dim : int
        Number of latent dimensions. Must be >= 1.
    y_dim : int
        Dimensionality of observed group.
    snr : float
        Signal-to-noise ratio. scalar. Must be > 0.
    random_seed : int, sequence of int, or None
        RNG seed for reproducibility. Accepts a single non-negative integer,
        a non-empty sequence of non-negative integers (for structured seeding
        in parallel experiments), or None for random initialization. Required
        for reproducible recipe saves.

    Examples
    --------
    >>> config = pPCASimConfig(
    ...     n_samples=100,
    ...     x_dim=5,
    ...     y_dim=10,
    ...     snr=1.0,
    ...     random_seed=42,
    ... )
    """

    n_samples: int
    x_dim: int
    y_dim: int
    snr: float | int
    random_seed: int | Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate n_samples
        if not isinstance(self.n_samples, int) or self.n_samples < 1:
            msg = f"n_samples must be an integer >= 1, got {self.n_samples!r}"
            raise ValueError(msg)

        # Validate x_dim
        if not isinstance(self.x_dim, int) or self.x_dim < 1:
            msg = f"x_dim must be an integer >= 1, got {self.x_dim!r}"
            raise ValueError(msg)

        # Validate y_dim
        if not isinstance(self.y_dim, int):
            msg = f"y_dim must be integer, got {type(self.y_dim).__name__}"
            raise TypeError(msg)
        if (self.y_dim < self.x_dim) or self.y_dim <= 1:
            msg = "y_dim must be >= x_dim >= 1"
            raise ValueError(msg)

        # validate snr: float, must be > 0
        snr = self.snr
        if isinstance(snr, (float, int)):
            if snr <= 0:
                msg = f"snr must be > 0, got {snr}"
                raise ValueError(msg)
        else:
            msg = f"snr must be float or int, got {type(self.snr).__name__}"
            raise TypeError(msg)

        _validate_random_seed(self.random_seed)


@dataclass(frozen=True, slots=True, kw_only=True)
class pPCAFitConfig:
    """Configuration for pPCA model fitting.

    All parameters have sensible defaults. Create with keyword arguments only.
    Instances are immutable (frozen).

    Parameters
    ----------
    x_dim : int
        Number of latent dimensions. Must be >= 1.
    fit_tol : float
        Convergence tolerance for ELBO relative change. Must be > 0.
    max_iter : int
        Maximum EM iterations. Must be >= 1.
    save_x : bool
        If True, save posterior latent estimates.
    save_c_cov : bool
        If True, save loading covariances.
    save_fit_progress : bool
        If True, track ELBO and runtime per iteration.
    random_seed : int, sequence of int, or None
        RNG seed for reproducibility. Accepts a single non-negative integer,
        a non-empty sequence of non-negative integers (for structured seeding
        in parallel experiments), or None for random initialization.
    min_var_frac : float
        Noise variance floor as fraction of data variance. Must be in (0, 1).

    Examples
    --------
    >>> config = pPCAFitConfig(x_dim=5)
    >>> config.x_dim
    5
    >>> # Configs are immutable
    >>> config.x_dim = 20 # Raises FrozenInstanceError
    Traceback (most recent call last):
        ...
    dataclasses.FrozenInstanceError: cannot assign to field 'x_dim'
    """

    # Model structure
    x_dim: int = 5

    # Convergence
    fit_tol: float = 1e-8
    max_iter: int = 1_000_000

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
        # Validate x_dim
        if not isinstance(self.x_dim, int) or self.x_dim < 1:
            msg = f"x_dim must be an integer >= 1, got {self.x_dim!r}"
            raise ValueError(msg)

        # Validate fit_tol
        if self.fit_tol <= 0:
            msg = f"fit_tol must be > 0, got {self.fit_tol}"
            raise ValueError(msg)

        # Validate max_iter
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            msg = f"max_iter must be an integer >= 1, got {self.max_iter!r}"
            raise ValueError(msg)

        # Validate min_var_frac
        if not 0 < self.min_var_frac < 1:
            msg = f"min_var_frac must be in (0, 1), got {self.min_var_frac}"
            raise ValueError(msg)

        _validate_random_seed(self.random_seed)
