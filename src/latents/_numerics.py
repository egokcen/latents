"""Numerical stability utilities for the latents package."""

from __future__ import annotations

import numpy as np


def stability_floor(dtype=np.float64) -> float:
    """Return floor value for log/division stability based on dtype.

    Uses max(x, floor) pattern to prevent -inf/inf from near-zero values.
    Provides headroom above machine epsilon while remaining small enough
    not to affect results.

    Parameters
    ----------
    dtype
        NumPy dtype to determine precision. Defaults to float64.

    Returns
    -------
    float
        1e-10 for float64, 1e-6 for float32.
    """
    eps = np.finfo(dtype).eps
    return 1e-10 if eps < 1e-10 else 1e-6


def validate_tolerance(tol: float, dtype, name: str) -> None:
    """Warn if tolerance is below meaningful precision for dtype.

    Convergence and pruning tolerances are algorithmic choices, not
    precision-bound. However, setting them below the stability floor
    can lead to unreliable convergence behavior.

    Parameters
    ----------
    tol
        Tolerance value to validate.
    dtype
        NumPy dtype to determine precision floor.
    name
        Parameter name for the warning message.

    Warns
    -----
    UserWarning
        If ``tol`` is below the stability floor for ``dtype``.
    """
    import warnings

    floor = stability_floor(dtype)
    if tol < floor:
        warnings.warn(
            f"{name}={tol} is below precision floor {floor} for {dtype}. "
            "Convergence may be unreliable.",
            stacklevel=2,
        )
