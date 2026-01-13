"""Pytest configuration and shared test utilities."""

from __future__ import annotations

import numpy as np

from latents._core.numerics import stability_floor


def testing_tols(dtype) -> dict:
    """Return dtype-aware tolerances for np.testing.assert_allclose.

    Parameters
    ----------
    dtype
        NumPy dtype to determine precision. Defaults to float64.

    Returns
    -------
    dict
        Dictionary with 'rtol' and 'atol' keys suitable for unpacking
        into np.testing.assert_allclose.

    Examples
    --------
    >>> tols = testing_tols(np.float64)
    >>> np.testing.assert_allclose(actual, expected, **tols)
    """
    eps = np.finfo(dtype).eps
    floor = stability_floor(dtype)
    return {
        "rtol": float(np.sqrt(eps)),  # ~1.5e-8 (f64), ~3.5e-4 (f32)
        "atol": floor,  # 1e-10 (f64), 1e-6 (f32)
    }
