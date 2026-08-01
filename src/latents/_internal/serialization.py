"""Serialization utilities shared by model and simulation save paths."""

from __future__ import annotations

import os

import numpy as np
from safetensors.numpy import save_file


def as_contiguous(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous view or copy of ``array``.

    safetensors requires contiguous, dense arrays: since 0.8.0 its numpy
    binding serializes straight from ``ndarray.ctypes.data`` for zero-copy
    writes, ignoring strides. A non-contiguous array is therefore written as
    a raw buffer read under the *view's* shape, silently reordering elements
    (e.g. an F-contiguous array round-trips transposed). Earlier versions
    routed through ``tobytes()``, which always emitted C order and hid the
    requirement.

    Non-contiguous arrays arise naturally here: column indexing such as
    ``mean[:, x_indices]`` during ARD pruning returns F-contiguous results.

    Parameters
    ----------
    array : ndarray
        Array to normalize.

    Returns
    -------
    ndarray
        ``array`` itself if already C-contiguous, else a C-contiguous copy.

    Notes
    -----
    The contiguity check is not merely an optimization: ``np.ascontiguousarray``
    promotes 0-d arrays to shape ``(1,)``, which would corrupt scalars stored
    as tensors. 0-d arrays are always contiguous, so the check skips them.
    """
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def save_tensors(
    tensors: dict[str, np.ndarray],
    path: str | os.PathLike[str],
    metadata: dict[str, str],
) -> None:
    """Write ``tensors`` to a safetensors file, normalizing memory layout.

    Wraps :func:`safetensors.numpy.save_file`, applying :func:`as_contiguous`
    to every array. All safetensors writes in this package go through here so
    the contiguity requirement is enforced in one place.

    Parameters
    ----------
    tensors : dict of str to ndarray
        Arrays to serialize, keyed by name.
    path : str or PathLike
        Output file path (conventionally ends in .safetensors).
    metadata : dict of str to str
        String metadata stored alongside the tensors.
    """
    save_file(
        {k: as_contiguous(v) for k, v in tensors.items()}, path, metadata=metadata
    )
