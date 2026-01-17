"""Concrete parameter values for state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatentsRealization:
    """A single realization of latent variables.

    Sources: prior sampling, posterior means, posterior samples.

    Parameters
    ----------
    X : ndarray of float, shape (x_dim, n_samples)
        Latent variables.
    """

    X: np.ndarray

    @property
    def x_dim(self) -> int:
        """Number of latent dimensions."""
        return self.X.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[1]
