"""Test state model realization classes."""

from __future__ import annotations

import numpy as np

from latents.state.realizations import LatentsRealization

X_DIM = 3
N_SAMPLES = 10


class TestLatentsRealization:
    """Tests for LatentsRealization dataclass."""

    def test_properties(self):
        """Derived properties x_dim and n_samples are correct."""
        r = LatentsRealization(data=np.zeros((X_DIM, N_SAMPLES)))
        assert r.x_dim == X_DIM
        assert r.n_samples == N_SAMPLES
