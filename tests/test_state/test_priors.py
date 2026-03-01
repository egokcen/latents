"""Test state model prior classes."""

from __future__ import annotations

import numpy as np

from latents.state import LatentsPriorStatic
from latents.state.realizations import LatentsRealization

X_DIM = 3
N_SAMPLES = 10


class TestLatentsPriorStatic:
    """Tests for LatentsPriorStatic.sample()."""

    def test_sample_shapes(self):
        """Sampled realization has correct shape."""
        prior = LatentsPriorStatic()
        s = prior.sample(X_DIM, N_SAMPLES, rng=np.random.default_rng(0))
        assert isinstance(s, LatentsRealization)
        assert s.data.shape == (X_DIM, N_SAMPLES)

    def test_sample_reproducibility(self):
        """Same seed produces identical samples."""
        prior = LatentsPriorStatic()
        s1 = prior.sample(X_DIM, N_SAMPLES, rng=np.random.default_rng(42))
        s2 = prior.sample(X_DIM, N_SAMPLES, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1.data, s2.data)
