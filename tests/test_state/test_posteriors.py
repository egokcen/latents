"""Test state model posterior classes."""

from __future__ import annotations

import numpy as np
import pytest

from latents.state import LatentsPosteriorStatic
from latents.state.realizations import LatentsRealization

# --------------------------------------------------------------------------- #
# Shared test dimensions: x_dim=3, n_samples=10
# --------------------------------------------------------------------------- #
X_DIM = 3
N_SAMPLES = 10


def _make_posterior() -> LatentsPosteriorStatic:
    """Small deterministic LatentsPosteriorStatic for testing."""
    rng = np.random.default_rng(0)
    mean = rng.standard_normal((X_DIM, N_SAMPLES))
    cov = np.eye(X_DIM) * 0.1
    moment = N_SAMPLES * cov + mean @ mean.T  # (x_dim, x_dim)
    return LatentsPosteriorStatic(mean=mean, cov=cov, moment=moment)


class TestLatentsPosteriorStatic:
    """Tests for LatentsPosteriorStatic dataclass."""

    def test_init_defaults(self):
        """All fields default to None."""
        lps = LatentsPosteriorStatic()
        assert lps.mean is None
        assert lps.cov is None
        assert lps.moment is None

    def test_init_with_arrays(self):
        """Arrays are stored with correct shapes."""
        lps = _make_posterior()
        assert lps.mean.shape == (X_DIM, N_SAMPLES)
        assert lps.cov.shape == (X_DIM, X_DIM)
        assert lps.moment.shape == (X_DIM, X_DIM)

    def test_type_validation(self):
        """Non-ndarray inputs are rejected."""
        with pytest.raises(TypeError, match=r"mean must be a numpy\.ndarray"):
            LatentsPosteriorStatic(mean=[[1, 2]])

    def test_properties(self):
        """Derived properties x_dim and n_samples are correct."""
        lps = _make_posterior()
        assert lps.x_dim == X_DIM
        assert lps.n_samples == N_SAMPLES

    def test_properties_none(self):
        """Properties return None when uninitialized."""
        lps = LatentsPosteriorStatic()
        assert lps.x_dim is None
        assert lps.n_samples is None

    def test_is_initialized(self):
        """Initialization check reflects whether arrays are set."""
        assert _make_posterior().is_initialized()
        assert not LatentsPosteriorStatic().is_initialized()

    def test_posterior_mean(self):
        """Posterior mean returns a LatentsRealization copy."""
        lps = _make_posterior()
        pm = lps.posterior_mean
        assert isinstance(pm, LatentsRealization)
        np.testing.assert_array_equal(pm.data, lps.mean)
        # Should be a copy, not a view
        pm.data[:] = 999.0
        assert not np.any(lps.mean == 999.0)

    def test_sample_shapes(self):
        """Sampled realization has correct shape."""
        lps = _make_posterior()
        s = lps.sample(np.random.default_rng(0))
        assert isinstance(s, LatentsRealization)
        assert s.data.shape == (X_DIM, N_SAMPLES)

    def test_sample_reproducibility(self):
        """Same seed produces identical samples."""
        lps = _make_posterior()
        s1 = lps.sample(np.random.default_rng(42))
        s2 = lps.sample(np.random.default_rng(42))
        np.testing.assert_array_equal(s1.data, s2.data)

    def test_compute_moment_in_place(self):
        """In-place moment computation stores result on the object."""
        lps = _make_posterior()
        expected = N_SAMPLES * lps.cov + lps.mean @ lps.mean.T
        result = lps.compute_moment(in_place=True)
        np.testing.assert_allclose(result, expected)
        assert result is lps.moment

    def test_compute_moment_copy(self):
        """Copy mode returns a new array without modifying the object."""
        lps = _make_posterior()
        expected = N_SAMPLES * lps.cov + lps.mean @ lps.mean.T
        result = lps.compute_moment(in_place=False)
        np.testing.assert_allclose(result, expected)
        assert result is not lps.moment

    def test_get_subset_dims_in_place(self):
        """In-place subsetting reduces x_dim on the object."""
        lps = _make_posterior()
        lps.get_subset_dims(np.array([0, 2]), in_place=True)
        assert lps.mean.shape == (2, N_SAMPLES)
        assert lps.cov.shape == (2, 2)
        assert lps.moment.shape == (2, 2)

    def test_get_subset_dims_copy(self):
        """Copy mode returns a subset without modifying the original."""
        lps = _make_posterior()
        new = lps.get_subset_dims(np.array([1]), in_place=False)
        assert lps.mean.shape == (X_DIM, N_SAMPLES)  # original unchanged
        assert new.mean.shape == (1, N_SAMPLES)
        assert new.cov.shape == (1, 1)

    def test_copy_independence(self):
        """Copied posterior is independent of the original."""
        lps = _make_posterior()
        lps2 = lps.copy()
        lps2.mean[:] = 999.0
        assert not np.any(lps.mean == 999.0)

    def test_clear(self):
        """Clear resets all fields to None."""
        lps = _make_posterior()
        lps.clear()
        assert lps.mean is None
        assert lps.cov is None
        assert lps.moment is None

    def test_repr(self):
        """Repr includes array shape information."""
        lps = _make_posterior()
        r = repr(lps)
        assert "mean.shape=" in r
        assert "cov.shape=" in r
