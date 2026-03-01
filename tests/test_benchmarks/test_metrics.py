"""Tests for benchmark metrics."""

import numpy as np

from benchmarks.metrics import (
    denoised_r2,
    latent_permutation,
    relative_l2_error,
    subspace_error,
)
from tests.conftest import testing_tols as _testing_tols


class TestLatentPermutation:
    """Tests for latent_permutation alignment."""

    def test_identity_when_aligned(self):
        """Already-aligned latents should return identity permutation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 100))
        perm = latent_permutation(X, X)
        np.testing.assert_array_equal(perm, [0, 1, 2])

    def test_recovers_known_permutation(self):
        """Should recover the inverse of a known row shuffle."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((4, 200))
        # Shuffle: X_est[0]=X_true[2], X_est[1]=X_true[0], etc.
        shuffle = [2, 0, 3, 1]
        X_est = X_true[shuffle, :]
        perm = latent_permutation(X_true, X_est)
        # perm[k] = estimated index matching true dim k (inverse of shuffle)
        # true 0 -> est 1, true 1 -> est 3, true 2 -> est 0, true 3 -> est 2
        np.testing.assert_array_equal(perm, [1, 3, 0, 2])

    def test_handles_sign_flip(self):
        """Sign-flipped latents should still match correctly."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((3, 200))
        # Flip sign of all dimensions
        X_est = -X_true
        perm = latent_permutation(X_true, X_est)
        np.testing.assert_array_equal(perm, [0, 1, 2])

    def test_permutation_with_sign_flip(self):
        """Combined permutation and sign flip should still align."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((3, 200))
        # Shuffle [1, 2, 0]: X_est[0]=X_true[1], X_est[1]=X_true[2], X_est[2]=X_true[0]
        X_est = X_true[[1, 2, 0], :].copy()
        X_est[1, :] *= -1  # flip sign of est dim 1 (= true dim 2)
        perm = latent_permutation(X_true, X_est)
        # Inverse: true 0 -> est 2, true 1 -> est 0, true 2 -> est 1
        np.testing.assert_array_equal(perm, [2, 0, 1])

    def test_zero_variance_dimension(self):
        """Zero-variance latent dimension should not cause errors."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((3, 100))
        X_est = rng.standard_normal((3, 100))
        # Make one estimated dimension constant (zero variance)
        X_est[1, :] = 0.0
        perm = latent_permutation(X_true, X_est)
        assert perm.shape == (3,)
        assert len(set(perm)) == 3  # still a valid permutation

    def test_returns_integer_array(self):
        """Returned permutation should be an integer array."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 100))
        perm = latent_permutation(X, X)
        assert perm.dtype.kind == "i"


class TestSubspaceError:
    """Tests for subspace_error metric."""

    def test_identical_matrices_returns_zero(self):
        """Identical matrices should have zero subspace error."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        tols = _testing_tols(C.dtype)
        np.testing.assert_allclose(subspace_error(C, C), 0.0, **tols)

    def test_scaled_matrix_returns_zero(self):
        """Scaled version of same matrix spans same subspace."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        C_scaled = C * 2.5
        tols = _testing_tols(C.dtype)
        np.testing.assert_allclose(subspace_error(C, C_scaled), 0.0, **tols)

    def test_reordered_columns_returns_zero(self):
        """Column reordering doesn't change subspace."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        C_reordered = C[:, [2, 0, 1]]
        tols = _testing_tols(C.dtype)
        np.testing.assert_allclose(subspace_error(C, C_reordered), 0.0, **tols)

    def test_orthogonal_subspaces_returns_one(self):
        """Orthogonal subspaces should have error of 1."""
        # Create orthogonal subspaces using first and last columns of identity
        C_true = np.eye(10)[:, :3]  # First 3 columns
        C_est = np.eye(10)[:, 7:]  # Last 3 columns
        tols = _testing_tols(C_true.dtype)
        np.testing.assert_allclose(subspace_error(C_true, C_est), 1.0, **tols)

    def test_partial_overlap(self):
        """Partial overlap should give error between 0 and 1."""
        # C_est captures 2 of 3 dimensions of C_true
        C_true = np.eye(10)[:, :3]
        C_est = np.eye(10)[:, :2]
        err = subspace_error(C_true, C_est)
        assert 0 < err < 1

    def test_different_column_counts(self):
        """Should handle different numbers of columns."""
        rng = np.random.default_rng(42)
        C_true = rng.standard_normal((10, 3))
        C_est = rng.standard_normal((10, 5))
        err = subspace_error(C_true, C_est)
        assert 0 <= err <= 1

    def test_returns_float(self):
        """Should return a Python float, not numpy scalar."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        result = subspace_error(C, C)
        assert isinstance(result, float)


class TestRelativeL2Error:
    """Tests for relative_l2_error metric."""

    def test_identical_vectors_returns_zero(self):
        """Identical vectors should have zero error."""
        v = np.array([1.0, 2.0, 3.0])
        tols = _testing_tols(v.dtype)
        np.testing.assert_allclose(relative_l2_error(v, v), 0.0, **tols)

    def test_known_error(self):
        """Test with known error value."""
        v_true = np.array([3.0, 4.0])  # norm = 5
        v_est = np.array([0.0, 0.0])  # error = 5
        # normalized error = 5 / 5 = 1.0
        tols = _testing_tols(v_true.dtype)
        np.testing.assert_allclose(relative_l2_error(v_true, v_est), 1.0, **tols)

    def test_handles_multidimensional(self):
        """Should flatten multidimensional arrays."""
        v_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        v_est = np.array([[1.0, 2.0], [3.0, 4.0]])
        tols = _testing_tols(v_true.dtype)
        np.testing.assert_allclose(relative_l2_error(v_true, v_est), 0.0, **tols)

    def test_returns_float(self):
        """Should return a Python float, not numpy scalar."""
        v = np.array([1.0, 2.0, 3.0])
        result = relative_l2_error(v, v)
        assert isinstance(result, float)


class TestDenoisedR2:
    """Tests for denoised_r2 metric."""

    def test_perfect_recovery_returns_one(self):
        """Perfect parameter recovery should give R² = 1."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        X = rng.standard_normal((3, 50))
        d = rng.standard_normal(10)

        tols = _testing_tols(C.dtype)
        np.testing.assert_allclose(denoised_r2(C, X, d, C, X, d), 1.0, **tols)

    def test_wrong_mean_reduces_r2(self):
        """Wrong mean estimate should reduce R²."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        X = rng.standard_normal((3, 50))
        d_true = rng.standard_normal(10)
        d_wrong = d_true + 10.0  # Large offset

        r2 = denoised_r2(C, X, d_true, C, X, d_wrong)
        assert r2 < 1.0

    def test_noisy_estimate_reduces_r2(self):
        """Noisy parameter estimates should reduce R²."""
        rng = np.random.default_rng(42)
        C_true = rng.standard_normal((10, 3))
        X_true = rng.standard_normal((3, 50))
        d_true = rng.standard_normal(10)

        # Add noise to estimates
        C_est = C_true + 0.5 * rng.standard_normal((10, 3))
        X_est = X_true + 0.5 * rng.standard_normal((3, 50))
        d_est = d_true + 0.5 * rng.standard_normal(10)

        r2 = denoised_r2(C_true, X_true, d_true, C_est, X_est, d_est)
        assert r2 < 1.0

    def test_returns_float(self):
        """Should return a Python float, not numpy scalar."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal((10, 3))
        X = rng.standard_normal((3, 50))
        d = rng.standard_normal(10)

        result = denoised_r2(C, X, d, C, X, d)
        assert isinstance(result, float)
