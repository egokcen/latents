"""Tests for GFA benchmark data generation and subsetting."""

import numpy as np
import pytest

from benchmarks.gfa.config import RUNTIME_CONFIG
from benchmarks.gfa.data import (
    BENCHMARK_HYPERPRIOR,
    build_y_dims,
    generate_ground_truth,
    resample_at_snr,
    subset_by_samples,
    subset_by_y_dim,
)


class TestBuildYDims:
    """Tests for build_y_dims function."""

    def test_default_case(self):
        """Non-y_dim/n_groups sweeps should use defaults."""
        y_dims = build_y_dims(RUNTIME_CONFIG, "n_samples", 500)
        expected_n_groups = int(RUNTIME_CONFIG.n_groups.default)
        expected_y_dim = int(RUNTIME_CONFIG.y_dim_per_group.default)

        assert y_dims.shape == (expected_n_groups,)
        assert np.all(y_dims == expected_y_dim)

    def test_y_dim_per_group_sweep(self):
        """y_dim_per_group sweep should use swept value for per-group dim."""
        y_dims = build_y_dims(RUNTIME_CONFIG, "y_dim_per_group", 25)
        expected_n_groups = int(RUNTIME_CONFIG.n_groups.default)

        assert y_dims.shape == (expected_n_groups,)
        assert np.all(y_dims == 25)

    def test_n_groups_sweep(self):
        """n_groups sweep should divide total_y_dim evenly."""
        # RUNTIME_CONFIG.n_groups_total_y_dim = 100
        y_dims = build_y_dims(RUNTIME_CONFIG, "n_groups", 5)

        assert y_dims.shape == (5,)
        assert np.all(y_dims == 20)  # 100 / 5 = 20
        assert y_dims.sum() == RUNTIME_CONFIG.n_groups_total_y_dim

    def test_n_groups_sweep_indivisible(self):
        """n_groups sweep should raise if total not divisible."""
        # 100 is not divisible by 3
        with pytest.raises(ValueError, match="must be divisible"):
            build_y_dims(RUNTIME_CONFIG, "n_groups", 3)

    def test_returns_intp_dtype(self):
        """y_dims should have integer dtype."""
        y_dims = build_y_dims(RUNTIME_CONFIG, "n_samples", 100)
        assert y_dims.dtype == np.intp


class TestGenerateGroundTruth:
    """Tests for generate_ground_truth function."""

    def test_output_shapes(self):
        """Generated data should have correct shapes."""
        y_dims = np.array([20, 30])
        result = generate_ground_truth(
            n_samples=100,
            y_dims=y_dims,
            x_dim=5,
            snr=1.0,
            seed=42,
        )

        assert result.observations.data.shape == (50, 100)
        assert result.latents.data.shape == (5, 100)
        assert result.obs_params.C.shape == (50, 5)
        assert result.obs_params.d.shape == (50,)
        assert result.obs_params.phi.shape == (50,)
        assert result.obs_params.alpha.shape == (2, 5)

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        y_dims = np.array([10, 10])
        result1 = generate_ground_truth(50, y_dims, 3, 1.0, seed=123)
        result2 = generate_ground_truth(50, y_dims, 3, 1.0, seed=123)

        np.testing.assert_array_equal(
            result1.observations.data, result2.observations.data
        )
        np.testing.assert_array_equal(result1.latents.data, result2.latents.data)

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        y_dims = np.array([10, 10])
        result1 = generate_ground_truth(50, y_dims, 3, 1.0, seed=1)
        result2 = generate_ground_truth(50, y_dims, 3, 1.0, seed=2)

        assert not np.allclose(result1.observations.data, result2.observations.data)

    def test_list_seed_reproducible(self):
        """List seeds (from BenchmarkConfig methods) should be reproducible."""
        y_dims = np.array([10, 10])
        seed = [0, 1, 0, 42]  # [sweep_idx, run_idx, stream, base_seed]
        result1 = generate_ground_truth(50, y_dims, 3, 1.0, seed=seed)
        result2 = generate_ground_truth(50, y_dims, 3, 1.0, seed=seed)

        np.testing.assert_array_equal(
            result1.observations.data, result2.observations.data
        )

    def test_list_seeds_differ_by_component(self):
        """Different list seed components should produce different results."""
        y_dims = np.array([10, 10])
        seed1 = [0, 0, 0, 42]
        seed2 = [0, 1, 0, 42]  # Different run_idx

        result1 = generate_ground_truth(50, y_dims, 3, 1.0, seed=seed1)
        result2 = generate_ground_truth(50, y_dims, 3, 1.0, seed=seed2)

        assert not np.allclose(result1.observations.data, result2.observations.data)

    def test_uses_benchmark_hyperprior(self):
        """Should use BENCHMARK_HYPERPRIOR (Gamma(1,1) priors)."""
        y_dims = np.array([10, 10])
        result = generate_ground_truth(50, y_dims, 3, 1.0, seed=42)

        # Verify hyperprior is the benchmark one (not the 1e-12 defaults)
        assert result.hyperprior.a_alpha == BENCHMARK_HYPERPRIOR.a_alpha
        assert result.hyperprior.a_phi == BENCHMARK_HYPERPRIOR.a_phi


class TestSubsetBySamples:
    """Tests for subset_by_samples function."""

    @pytest.fixture
    def base_result(self):
        """Generate a base result to subset."""
        y_dims = np.array([20, 20])
        return generate_ground_truth(100, y_dims, 5, 1.0, seed=42)

    def test_output_shapes(self, base_result):
        """Subsetted data should have correct shapes."""
        Y_sub, X_sub = subset_by_samples(base_result, n_samples=30)

        assert Y_sub.data.shape == (40, 30)
        assert X_sub.data.shape == (5, 30)

    def test_dims_preserved(self, base_result):
        """Group dimensions should be preserved."""
        Y_sub, _ = subset_by_samples(base_result, n_samples=30)

        np.testing.assert_array_equal(Y_sub.dims, base_result.observations.dims)

    def test_correct_samples_kept(self, base_result):
        """Should keep first n_samples."""
        Y_sub, X_sub = subset_by_samples(base_result, n_samples=30)

        np.testing.assert_array_equal(Y_sub.data, base_result.observations.data[:, :30])
        np.testing.assert_array_equal(X_sub.data, base_result.latents.data[:, :30])

    def test_returns_copies(self, base_result):
        """Subsetted arrays should be copies, not views."""
        Y_sub, X_sub = subset_by_samples(base_result, n_samples=30)

        # Modify subsets
        Y_sub.data[0, 0] = 999.0
        X_sub.data[0, 0] = 999.0

        # Originals should be unchanged
        assert base_result.observations.data[0, 0] != 999.0
        assert base_result.latents.data[0, 0] != 999.0


class TestSubsetByYDim:
    """Tests for subset_by_y_dim function."""

    @pytest.fixture
    def base_result(self):
        """Generate a base result to subset."""
        y_dims = np.array([30, 30])
        return generate_ground_truth(50, y_dims, 5, 1.0, seed=42)

    def test_output_shapes(self, base_result):
        """Subsetted data should have correct shapes."""
        Y_sub, obs_sub = subset_by_y_dim(base_result, y_dim_per_group=10)

        assert Y_sub.data.shape == (20, 50)  # 2 groups * 10 dims
        assert obs_sub.C.shape == (20, 5)
        assert obs_sub.d.shape == (20,)
        assert obs_sub.phi.shape == (20,)

    def test_alpha_unchanged(self, base_result):
        """Alpha should be unchanged (n_groups x x_dim)."""
        _, obs_sub = subset_by_y_dim(base_result, y_dim_per_group=10)

        assert obs_sub.alpha.shape == base_result.obs_params.alpha.shape
        np.testing.assert_array_equal(obs_sub.alpha, base_result.obs_params.alpha)

    def test_dims_updated(self, base_result):
        """Group dimensions should be updated."""
        Y_sub, obs_sub = subset_by_y_dim(base_result, y_dim_per_group=10)

        np.testing.assert_array_equal(Y_sub.dims, [10, 10])
        np.testing.assert_array_equal(obs_sub.y_dims, [10, 10])

    def test_correct_dims_kept(self, base_result):
        """Should keep first y_dim_per_group dims from each group."""
        Y_sub, _ = subset_by_y_dim(base_result, y_dim_per_group=10)

        # Group 0: dims 0-9 of original
        np.testing.assert_array_equal(
            Y_sub.data[:10, :], base_result.observations.data[:10, :]
        )
        # Group 1: dims 30-39 of original (first 10 of second group)
        np.testing.assert_array_equal(
            Y_sub.data[10:, :], base_result.observations.data[30:40, :]
        )

    def test_returns_copies(self, base_result):
        """Subsetted arrays should be copies, not views."""
        Y_sub, obs_sub = subset_by_y_dim(base_result, y_dim_per_group=10)

        # Modify subsets
        Y_sub.data[0, 0] = 999.0
        obs_sub.C[0, 0] = 999.0

        # Originals should be unchanged
        assert base_result.observations.data[0, 0] != 999.0
        assert base_result.obs_params.C[0, 0] != 999.0


class TestResampleAtSnr:
    """Tests for resample_at_snr function."""

    @pytest.fixture
    def base_result(self):
        """Generate a base result to resample."""
        y_dims = np.array([20, 20])
        return generate_ground_truth(50, y_dims, 5, 1.0, seed=42)

    def test_output_shapes(self, base_result):
        """Resampled data should have same shapes as original."""
        Y_snr, obs_snr = resample_at_snr(base_result, snr=2.0, obs_seed=99)

        assert Y_snr.data.shape == base_result.observations.data.shape
        assert obs_snr.C.shape == base_result.obs_params.C.shape
        assert obs_snr.phi.shape == base_result.obs_params.phi.shape

    def test_c_d_alpha_unchanged(self, base_result):
        """C, d, and alpha should be unchanged from original."""
        _, obs_snr = resample_at_snr(base_result, snr=2.0, obs_seed=99)

        np.testing.assert_array_equal(obs_snr.C, base_result.obs_params.C)
        np.testing.assert_array_equal(obs_snr.d, base_result.obs_params.d)
        np.testing.assert_array_equal(obs_snr.alpha, base_result.obs_params.alpha)

    def test_phi_changes_with_snr(self, base_result):
        """Different SNR should produce different phi values."""
        _, obs_snr_low = resample_at_snr(base_result, snr=0.1, obs_seed=99)
        _, obs_snr_high = resample_at_snr(base_result, snr=10.0, obs_seed=99)

        # Higher SNR → higher phi (less noise variance = higher precision)
        assert np.mean(obs_snr_high.phi) > np.mean(obs_snr_low.phi)

    def test_observations_differ_across_seeds(self, base_result):
        """Different obs_seeds should produce different observations."""
        Y1, _ = resample_at_snr(base_result, snr=1.0, obs_seed=1)
        Y2, _ = resample_at_snr(base_result, snr=1.0, obs_seed=2)

        assert not np.allclose(Y1.data, Y2.data)

    def test_reproducible_with_same_seed(self, base_result):
        """Same obs_seed should produce identical observations."""
        Y1, _ = resample_at_snr(base_result, snr=1.0, obs_seed=42)
        Y2, _ = resample_at_snr(base_result, snr=1.0, obs_seed=42)

        np.testing.assert_array_equal(Y1.data, Y2.data)
