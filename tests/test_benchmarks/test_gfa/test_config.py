"""Tests for GFA benchmark configuration."""

import pytest

from benchmarks.gfa.config import (
    DIMENSIONALITY_CONFIG,
    RECOVERY_CONFIG,
    RUNTIME_CONFIG,
    SweepConfig,
)


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_is_swept_with_values(self):
        """SweepConfig with values should report is_swept=True."""
        sc = SweepConfig(default=200, values=(50, 100, 200))
        assert sc.is_swept is True

    def test_is_swept_without_values(self):
        """SweepConfig without values should report is_swept=False."""
        sc = SweepConfig(default=1.0)
        assert sc.is_swept is False

    def test_max_value_with_values(self):
        """max_value should return maximum of values tuple."""
        sc = SweepConfig(default=200, values=(50, 100, 500, 200))
        assert sc.max_value == 500

    def test_max_value_without_values(self):
        """max_value should return default when no values."""
        sc = SweepConfig(default=42)
        assert sc.max_value == 42

    def test_frozen(self):
        """SweepConfig should be immutable."""
        sc = SweepConfig(default=200)
        with pytest.raises(AttributeError):
            sc.default = 100


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    # --- Seed method tests ---

    def test_get_data_seed_deterministic(self):
        """Same inputs should produce same seed."""
        config = RUNTIME_CONFIG
        seed1 = config.get_data_seed("n_samples", 0)
        seed2 = config.get_data_seed("n_samples", 0)
        assert seed1 == seed2

    def test_get_data_seed_different_sweeps(self):
        """Different sweeps should produce different seeds."""
        config = RUNTIME_CONFIG
        seed_n_samples = config.get_data_seed("n_samples", 0)
        seed_y_dim = config.get_data_seed("y_dim_per_group", 0)
        assert seed_n_samples != seed_y_dim

    def test_get_data_seed_different_runs(self):
        """Different run indices should produce different seeds."""
        config = RUNTIME_CONFIG
        seed_run0 = config.get_data_seed("n_samples", 0)
        seed_run1 = config.get_data_seed("n_samples", 1)
        assert seed_run0 != seed_run1

    def test_get_data_seed_with_sweep_value_idx(self):
        """Structural sweeps should include sweep_value_idx in seed."""
        config = RUNTIME_CONFIG
        seed_val0 = config.get_data_seed("x_dim", 0, sweep_value_idx=0)
        seed_val1 = config.get_data_seed("x_dim", 0, sweep_value_idx=1)
        assert seed_val0 != seed_val1

    def test_get_data_seed_invalid_sweep(self):
        """Invalid sweep name should raise ValueError."""
        config = RUNTIME_CONFIG
        with pytest.raises(ValueError, match="Unknown sweep"):
            config.get_data_seed("invalid_sweep", 0)

    def test_get_obs_seed_deterministic(self):
        """Same inputs should produce same obs seed."""
        config = RUNTIME_CONFIG
        seed1 = config.get_obs_seed("snr", 0, 0)
        seed2 = config.get_obs_seed("snr", 0, 0)
        assert seed1 == seed2

    def test_get_obs_seed_different_sweep_values(self):
        """Different sweep values should produce different obs seeds."""
        config = RUNTIME_CONFIG
        seed_val0 = config.get_obs_seed("snr", 0, 0)
        seed_val1 = config.get_obs_seed("snr", 0, 1)
        assert seed_val0 != seed_val1

    def test_get_fit_seed_deterministic(self):
        """Same inputs should produce same fit seed."""
        config = RUNTIME_CONFIG
        seed1 = config.get_fit_seed("n_samples", 0, 0)
        seed2 = config.get_fit_seed("n_samples", 0, 0)
        assert seed1 == seed2

    def test_get_fit_seed_different_sweep_values(self):
        """Different sweep values should produce different fit seeds."""
        config = RUNTIME_CONFIG
        seed_val0 = config.get_fit_seed("n_samples", 0, 0)
        seed_val1 = config.get_fit_seed("n_samples", 0, 1)
        assert seed_val0 != seed_val1

    def test_data_obs_fit_seeds_differ(self):
        """Data, obs, and fit seeds should differ for same inputs."""
        config = RUNTIME_CONFIG
        # For structural sweep with all indices matching
        data_seed = config.get_data_seed("x_dim", 0, sweep_value_idx=0)
        obs_seed = config.get_obs_seed("x_dim", 0, 0)
        fit_seed = config.get_fit_seed("x_dim", 0, 0)
        assert data_seed != obs_seed
        assert data_seed != fit_seed
        assert obs_seed != fit_seed

    # --- get_sweep_config tests ---

    def test_get_sweep_config(self):
        """get_sweep_config should return correct SweepConfig."""
        config = RUNTIME_CONFIG
        n_samples_config = config.get_sweep_config("n_samples")
        assert n_samples_config is config.n_samples

    def test_get_sweep_config_y_dim_per_group(self):
        """y_dim_per_group should return y_dim_per_group field."""
        config = RUNTIME_CONFIG
        y_dim_config = config.get_sweep_config("y_dim_per_group")
        assert y_dim_config is config.y_dim_per_group

    def test_get_sweep_config_invalid(self):
        """Invalid sweep name should raise ValueError."""
        config = RUNTIME_CONFIG
        with pytest.raises(ValueError, match="Unknown sweep"):
            config.get_sweep_config("invalid")

    # --- get_active_sweeps tests ---

    def test_get_active_sweeps(self):
        """get_active_sweeps should return list of swept factors."""
        config = RUNTIME_CONFIG
        active = config.get_active_sweeps()
        # RUNTIME_CONFIG has all factors swept except snr
        assert "n_samples" in active
        assert "y_dim_per_group" in active
        assert "x_dim" in active
        assert "n_groups" in active
        assert "snr" not in active

    # --- Frozen tests ---

    def test_frozen(self):
        """BenchmarkConfig should be immutable."""
        with pytest.raises(AttributeError):
            RUNTIME_CONFIG.n_runs = 5


class TestDimensionalityConfig:
    """Tests for DimensionalityConfig dataclass."""

    def test_frozen(self):
        """DimensionalityConfig should be immutable."""
        with pytest.raises(AttributeError):
            DIMENSIONALITY_CONFIG.n_runs = 5

    def test_max_n_samples(self):
        """max_n_samples should return maximum of n_samples_values."""
        assert DIMENSIONALITY_CONFIG.max_n_samples == 10_000

    def test_y_dims_shape(self):
        """y_dims should be array of length n_groups."""
        y_dims = DIMENSIONALITY_CONFIG.y_dims
        assert y_dims.shape == (1,)
        assert y_dims[0] == 50

    def test_default_values(self):
        """Check default preset values."""
        c = DIMENSIONALITY_CONFIG
        assert c.x_dim_true == 10
        assert c.x_dim_init == 20
        assert c.n_groups == 1
        assert c.y_dim_per_group == 50
        assert c.n_runs == 10

    # --- Seed determinism ---

    def test_data_seed_deterministic(self):
        """Same run_idx should produce same data seed."""
        seed1 = DIMENSIONALITY_CONFIG.get_data_seed(0)
        seed2 = DIMENSIONALITY_CONFIG.get_data_seed(0)
        assert seed1 == seed2

    def test_data_seed_differs_by_run(self):
        """Different run indices should produce different data seeds."""
        seed0 = DIMENSIONALITY_CONFIG.get_data_seed(0)
        seed1 = DIMENSIONALITY_CONFIG.get_data_seed(1)
        assert seed0 != seed1

    def test_obs_seed_differs_by_snr(self):
        """Different SNR indices should produce different obs seeds."""
        seed0 = DIMENSIONALITY_CONFIG.get_obs_seed(0, 0)
        seed1 = DIMENSIONALITY_CONFIG.get_obs_seed(0, 1)
        assert seed0 != seed1

    def test_fit_seed_differs_by_n_samples(self):
        """Different n_samples indices should produce different fit seeds."""
        seed0 = DIMENSIONALITY_CONFIG.get_fit_seed(0, 0, 0)
        seed1 = DIMENSIONALITY_CONFIG.get_fit_seed(0, 0, 1)
        assert seed0 != seed1

    def test_data_obs_fit_seeds_differ(self):
        """Data, obs, and fit seeds should differ for same run."""
        data = DIMENSIONALITY_CONFIG.get_data_seed(0)
        obs = DIMENSIONALITY_CONFIG.get_obs_seed(0, 0)
        fit = DIMENSIONALITY_CONFIG.get_fit_seed(0, 0, 0)
        assert data != obs
        assert data != fit
        assert obs != fit


class TestPresetConfigs:
    """Tests for preset configuration instances."""

    def test_runtime_config_snr_not_swept(self):
        """RUNTIME_CONFIG should have SNR fixed (not swept)."""
        assert RUNTIME_CONFIG.snr.is_swept is False

    def test_recovery_config_snr_swept(self):
        """RECOVERY_CONFIG should have SNR swept."""
        assert RECOVERY_CONFIG.snr.is_swept is True

    def test_dimensionality_config_overspecified(self):
        """DIMENSIONALITY_CONFIG x_dim_init should exceed x_dim_true."""
        assert DIMENSIONALITY_CONFIG.x_dim_init > DIMENSIONALITY_CONFIG.x_dim_true
