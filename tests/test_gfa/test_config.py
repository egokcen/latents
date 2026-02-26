"""Test configuration validation for GFA module."""

import pytest
from dataclasses import FrozenInstanceError

import numpy as np

from latents.gfa import GFAFitConfig
from latents.gfa.config import GFASimConfig


class TestGFAFitConfig:
    """Tests for GFAFitConfig dataclass."""

    def test_defaults(self):
        """Test default values are set correctly."""
        config = GFAFitConfig()
        assert config.x_dim_init == 1
        assert config.fit_tol == 1e-8
        assert config.max_iter == 1_000_000
        assert config.prune_x is True
        assert config.prune_tol == 1e-7
        assert config.save_x is False
        assert config.save_c_cov is False
        assert config.save_fit_progress is True
        assert config.random_seed is None
        assert config.min_var_frac == 0.001

    def test_custom_values(self):
        """Test custom values are accepted."""
        config = GFAFitConfig(
            x_dim_init=10,
            fit_tol=1e-6,
            max_iter=5000,
            prune_x=False,
            prune_tol=1e-5,
            save_x=True,
            save_c_cov=True,
            save_fit_progress=False,
            random_seed=42,
            min_var_frac=0.01,
        )
        assert config.x_dim_init == 10
        assert config.fit_tol == 1e-6
        assert config.max_iter == 5000
        assert config.prune_x is False
        assert config.prune_tol == 1e-5
        assert config.save_x is True
        assert config.save_c_cov is True
        assert config.save_fit_progress is False
        assert config.random_seed == 42
        assert config.min_var_frac == 0.01

    def test_frozen(self):
        """Test that config is immutable."""
        config = GFAFitConfig()
        with pytest.raises(FrozenInstanceError):
            config.x_dim_init = 10

    def test_x_dim_init_validation(self):
        """Test x_dim_init must be integer >= 1."""
        with pytest.raises(ValueError, match="x_dim_init must be an integer >= 1"):
            GFAFitConfig(x_dim_init=0)
        with pytest.raises(ValueError, match="x_dim_init must be an integer >= 1"):
            GFAFitConfig(x_dim_init=-1)
        with pytest.raises(ValueError, match="x_dim_init must be an integer >= 1"):
            GFAFitConfig(x_dim_init=1.5)

    def test_fit_tol_validation(self):
        """Test fit_tol must be > 0."""
        with pytest.raises(ValueError, match="fit_tol must be > 0"):
            GFAFitConfig(fit_tol=0)
        with pytest.raises(ValueError, match="fit_tol must be > 0"):
            GFAFitConfig(fit_tol=-1e-8)

    def test_max_iter_validation(self):
        """Test max_iter must be integer >= 1."""
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            GFAFitConfig(max_iter=0)
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            GFAFitConfig(max_iter=-100)
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            GFAFitConfig(max_iter=100.5)

    def test_prune_tol_validation(self):
        """Test prune_tol must be > 0."""
        with pytest.raises(ValueError, match="prune_tol must be > 0"):
            GFAFitConfig(prune_tol=0)
        with pytest.raises(ValueError, match="prune_tol must be > 0"):
            GFAFitConfig(prune_tol=-1e-7)

    def test_min_var_frac_validation(self):
        """Test min_var_frac must be in (0, 1)."""
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            GFAFitConfig(min_var_frac=0)
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            GFAFitConfig(min_var_frac=1)
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            GFAFitConfig(min_var_frac=-0.1)
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            GFAFitConfig(min_var_frac=1.5)

    def test_random_seed_int_valid(self):
        """Test random_seed accepts non-negative integers and None."""
        GFAFitConfig(random_seed=None)
        GFAFitConfig(random_seed=0)
        GFAFitConfig(random_seed=42)

    def test_random_seed_sequence_valid(self):
        """Test random_seed accepts sequences of non-negative integers."""
        config = GFAFitConfig(random_seed=[0, 1, 42])
        assert config.random_seed == [0, 1, 42]

        # Tuples also work
        config = GFAFitConfig(random_seed=(1, 2, 3))
        assert config.random_seed == (1, 2, 3)

    @pytest.mark.parametrize(
        ("seed", "error", "match"),
        [
            (-1, ValueError, "non-negative"),
            ([], ValueError, "must not be empty"),
            ([1, -2, 3], ValueError, "non-negative"),
            ([1, 2.5, 3], TypeError, "must contain integers"),
            (3.14, TypeError, "int, sequence of ints, or None"),
            ("42", TypeError, "int, sequence of ints, or None"),
        ],
    )
    def test_random_seed_invalid(self, seed, error, match):
        """Test random_seed rejects invalid inputs."""
        with pytest.raises(error, match=match):
            GFAFitConfig(random_seed=seed)


class TestGFASimConfig:
    """Tests for GFASimConfig random_seed validation.

    GFASimConfig and GFAFitConfig share the same validation helper, so we
    only test a subset of cases here to confirm the integration.
    """

    def test_random_seed_sequence_valid(self):
        """Test random_seed accepts sequences of non-negative integers."""
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([10]),
            x_dim=3,
            random_seed=[0, 1, 42],
        )
        assert config.random_seed == [0, 1, 42]

    def test_random_seed_sequence_invalid(self):
        """Test random_seed rejects invalid sequences."""
        with pytest.raises(ValueError, match="must not be empty"):
            GFASimConfig(
                n_samples=50,
                y_dims=np.array([10]),
                x_dim=3,
                random_seed=[],
            )
