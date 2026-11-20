"""Test configuration validation for pPCA module."""

import pytest
from dataclasses import FrozenInstanceError


from latents.ppca import pPCAFitConfig
from latents.ppca.config import pPCASimConfig


class TestpPCAFitConfig:
    """Tests for pPCAFitConfig dataclass."""

    def test_defaults(self):
        """Test default values are set correctly."""
        config = pPCAFitConfig()
        assert config.x_dim == 5
        assert config.fit_tol == 1e-8
        assert config.max_iter == 1_000_000
        assert config.save_x is False
        assert config.save_fit_progress is True
        assert config.random_seed is None
        assert config.min_var_frac == 0.001

    def test_custom_values(self):
        """Test custom values are accepted."""
        config = pPCAFitConfig(
            x_dim=10,
            fit_tol=1e-6,
            max_iter=5000,
            save_x=True,
            save_fit_progress=False,
            random_seed=42,
            min_var_frac=0.01,
        )
        assert config.x_dim == 10
        assert config.fit_tol == 1e-6
        assert config.max_iter == 5000
        assert config.save_x is True
        assert config.save_fit_progress is False
        assert config.random_seed == 42
        assert config.min_var_frac == 0.01

    def test_frozen(self):
        """Test that config is immutable."""
        config = pPCAFitConfig()
        with pytest.raises(FrozenInstanceError):
            config.x_dim = 10

    def test_x_dim_validation(self):
        """Test x_dim must be integer >= 1."""
        with pytest.raises(ValueError, match="x_dim must be an integer >= 1"):
            pPCAFitConfig(x_dim=0)
        with pytest.raises(ValueError, match="x_dim must be an integer >= 1"):
            pPCAFitConfig(x_dim=-1)
        with pytest.raises(ValueError, match="x_dim must be an integer >= 1"):
            pPCAFitConfig(x_dim=1.5)

    def test_fit_tol_validation(self):
        """Test fit_tol must be > 0."""
        with pytest.raises(ValueError, match="fit_tol must be > 0"):
            pPCAFitConfig(fit_tol=0)
        with pytest.raises(ValueError, match="fit_tol must be > 0"):
            pPCAFitConfig(fit_tol=-1e-8)

    def test_max_iter_validation(self):
        """Test max_iter must be integer >= 1."""
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            pPCAFitConfig(max_iter=0)
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            pPCAFitConfig(max_iter=-100)
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            pPCAFitConfig(max_iter=100.5)

    def test_min_var_frac_validation(self):
        """Test min_var_frac must be in (0, 1)."""
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            pPCAFitConfig(min_var_frac=0)
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            pPCAFitConfig(min_var_frac=1)
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            pPCAFitConfig(min_var_frac=-0.1)
        with pytest.raises(ValueError, match="min_var_frac must be in"):
            pPCAFitConfig(min_var_frac=1.5)

    def test_random_seed_int_valid(self):
        """Test random_seed accepts non-negative integers and None."""
        pPCAFitConfig(random_seed=None)
        pPCAFitConfig(random_seed=0)
        pPCAFitConfig(random_seed=42)

    def test_random_seed_sequence_valid(self):
        """Test random_seed accepts sequences of non-negative integers."""
        config = pPCAFitConfig(random_seed=[0, 1, 42])
        assert config.random_seed == [0, 1, 42]

        # Tuples also work
        config = pPCAFitConfig(random_seed=(1, 2, 3))
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
            pPCAFitConfig(random_seed=seed)


class TestpPCASimConfig:
    """Tests for pPCASimConfig random_seed validation.

    pPCASimConfig and pPCAFitConfig share the same validation helper, so we
    only test a subset of cases here to confirm the integration.
    """

    def test_random_seed_sequence_valid(self):
        """Test random_seed accepts sequences of non-negative integers."""
        config = pPCASimConfig(
            n_samples=50,
            x_dim=3,
            y_dim=10,
            snr=1.0,
            random_seed=[0, 1, 42],
        )
        assert config.random_seed == [0, 1, 42]

    def test_random_seed_sequence_invalid(self):
        """Test random_seed rejects invalid sequences."""
        with pytest.raises(ValueError, match="must not be empty"):
            pPCASimConfig(
                n_samples=50,
                x_dim=3,
                y_dim=5,
                snr=1.0,
                random_seed=[],
            )

    def test_y_dim_invalid(self):
        """Test y_dim rejects invalid entries."""
        with pytest.raises(ValueError, match=">= 1"):
            pPCASimConfig(
                n_samples=50,
                x_dim=3,
                y_dim=2,
                snr=1.0,
                random_seed=[42],
            )
