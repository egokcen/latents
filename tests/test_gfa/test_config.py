"""Test configuration validation for GFA module."""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from latents.gfa import GFAFitConfig
from latents.observation import ObsParamsHyperPrior, ObsParamsHyperPriorStructured


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
        assert config.verbose is False
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
            verbose=True,
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
        assert config.verbose is True
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

    def test_random_seed_validation(self):
        """Test random_seed must be non-negative integer or None."""
        # Valid cases
        GFAFitConfig(random_seed=None)
        GFAFitConfig(random_seed=0)
        GFAFitConfig(random_seed=42)

        # Invalid cases
        with pytest.raises(ValueError, match="random_seed must be a non-negative"):
            GFAFitConfig(random_seed=-1)
        with pytest.raises(ValueError, match="random_seed must be a non-negative"):
            GFAFitConfig(random_seed=3.14)


class TestObsParamsHyperPrior:
    """Tests for ObsParamsHyperPrior dataclass."""

    def test_defaults(self):
        """Test default values are set correctly."""
        hp = ObsParamsHyperPrior()
        assert hp.beta_d == 1e-12
        assert hp.a_alpha == 1e-12
        assert hp.b_alpha == 1e-12
        assert hp.a_phi == 1e-12
        assert hp.b_phi == 1e-12

    def test_custom_values(self):
        """Test custom values are accepted."""
        hp = ObsParamsHyperPrior(
            beta_d=1.0, a_alpha=0.1, b_alpha=0.1, a_phi=1.0, b_phi=1.0
        )
        assert hp.beta_d == 1.0
        assert hp.a_alpha == 0.1
        assert hp.b_alpha == 0.1
        assert hp.a_phi == 1.0
        assert hp.b_phi == 1.0

    def test_frozen(self):
        """Test that ObsParamsHyperPrior is immutable."""
        hp = ObsParamsHyperPrior()
        with pytest.raises(FrozenInstanceError):
            hp.beta_d = 1.0

    def test_beta_d_validation(self):
        """Test beta_d must be > 0."""
        with pytest.raises(ValueError, match="beta_d must be a positive number"):
            ObsParamsHyperPrior(beta_d=0)
        with pytest.raises(ValueError, match="beta_d must be a positive number"):
            ObsParamsHyperPrior(beta_d=-1)

    def test_a_alpha_validation(self):
        """Test a_alpha must be > 0."""
        with pytest.raises(ValueError, match="a_alpha must be a positive number"):
            ObsParamsHyperPrior(a_alpha=0)
        with pytest.raises(ValueError, match="a_alpha must be a positive number"):
            ObsParamsHyperPrior(a_alpha=-0.1)

    def test_b_alpha_validation(self):
        """Test b_alpha must be > 0."""
        with pytest.raises(ValueError, match="b_alpha must be a positive number"):
            ObsParamsHyperPrior(b_alpha=0)
        with pytest.raises(ValueError, match="b_alpha must be a positive number"):
            ObsParamsHyperPrior(b_alpha=-0.1)

    def test_a_phi_validation(self):
        """Test a_phi must be > 0."""
        with pytest.raises(ValueError, match="a_phi must be a positive number"):
            ObsParamsHyperPrior(a_phi=0)
        with pytest.raises(ValueError, match="a_phi must be a positive number"):
            ObsParamsHyperPrior(a_phi=-1)

    def test_b_phi_validation(self):
        """Test b_phi must be > 0."""
        with pytest.raises(ValueError, match="b_phi must be a positive number"):
            ObsParamsHyperPrior(b_phi=0)
        with pytest.raises(ValueError, match="b_phi must be a positive number"):
            ObsParamsHyperPrior(b_phi=-1)


class TestObsParamsHyperPriorStructured:
    """Tests for ObsParamsHyperPriorStructured dataclass."""

    def test_required_arrays(self):
        """Test that a_alpha and b_alpha are required."""
        # Must provide both arrays
        a_alpha = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_alpha = np.array([[1.0, 1.0], [1.0, 1.0]])

        hp = ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha)
        np.testing.assert_array_equal(hp.a_alpha, a_alpha)
        np.testing.assert_array_equal(hp.b_alpha, b_alpha)

    def test_defaults_for_scalars(self):
        """Test scalar parameters have defaults."""
        a_alpha = np.array([[1.0]])
        b_alpha = np.array([[1.0]])

        hp = ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha)
        assert hp.beta_d == 1.0
        assert hp.a_phi == 1.0
        assert hp.b_phi == 1.0

    def test_custom_scalars(self):
        """Test custom scalar values are accepted."""
        a_alpha = np.array([[1.0]])
        b_alpha = np.array([[1.0]])

        hp = ObsParamsHyperPriorStructured(
            a_alpha=a_alpha, b_alpha=b_alpha, beta_d=2.0, a_phi=0.5, b_phi=0.5
        )
        assert hp.beta_d == 2.0
        assert hp.a_phi == 0.5
        assert hp.b_phi == 0.5

    def test_frozen(self):
        """Test that ObsParamsHyperPriorStructured is immutable."""
        a_alpha = np.array([[1.0]])
        b_alpha = np.array([[1.0]])
        hp = ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha)

        with pytest.raises(FrozenInstanceError):
            hp.beta_d = 2.0

    def test_shape_mismatch_validation(self):
        """Test a_alpha and b_alpha must have matching shapes."""
        a_alpha = np.array([[1.0, 2.0]])
        b_alpha = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(
            ValueError, match=r"a_alpha shape.*must match b_alpha shape"
        ):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha)

    def test_beta_d_validation(self):
        """Test beta_d must be > 0."""
        a_alpha = np.array([[1.0]])
        b_alpha = np.array([[1.0]])

        with pytest.raises(ValueError, match="beta_d must be > 0"):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha, beta_d=0)
        with pytest.raises(ValueError, match="beta_d must be > 0"):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha, beta_d=-1)

    def test_a_phi_validation(self):
        """Test a_phi must be > 0."""
        a_alpha = np.array([[1.0]])
        b_alpha = np.array([[1.0]])

        with pytest.raises(ValueError, match="a_phi must be > 0"):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha, a_phi=0)
        with pytest.raises(ValueError, match="a_phi must be > 0"):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha, a_phi=-1)

    def test_b_phi_validation(self):
        """Test b_phi must be > 0."""
        a_alpha = np.array([[1.0]])
        b_alpha = np.array([[1.0]])

        with pytest.raises(ValueError, match="b_phi must be > 0"):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha, b_phi=0)
        with pytest.raises(ValueError, match="b_phi must be > 0"):
            ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha, b_phi=-1)
