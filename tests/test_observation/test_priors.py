"""Test observation model prior classes."""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from latents.observation import ObsParamsHyperPrior, ObsParamsHyperPriorStructured


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
