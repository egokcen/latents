"""Test observation model prior classes."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from latents.observation import (
    ObsParamsHyperPrior,
    ObsParamsHyperPriorStructured,
    ObsParamsPrior,
)


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


# --------------------------------------------------------------------------- #
# Shared test dimensions: 2 groups, y_dims=[3,2] (y_dim=5), x_dim=2
# --------------------------------------------------------------------------- #
Y_DIMS = np.array([3, 2])
Y_DIM = int(Y_DIMS.sum())
X_DIM = 2
N_GROUPS = len(Y_DIMS)


class TestObsParamsPrior:
    """Tests for ObsParamsPrior.sample()."""

    # Moderate hyperpriors to avoid extreme Gamma samples from defaults (1e-12)
    _hp = ObsParamsHyperPrior(a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0)

    def test_sample_shapes(self):
        """Sampled arrays have correct shapes."""
        prior = ObsParamsPrior(hyperprior=self._hp)
        sample = prior.sample(Y_DIMS, X_DIM, rng=np.random.default_rng(0))
        assert sample.C.shape == (Y_DIM, X_DIM)
        assert sample.d.shape == (Y_DIM,)
        assert sample.phi.shape == (Y_DIM,)
        assert sample.alpha.shape == (N_GROUPS, X_DIM)
        assert sample.x_dim == X_DIM
        np.testing.assert_array_equal(sample.y_dims, Y_DIMS)

    def test_sample_positivity(self):
        """Phi and finite alpha values are positive (Gamma-distributed)."""
        prior = ObsParamsPrior(hyperprior=self._hp)
        sample = prior.sample(Y_DIMS, X_DIM, rng=np.random.default_rng(0))
        assert np.all(sample.phi > 0)
        finite_alpha = sample.alpha[np.isfinite(sample.alpha)]
        assert np.all(finite_alpha > 0)

    def test_sample_reproducibility(self):
        """Same seed produces identical samples."""
        prior = ObsParamsPrior(hyperprior=self._hp)
        s1 = prior.sample(Y_DIMS, X_DIM, rng=np.random.default_rng(42))
        s2 = prior.sample(Y_DIMS, X_DIM, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1.C, s2.C)
        np.testing.assert_array_equal(s1.d, s2.d)
        np.testing.assert_array_equal(s1.phi, s2.phi)
        np.testing.assert_array_equal(s1.alpha, s2.alpha)

    def test_sample_structured_sparsity(self):
        """np.inf in a_alpha produces zero loadings and inf alpha."""
        # Group 0 uses both latents; group 1 only uses latent 0
        a_alpha = np.array([[1.0, 1.0], [1.0, np.inf]])
        b_alpha = np.ones((N_GROUPS, X_DIM))
        hp = ObsParamsHyperPriorStructured(a_alpha=a_alpha, b_alpha=b_alpha)
        prior = ObsParamsPrior(hyperprior=hp)
        sample = prior.sample(Y_DIMS, X_DIM, rng=np.random.default_rng(0))

        # Group 1 (rows 3:5), latent 1 should be zeroed out
        C_group1 = sample.C[Y_DIMS[0] :, :]
        np.testing.assert_array_equal(C_group1[:, 1], 0.0)
        assert sample.alpha[1, 1] == np.inf

        # Group 0, latent 1 should have non-zero loadings (with high probability)
        C_group0 = sample.C[: Y_DIMS[0], :]
        assert not np.all(C_group0[:, 1] == 0.0)
