"""Test the GFAModel class."""

from __future__ import annotations

import numpy as np
import pytest

import latents.gfa.simulation as gfa_sim
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.observation import ObsParamsHyperPrior


# --- Fixtures ---


@pytest.fixture
def unfitted_model():
    """Create an unfitted GFAModel with non-default config."""
    config = GFAFitConfig(
        x_dim_init=15,
        fit_tol=1e-6,
        max_iter=500,
    )
    obs_hyperprior = ObsParamsHyperPrior(
        a_alpha=1e-10,
        b_alpha=1e-10,
        a_phi=1e-8,
        b_phi=1e-8,
        beta_d=1e-6,
    )
    return GFAModel(config=config, obs_hyperprior=obs_hyperprior)


@pytest.fixture
def fitted_model():
    """Fit a small GFA model for testing save/load."""
    hyperprior = ObsParamsHyperPrior(
        a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
    )
    sim_config = GFASimConfig(
        n_samples=50,
        y_dims=np.array([8, 8]),
        x_dim=3,
        snr=1.0,
        random_seed=42,
    )
    sim_result = gfa_sim.simulate(sim_config, hyperprior)

    fit_config = GFAFitConfig(
        x_dim_init=5,
        fit_tol=1e-4,
        max_iter=100,
        random_seed=0,
        save_x=True,
        save_c_cov=True,
        save_fit_progress=True,
    )

    model = GFAModel(config=fit_config)
    model.fit(sim_result.observations)
    return model


# --- Tests ---


@pytest.mark.fit
class TestSaveLoad:
    """Tests for GFAModel.save() and GFAModel.load().

    Uses tmp_path fixture (built-in pytest fixture) for temporary files.
    Uses assert_array_equal for exact equality since safetensors produces
    bit-exact round-trips (no floating-point computation).
    """

    def test_save_load_unfitted(self, unfitted_model, tmp_path):
        """Test round-trip for unfitted model preserves config and hyperprior."""
        path = tmp_path / "unfitted.safetensors"

        unfitted_model.save(path)
        loaded = GFAModel.load(path)

        # Config preserved
        assert loaded.config.x_dim_init == unfitted_model.config.x_dim_init
        assert loaded.config.fit_tol == unfitted_model.config.fit_tol
        assert loaded.config.max_iter == unfitted_model.config.max_iter
        assert loaded.config.min_var_frac == unfitted_model.config.min_var_frac

        # Hyperprior preserved
        assert loaded.obs_hyperprior.a_alpha == unfitted_model.obs_hyperprior.a_alpha
        assert loaded.obs_hyperprior.b_alpha == unfitted_model.obs_hyperprior.b_alpha
        assert loaded.obs_hyperprior.a_phi == unfitted_model.obs_hyperprior.a_phi
        assert loaded.obs_hyperprior.b_phi == unfitted_model.obs_hyperprior.b_phi
        assert loaded.obs_hyperprior.beta_d == unfitted_model.obs_hyperprior.beta_d

        # Posteriors not present
        assert loaded.obs_posterior is None
        assert loaded.latents_posterior is None
        assert loaded.tracker is None
        assert loaded.flags is None

    def test_save_load_fitted(self, fitted_model, tmp_path):
        """Test round-trip for fitted model preserves all state."""
        path = tmp_path / "fitted.safetensors"

        fitted_model.save(path)
        loaded = GFAModel.load(path)

        # Config preserved
        assert loaded.config == fitted_model.config

        # Obs posterior preserved
        assert loaded.obs_posterior is not None
        assert loaded.obs_posterior.x_dim == fitted_model.obs_posterior.x_dim
        np.testing.assert_array_equal(
            loaded.obs_posterior.y_dims, fitted_model.obs_posterior.y_dims
        )
        np.testing.assert_array_equal(
            loaded.obs_posterior.C.mean, fitted_model.obs_posterior.C.mean
        )
        np.testing.assert_array_equal(
            loaded.obs_posterior.C.cov, fitted_model.obs_posterior.C.cov
        )
        np.testing.assert_array_equal(
            loaded.obs_posterior.alpha.mean, fitted_model.obs_posterior.alpha.mean
        )
        np.testing.assert_array_equal(
            loaded.obs_posterior.d.mean, fitted_model.obs_posterior.d.mean
        )
        np.testing.assert_array_equal(
            loaded.obs_posterior.phi.mean, fitted_model.obs_posterior.phi.mean
        )
        assert loaded.obs_posterior.phi.a == fitted_model.obs_posterior.phi.a

        # Latents posterior preserved
        assert loaded.latents_posterior is not None
        np.testing.assert_array_equal(
            loaded.latents_posterior.mean, fitted_model.latents_posterior.mean
        )
        np.testing.assert_array_equal(
            loaded.latents_posterior.cov, fitted_model.latents_posterior.cov
        )

        # Tracker preserved
        assert loaded.tracker is not None
        np.testing.assert_array_equal(loaded.tracker.lb, fitted_model.tracker.lb)
        np.testing.assert_array_equal(
            loaded.tracker.iter_time, fitted_model.tracker.iter_time
        )
        assert loaded.tracker.lb_base == fitted_model.tracker.lb_base

        # Flags preserved
        assert loaded.flags is not None
        assert loaded.flags.converged == fitted_model.flags.converged
        assert loaded.flags.decreasing_lb == fitted_model.flags.decreasing_lb
        assert loaded.flags.private_var_floor == fitted_model.flags.private_var_floor
        assert loaded.flags.x_dims_removed == fitted_model.flags.x_dims_removed

    def test_loaded_model_can_infer(self, fitted_model, tmp_path):
        """Test that loaded model can perform inference on new data."""
        path = tmp_path / "fitted.safetensors"

        fitted_model.save(path)
        loaded = GFAModel.load(path)

        # Generate new data with same structure
        y_dims = fitted_model.obs_posterior.y_dims
        x_dim = fitted_model.obs_posterior.x_dim
        n_samples_new = 20

        hyperprior = ObsParamsHyperPrior(
            a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
        )
        sim_config = GFASimConfig(
            n_samples=n_samples_new,
            y_dims=y_dims,
            x_dim=x_dim,
            snr=1.0,
            random_seed=123,
        )
        sim_result = gfa_sim.simulate(sim_config, hyperprior)

        # Infer latents with loaded model
        latents_new = loaded.infer_latents(sim_result.observations)

        # Verify shape
        assert latents_new.mean.shape == (x_dim, n_samples_new)

    def test_loaded_model_can_resume_fit(self, tmp_path):
        """Test that loaded model can resume fitting."""
        hyperprior = ObsParamsHyperPrior(
            a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
        )
        sim_config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            snr=1.0,
            random_seed=42,
        )
        sim_result = gfa_sim.simulate(sim_config, hyperprior)

        # Fit for a few iterations (won't converge)
        fit_config = GFAFitConfig(
            x_dim_init=5,
            fit_tol=1e-10,  # Very tight tolerance
            max_iter=10,  # Few iterations
            random_seed=0,
            save_x=True,  # Required for resume_fit
            save_fit_progress=True,
        )
        model = GFAModel(config=fit_config)
        model.fit(sim_result.observations)

        # Should not have converged
        assert not model.flags.converged
        n_iter_before = len(model.tracker.lb)

        # Save and reload
        path = tmp_path / "partial.safetensors"
        model.save(path)
        loaded = GFAModel.load(path)

        # Resume fitting
        loaded.resume_fit(sim_result.observations, max_iter=10)

        # Should have more iterations now
        assert len(loaded.tracker.lb) > n_iter_before


@pytest.mark.fit
class TestRecompute:
    """Tests for GFAModel.recompute_latents() and recompute_loadings().

    Uses fitted_model_copy (function-scoped deep copy) for test isolation.
    Each test gets its own copy of the converged model to mutate freely.
    """

    def test_recompute_latents(self, simulation_result, fitted_model_copy):
        """Test latent reconstruction.

        Since X is the final update during fitting, reconstruction from saved parameters
        should be exact.
        """
        from tests.conftest import testing_tols

        model = fitted_model_copy
        Y = simulation_result.observations

        # Store original latents
        X_original = model.latents_posterior.mean.copy()
        X_cov_original = model.latents_posterior.cov.copy()

        # Clear and recompute
        model.latents_posterior.clear()
        model.recompute_latents(Y)

        # Should be exact within numerical tolerance
        tols = testing_tols(X_original.dtype)
        np.testing.assert_allclose(model.latents_posterior.mean, X_original, **tols)
        np.testing.assert_allclose(model.latents_posterior.cov, X_cov_original, **tols)

    def test_recompute_latents_error_if_not_fitted(self, simulation_result):
        """Test that recompute_latents raises if model not fitted."""
        model = GFAModel()

        with pytest.raises(ValueError, match="must be fitted"):
            model.recompute_latents(simulation_result.observations)

    def test_recompute_loadings(self, simulation_result, fitted_model_copy):
        """Test loadings reconstruction.

        At convergence, the recomputed C should be close to original. Not exact
        because reconstruction uses final X rather than X from previous iteration.
        """
        model = fitted_model_copy
        Y = simulation_result.observations

        # Store original loadings
        C_mean_original = model.obs_posterior.C.mean.copy()
        C_cov_original = model.obs_posterior.C.cov.copy()

        # Clear C.cov and recompute
        model.obs_posterior.C.cov = None
        model.recompute_loadings(Y)

        # Looser tolerance than X - reconstruction uses X_N instead of X_{N-1}
        np.testing.assert_allclose(
            model.obs_posterior.C.mean, C_mean_original, rtol=1e-3, atol=1e-10
        )
        np.testing.assert_allclose(
            model.obs_posterior.C.cov, C_cov_original, rtol=1e-3, atol=1e-10
        )

    def test_recompute_loadings_error_if_not_fitted(self, simulation_result):
        """Test that recompute_loadings raises if model not fitted."""
        model = GFAModel()

        with pytest.raises(ValueError, match="must be fitted"):
            model.recompute_loadings(simulation_result.observations)

    def test_recompute_loadings_error_if_no_latents(
        self, simulation_result, fitted_model_copy
    ):
        """Test that recompute_loadings raises if latents not available."""
        model = fitted_model_copy

        # Clear latents
        model.latents_posterior.clear()

        with pytest.raises(ValueError, match="Latents must be available"):
            model.recompute_loadings(simulation_result.observations)

    def test_recompute_chaining(self, simulation_result, fitted_model_copy):
        """Test that recompute methods support method chaining."""
        from tests.conftest import testing_tols

        model = fitted_model_copy
        Y = simulation_result.observations

        # Store originals
        X_original = model.latents_posterior.mean.copy()
        C_cov_original = model.obs_posterior.C.cov.copy()

        # Clear both
        model.latents_posterior.clear()
        model.obs_posterior.C.cov = None

        # Chain recompute calls
        result = model.recompute_latents(Y).recompute_loadings(Y)

        # Should return self for chaining
        assert result is model

        # Both should be restored (X exact, C approximate)
        tols = testing_tols(X_original.dtype)
        np.testing.assert_allclose(model.latents_posterior.mean, X_original, **tols)
        np.testing.assert_allclose(
            model.obs_posterior.C.cov, C_cov_original, rtol=1e-3, atol=1e-10
        )
