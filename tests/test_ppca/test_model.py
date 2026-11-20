"""Test the pPCAModel class."""

from __future__ import annotations

import numpy as np
import pytest

from latents.observation import ObsParamsPoint
import latents.ppca.simulation as ppca_sim
from latents.ppca import pPCAFitConfig, pPCAModel
from latents.ppca.config import pPCASimConfig


# --- Fixtures ---


@pytest.fixture
def unfitted_model():
    """Create an unfitted pPCAModel with non-default config."""
    config = pPCAFitConfig(
        x_dim=15,
        fit_tol=1e-6,
        max_iter=500,
    )

    return pPCAModel(config=config)


@pytest.fixture
def fitted_model():
    """Fit a small pPCA model for testing save/load."""
    sim_config = pPCASimConfig(
        n_samples=50,
        x_dim=3,
        y_dim=20,
        snr=1.0,
        random_seed=42,
    )

    rng = np.random.default_rng(0)
    obs_params = ObsParamsPoint(
        C=rng.standard_normal((sim_config.y_dim, sim_config.x_dim)),
        d=rng.standard_normal(sim_config.y_dim),
        phi=np.ones(sim_config.y_dim),
        y_dims=np.array([sim_config.y_dim]),
        x_dim=sim_config.x_dim,
    )

    sim_result = ppca_sim.simulate(sim_config, obs_params)

    fit_config = pPCAFitConfig(
        x_dim=5,
        fit_tol=1e-4,
        max_iter=100,
        random_seed=0,
        save_x=True,
        save_fit_progress=True,
    )

    model = pPCAModel(config=fit_config)
    model.fit(sim_result.observations)
    return model


@pytest.mark.fit
class TestSaveLoad:
    """Tests for pPCAModel.save() and pPCAModel.load().

    Uses tmp_path fixture (built-in pytest fixture) for temporary files.
    Uses assert_array_equal for exact equality since safetensors produces
    bit-exact round trips (no floating-point computation).
    """

    def test_save_load_unfitted(self, unfitted_model: pPCAModel, tmp_path):
        """Test round-trip for unfitted model preserves config."""
        path = tmp_path / "unfitted.safetensors"

        unfitted_model.save(path)
        loaded = pPCAModel.load(path)

        # Config preserved
        assert loaded.config == unfitted_model.config
        assert loaded.config.fit_tol == unfitted_model.config.fit_tol
        assert loaded.config.max_iter == unfitted_model.config.max_iter
        assert loaded.config.min_var_frac == unfitted_model.config.min_var_frac

        # Posteriors not present
        assert loaded.obs_params is None
        assert loaded.latents_posterior is None
        assert loaded.tracker is None
        assert loaded.flags is None

    def test_save_load_fitted(self, fitted_model, tmp_path):
        """Test round-trip for fitted model preserves all state."""
        path = tmp_path / "fitted.safetensors"

        fitted_model.save(path)
        loaded = pPCAModel.load(path)

        # Config preserved
        assert loaded.config == fitted_model.config

        # Obs posterior preserved
        assert loaded.obs_params is not None
        assert loaded.obs_params.x_dim == fitted_model.obs_params.x_dim
        np.testing.assert_array_equal(loaded.obs_params.C, fitted_model.obs_params.C)
        np.testing.assert_array_equal(
            loaded.obs_params.y_dims, fitted_model.obs_params.y_dims
        )
        np.testing.assert_array_equal(loaded.obs_params.d, fitted_model.obs_params.d)
        np.testing.assert_array_equal(
            loaded.obs_params.phi, fitted_model.obs_params.phi
        )

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

    def test_save_load_noncontiguous_params(self, fitted_model, tmp_path):
        """Round-trip is exact for F-contiguous parameter arrays.

        Regression test for silent corruption under safetensors >=0.8.0,
        which serializes from the raw buffer pointer and ignores strides.
        The layout is forced explicitly here rather than relying on
        numpy's indexing heuristics.
        """
        obs = fitted_model.obs_params
        obs.C = np.asfortranarray(obs.C)
        assert not obs.C.flags.c_contiguous

        path = tmp_path / "noncontiguous.safetensors"
        fitted_model.save(path)
        loaded = pPCAModel.load(path)

        np.testing.assert_array_equal(loaded.obs_params.C, obs.C)
