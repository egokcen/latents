"""Tests for pPCA simulation and persistence."""

from __future__ import annotations

import numpy as np
import pytest

import latents.ppca.simulation as ppca_sim
from latents.observation import ObsParamsPoint
from latents.ppca.config import pPCASimConfig

Y_DIM = 20
X_DIM = 3
N_SAMPLES = 50


# --- Fixtures ---


@pytest.fixture
def sim_config():
    """Create simulation config with seed for reproducibility."""
    return pPCASimConfig(
        n_samples=N_SAMPLES,
        y_dim=Y_DIM,
        x_dim=X_DIM,
        snr=1.0,
        random_seed=42,
    )


@pytest.fixture
def sim_config_no_seed():
    """Create simulation config without seed."""
    return pPCASimConfig(
        n_samples=N_SAMPLES,
        y_dim=Y_DIM,
        x_dim=X_DIM,
        snr=1.0,
        random_seed=None,
    )


@pytest.fixture
def obs_params():
    """Create observation parameters matching sim_config dimensions."""
    rng = np.random.default_rng(0)
    return ObsParamsPoint(
        C=rng.standard_normal((Y_DIM, X_DIM)),
        d=rng.standard_normal(Y_DIM),
        phi=np.ones(Y_DIM),
        y_dims=np.array([Y_DIM]),
        x_dim=X_DIM,
    )


# --- Tests ---


class TestSimulate:
    """Tests for simulate() function."""

    def test_basic_simulation(self, sim_config, obs_params):
        """Test that simulate returns correct result structure."""
        result = ppca_sim.simulate(sim_config, obs_params)

        # Check result type
        assert isinstance(result, ppca_sim.pPCASimulationResult)

        # Config stored; obs_params is an SNR-adjusted copy with C, d unchanged
        assert result.config is sim_config
        np.testing.assert_array_equal(result.obs_params.C, obs_params.C)
        np.testing.assert_array_equal(result.obs_params.d, obs_params.d)

        # Check output shapes
        assert result.observations.data.shape == (Y_DIM, N_SAMPLES)
        assert result.latents.data.shape == (X_DIM, N_SAMPLES)
        assert result.obs_params.C.shape == (Y_DIM, X_DIM)
        np.testing.assert_array_equal(result.observations.dims, [Y_DIM])

    def test_reproducibility(self, sim_config, obs_params):
        """Test that same seed produces identical results."""
        result1 = ppca_sim.simulate(sim_config, obs_params)
        result2 = ppca_sim.simulate(sim_config, obs_params)

        np.testing.assert_array_equal(
            result1.observations.data, result2.observations.data
        )
        np.testing.assert_array_equal(result1.latents.data, result2.latents.data)
        np.testing.assert_array_equal(result1.obs_params.phi, result2.obs_params.phi)

    def test_different_seeds_differ(self, obs_params):
        """Test that different seeds produce different samples."""
        config1 = pPCASimConfig(
            n_samples=N_SAMPLES, y_dim=Y_DIM, x_dim=X_DIM, snr=1.0, random_seed=1
        )
        config2 = pPCASimConfig(
            n_samples=N_SAMPLES, y_dim=Y_DIM, x_dim=X_DIM, snr=1.0, random_seed=2
        )
        result1 = ppca_sim.simulate(config1, obs_params)
        result2 = ppca_sim.simulate(config2, obs_params)

        assert not np.array_equal(result1.observations.data, result2.observations.data)

    @pytest.mark.parametrize("snr", [0.5, 2.0])
    def test_snr_applied(self, obs_params, snr):
        """Test that phi is rescaled so sum(C**2) / sum(1/phi) equals the target."""
        phi_input = obs_params.phi.copy()
        config = pPCASimConfig(
            n_samples=N_SAMPLES, y_dim=Y_DIM, x_dim=X_DIM, snr=snr, random_seed=42
        )
        result = ppca_sim.simulate(config, obs_params)
        adjusted = result.obs_params

        achieved_snr = np.sum(adjusted.C**2) / np.sum(1 / adjusted.phi)
        np.testing.assert_allclose(achieved_snr, snr, rtol=1e-12)

        # Isotropic phi stays isotropic
        np.testing.assert_allclose(adjusted.phi, adjusted.phi[0], rtol=1e-12)

        # Input is not mutated
        np.testing.assert_array_equal(obs_params.phi, phi_input)

    def test_sample_moments(self, obs_params):
        """Test that sample moments match the pPCA marginal N(d, CC^T + diag(1/phi)).

        Uses a large fixed-seed sample. Tolerances are roughly 3x the observed
        estimation error at this sample size.
        """
        config = pPCASimConfig(
            n_samples=20_000, y_dim=Y_DIM, x_dim=X_DIM, snr=2.0, random_seed=1
        )
        result = ppca_sim.simulate(config, obs_params)
        Y = result.observations.data  # (y_dim, n_samples)
        params = result.obs_params

        expected_cov = params.C @ params.C.T + np.diag(1 / params.phi)

        np.testing.assert_allclose(Y.mean(axis=1), params.d, atol=0.1)
        np.testing.assert_allclose(np.cov(Y), expected_cov, atol=0.2)


class TestSaveLoadSimulation:
    """Tests for save_simulation() and load_simulation()."""

    def test_snapshot_roundtrip(self, sim_config, obs_params, tmp_path):
        """Test full snapshot round-trip preserves all state."""
        result = ppca_sim.simulate(sim_config, obs_params)
        path = tmp_path / "snapshot.safetensors"

        ppca_sim.save_simulation(path, result)
        loaded = ppca_sim.load_simulation(path)

        # Config preserved
        assert loaded.config == sim_config

        # Arrays preserved exactly (bit-exact)
        np.testing.assert_array_equal(
            loaded.observations.data, result.observations.data
        )
        np.testing.assert_array_equal(
            loaded.observations.dims, result.observations.dims
        )
        np.testing.assert_array_equal(loaded.latents.data, result.latents.data)
        np.testing.assert_array_equal(loaded.obs_params.C, result.obs_params.C)
        np.testing.assert_array_equal(loaded.obs_params.d, result.obs_params.d)
        np.testing.assert_array_equal(loaded.obs_params.phi, result.obs_params.phi)
        np.testing.assert_array_equal(
            loaded.obs_params.y_dims, result.obs_params.y_dims
        )
        assert loaded.obs_params.x_dim == result.obs_params.x_dim

    def test_snapshot_roundtrip_noncontiguous_arrays(
        self, sim_config, obs_params, tmp_path
    ):
        """Round-trip is exact for F-contiguous simulation arrays.

        Regression test for silent corruption under safetensors >=0.8.0,
        which serializes from the raw buffer pointer and ignores strides.
        """
        result = ppca_sim.simulate(sim_config, obs_params)
        result.obs_params.C = np.asfortranarray(result.obs_params.C)
        result.observations.data = np.asfortranarray(result.observations.data)
        assert not result.obs_params.C.flags.c_contiguous
        assert not result.observations.data.flags.c_contiguous

        path = tmp_path / "snapshot_noncontiguous.safetensors"
        ppca_sim.save_simulation(path, result)
        loaded = ppca_sim.load_simulation(path)

        np.testing.assert_array_equal(loaded.obs_params.C, result.obs_params.C)
        np.testing.assert_array_equal(
            loaded.observations.data, result.observations.data
        )

    def test_load_simulation_rejects_recipe(self, sim_config, tmp_path):
        """Test that load_simulation raises on recipe-only files."""
        path = tmp_path / "recipe.safetensors"
        ppca_sim.save_simulation_recipe(path, sim_config)

        with pytest.raises(ValueError, match="recipe"):
            ppca_sim.load_simulation(path)


class TestSaveLoadRecipe:
    """Tests for save_simulation_recipe() and load_simulation_recipe()."""

    def test_recipe_roundtrip(self, sim_config, tmp_path):
        """Test recipe round-trip preserves config."""
        path = tmp_path / "recipe.safetensors"
        ppca_sim.save_simulation_recipe(path, sim_config)

        assert ppca_sim.load_simulation_recipe(path) == sim_config

    def test_recipe_regeneration(self, sim_config, obs_params, tmp_path):
        """Test that recipe plus the same obs_params regenerates identical results.

        The recipe stores only the config, so the caller must supply the
        original obs_params.
        """
        original_result = ppca_sim.simulate(sim_config, obs_params)

        path = tmp_path / "recipe.safetensors"
        ppca_sim.save_simulation_recipe(path, sim_config)
        loaded_config = ppca_sim.load_simulation_recipe(path)
        regenerated_result = ppca_sim.simulate(loaded_config, obs_params)

        np.testing.assert_array_equal(
            regenerated_result.observations.data, original_result.observations.data
        )
        np.testing.assert_array_equal(
            regenerated_result.latents.data, original_result.latents.data
        )

    def test_recipe_from_snapshot(self, sim_config, obs_params, tmp_path):
        """Test that load_simulation_recipe works on snapshot files."""
        result = ppca_sim.simulate(sim_config, obs_params)
        path = tmp_path / "snapshot.safetensors"
        ppca_sim.save_simulation(path, result)

        assert ppca_sim.load_simulation_recipe(path) == sim_config

    def test_recipe_requires_seed(self, sim_config_no_seed, tmp_path):
        """Test that save_simulation_recipe requires random_seed."""
        path = tmp_path / "recipe.safetensors"

        with pytest.raises(ValueError, match="random_seed required"):
            ppca_sim.save_simulation_recipe(path, sim_config_no_seed)

    def test_recipe_sequence_seed_roundtrip(self, obs_params, tmp_path):
        """Test save/load recipe preserves sequence random_seed."""
        config = pPCASimConfig(
            n_samples=N_SAMPLES,
            y_dim=Y_DIM,
            x_dim=X_DIM,
            snr=1.0,
            random_seed=[1, 2, 42],
        )

        path = tmp_path / "recipe.safetensors"
        ppca_sim.save_simulation_recipe(path, config)

        loaded_config = ppca_sim.load_simulation_recipe(path)
        assert loaded_config.random_seed == [1, 2, 42]

        # Verify reproducibility after round-trip
        result1 = ppca_sim.simulate(config, obs_params)
        result2 = ppca_sim.simulate(loaded_config, obs_params)
        np.testing.assert_array_equal(
            result1.observations.data, result2.observations.data
        )
