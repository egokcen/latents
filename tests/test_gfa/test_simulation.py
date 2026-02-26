"""Tests for GFA simulation and persistence."""

from __future__ import annotations

import numpy as np
import pytest

import latents.gfa.simulation as gfa_sim
from latents.gfa.config import GFASimConfig
from latents.observation import ObsParamsHyperPrior, ObsParamsHyperPriorStructured


# --- Fixtures ---


@pytest.fixture
def sim_config():
    """Create simulation config with seed for reproducibility."""
    return GFASimConfig(
        n_samples=50,
        y_dims=np.array([8, 8]),
        x_dim=3,
        snr=1.0,
        random_seed=42,
    )


@pytest.fixture
def sim_config_no_seed():
    """Create simulation config without seed."""
    return GFASimConfig(
        n_samples=50,
        y_dims=np.array([8, 8]),
        x_dim=3,
        snr=1.0,
        random_seed=None,
    )


@pytest.fixture
def hyperprior_simple():
    """Create simple homogeneous hyperprior."""
    return ObsParamsHyperPrior(
        a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
    )


@pytest.fixture
def hyperprior_structured():
    """Structured hyperprior with sparsity pattern."""
    sparsity_pattern = np.array(
        [
            [1, 1, np.inf],
            [1, np.inf, 1],
        ],
    )
    return ObsParamsHyperPriorStructured(
        a_alpha=100 * sparsity_pattern,
        b_alpha=100 * np.ones_like(sparsity_pattern),
        a_phi=1.0,
        b_phi=1.0,
        beta_d=1.0,
    )


# --- Tests ---


class TestSimulate:
    """Tests for simulate() function."""

    def test_basic_simulation(self, sim_config, hyperprior_simple):
        """Test that simulate returns correct result structure."""
        result = gfa_sim.simulate(sim_config, hyperprior_simple)

        # Check result type
        assert isinstance(result, gfa_sim.GFASimulationResult)

        # Check config and hyperprior are stored
        assert result.config is sim_config
        assert result.hyperprior is hyperprior_simple

        # Check output shapes
        assert result.observations.data.shape == (16, 50)  # y_dim=16, n_samples=50
        assert result.latents.data.shape == (3, 50)  # x_dim=3, n_samples=50
        assert result.obs_params.C.shape == (16, 3)  # y_dim=16, x_dim=3

    def test_reproducibility(self, sim_config, hyperprior_simple):
        """Test that same seed produces identical results."""
        result1 = gfa_sim.simulate(sim_config, hyperprior_simple)
        result2 = gfa_sim.simulate(sim_config, hyperprior_simple)

        np.testing.assert_array_equal(
            result1.observations.data, result2.observations.data
        )
        np.testing.assert_array_equal(result1.latents.data, result2.latents.data)
        np.testing.assert_array_equal(result1.obs_params.C, result2.obs_params.C)

    def test_structured_hyperprior(self, hyperprior_structured):
        """Test simulation with structured hyperprior enforces sparsity."""
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            snr=1.0,
            random_seed=42,
        )
        result = gfa_sim.simulate(config, hyperprior_structured)

        # Check sparsity is enforced (columns with np.inf in a_alpha should be zero)
        # Group 0: column 2 should be zero, Group 1: column 1 should be zero
        C_groups = np.split(result.obs_params.C, [8], axis=0)
        np.testing.assert_array_equal(C_groups[0][:, 2], 0.0)
        np.testing.assert_array_equal(C_groups[1][:, 1], 0.0)

    def test_scalar_snr(self, hyperprior_simple):
        """Test that scalar SNR is broadcast correctly."""
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8, 8]),
            x_dim=3,
            snr=2.0,  # scalar
            random_seed=42,
        )
        result = gfa_sim.simulate(config, hyperprior_simple)

        # Should complete without error and have correct shape
        assert result.observations.data.shape == (24, 50)


class TestSaveLoadSimulation:
    """Tests for save_simulation() and load_simulation()."""

    def test_snapshot_roundtrip_simple_hyperprior(
        self, sim_config, hyperprior_simple, tmp_path
    ):
        """Test full snapshot round-trip with simple hyperprior."""
        result = gfa_sim.simulate(sim_config, hyperprior_simple)
        path = tmp_path / "snapshot.safetensors"

        gfa_sim.save_simulation(path, result)
        loaded = gfa_sim.load_simulation(path)

        # Config preserved
        assert loaded.config.n_samples == sim_config.n_samples
        assert loaded.config.x_dim == sim_config.x_dim
        assert loaded.config.random_seed == sim_config.random_seed
        np.testing.assert_array_equal(loaded.config.y_dims, sim_config.y_dims)

        # Hyperprior preserved
        assert isinstance(loaded.hyperprior, ObsParamsHyperPrior)
        assert loaded.hyperprior.a_alpha == hyperprior_simple.a_alpha
        assert loaded.hyperprior.b_alpha == hyperprior_simple.b_alpha

        # Arrays preserved exactly (bit-exact)
        np.testing.assert_array_equal(
            loaded.observations.data, result.observations.data
        )
        np.testing.assert_array_equal(loaded.latents.data, result.latents.data)
        np.testing.assert_array_equal(loaded.obs_params.C, result.obs_params.C)
        np.testing.assert_array_equal(loaded.obs_params.d, result.obs_params.d)
        np.testing.assert_array_equal(loaded.obs_params.phi, result.obs_params.phi)
        np.testing.assert_array_equal(loaded.obs_params.alpha, result.obs_params.alpha)

    def test_snapshot_roundtrip_structured_hyperprior(
        self, hyperprior_structured, tmp_path
    ):
        """Test full snapshot round-trip with structured hyperprior."""
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            snr=1.0,
            random_seed=42,
        )
        result = gfa_sim.simulate(config, hyperprior_structured)
        path = tmp_path / "snapshot_structured.safetensors"

        gfa_sim.save_simulation(path, result)
        loaded = gfa_sim.load_simulation(path)

        # Hyperprior type and values preserved
        assert isinstance(loaded.hyperprior, ObsParamsHyperPriorStructured)
        np.testing.assert_array_equal(
            loaded.hyperprior.a_alpha, hyperprior_structured.a_alpha
        )
        np.testing.assert_array_equal(
            loaded.hyperprior.b_alpha, hyperprior_structured.b_alpha
        )
        assert loaded.hyperprior.a_phi == hyperprior_structured.a_phi

        # Arrays preserved
        np.testing.assert_array_equal(
            loaded.observations.data, result.observations.data
        )

    def test_snapshot_with_array_snr(self, hyperprior_simple, tmp_path):
        """Test snapshot round-trip with array SNR in config."""
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            snr=np.array([1.0, 2.0]),  # array SNR
            random_seed=42,
        )
        result = gfa_sim.simulate(config, hyperprior_simple)
        path = tmp_path / "snapshot_array_snr.safetensors"

        gfa_sim.save_simulation(path, result)
        loaded = gfa_sim.load_simulation(path)

        # SNR array preserved
        np.testing.assert_array_equal(loaded.config.snr, config.snr)


class TestSaveLoadRecipe:
    """Tests for save_simulation_recipe() and load_simulation_recipe()."""

    def test_recipe_roundtrip_simple_hyperprior(
        self, sim_config, hyperprior_simple, tmp_path
    ):
        """Test recipe round-trip with simple hyperprior."""
        path = tmp_path / "recipe.safetensors"

        gfa_sim.save_simulation_recipe(path, sim_config, hyperprior_simple)
        loaded_config, loaded_hyperprior = gfa_sim.load_simulation_recipe(path)

        # Config preserved
        assert loaded_config.n_samples == sim_config.n_samples
        assert loaded_config.x_dim == sim_config.x_dim
        assert loaded_config.random_seed == sim_config.random_seed
        np.testing.assert_array_equal(loaded_config.y_dims, sim_config.y_dims)

        # Hyperprior preserved
        assert isinstance(loaded_hyperprior, ObsParamsHyperPrior)
        assert loaded_hyperprior.a_alpha == hyperprior_simple.a_alpha

    def test_recipe_roundtrip_structured_hyperprior(
        self, hyperprior_structured, tmp_path
    ):
        """Test recipe round-trip with structured hyperprior."""
        # Need config with matching x_dim for structured hyperprior
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            snr=1.0,
            random_seed=42,
        )
        path = tmp_path / "recipe_structured.safetensors"

        gfa_sim.save_simulation_recipe(path, config, hyperprior_structured)
        _, loaded_hyperprior = gfa_sim.load_simulation_recipe(path)

        # Hyperprior type and values preserved
        assert isinstance(loaded_hyperprior, ObsParamsHyperPriorStructured)
        np.testing.assert_array_equal(
            loaded_hyperprior.a_alpha, hyperprior_structured.a_alpha
        )
        np.testing.assert_array_equal(
            loaded_hyperprior.b_alpha, hyperprior_structured.b_alpha
        )

    def test_recipe_regeneration(self, sim_config, hyperprior_simple, tmp_path):
        """Test that loading recipe and re-simulating produces identical results."""
        # Original simulation
        original_result = gfa_sim.simulate(sim_config, hyperprior_simple)

        # Save recipe
        path = tmp_path / "recipe.safetensors"
        gfa_sim.save_simulation_recipe(path, sim_config, hyperprior_simple)

        # Load recipe and regenerate
        loaded_config, loaded_hyperprior = gfa_sim.load_simulation_recipe(path)
        regenerated_result = gfa_sim.simulate(loaded_config, loaded_hyperprior)

        # Results should be identical
        np.testing.assert_array_equal(
            regenerated_result.observations.data, original_result.observations.data
        )
        np.testing.assert_array_equal(
            regenerated_result.latents.data, original_result.latents.data
        )
        np.testing.assert_array_equal(
            regenerated_result.obs_params.C, original_result.obs_params.C
        )

    def test_recipe_from_snapshot(self, sim_config, hyperprior_simple, tmp_path):
        """Test that load_simulation_recipe works on snapshot files."""
        result = gfa_sim.simulate(sim_config, hyperprior_simple)
        path = tmp_path / "snapshot.safetensors"

        # Save as full snapshot
        gfa_sim.save_simulation(path, result)

        # Load as recipe (should work)
        loaded_config, loaded_hyperprior = gfa_sim.load_simulation_recipe(path)

        assert loaded_config.n_samples == sim_config.n_samples
        assert isinstance(loaded_hyperprior, ObsParamsHyperPrior)

    def test_recipe_requires_seed(
        self, sim_config_no_seed, hyperprior_simple, tmp_path
    ):
        """Test that save_simulation_recipe requires random_seed."""
        path = tmp_path / "recipe.safetensors"

        with pytest.raises(ValueError, match="random_seed required"):
            gfa_sim.save_simulation_recipe(path, sim_config_no_seed, hyperprior_simple)

    def test_recipe_sequence_seed_roundtrip(self, hyperprior_simple, tmp_path):
        """Test save/load recipe preserves sequence random_seed."""
        config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            random_seed=[1, 2, 42],
        )

        path = tmp_path / "recipe.safetensors"
        gfa_sim.save_simulation_recipe(path, config, hyperprior_simple)

        loaded_config, _ = gfa_sim.load_simulation_recipe(path)
        assert loaded_config.random_seed == [1, 2, 42]

        # Verify reproducibility after round-trip
        result1 = gfa_sim.simulate(config, hyperprior_simple)
        result2 = gfa_sim.simulate(loaded_config, hyperprior_simple)
        np.testing.assert_array_equal(
            result1.observations.data, result2.observations.data
        )

    def test_load_simulation_rejects_recipe(
        self, sim_config, hyperprior_simple, tmp_path
    ):
        """Test that load_simulation raises on recipe-only files."""
        path = tmp_path / "recipe.safetensors"
        gfa_sim.save_simulation_recipe(path, sim_config, hyperprior_simple)

        with pytest.raises(ValueError, match="recipe"):
            gfa_sim.load_simulation(path)
