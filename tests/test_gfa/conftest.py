"""GFA-specific test fixtures."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import latents.gfa.simulation as gfa_sim
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.observation import ObsParamsHyperPriorStructured


@pytest.fixture(scope="module")
def simulation_result():
    """Generate simulated GFA data with known ground truth.

    Uses fixed seed for reproducibility. Returns GFASimulationResult
    for regression testing.
    """
    # Sparsity pattern: rows=groups, cols=latents. np.inf means latent absent.
    sparsity_pattern = np.array(
        [
            [1, 1, 1, np.inf, 1, np.inf, np.inf],
            [1, 1, np.inf, 1, np.inf, 1, np.inf],
            [1, np.inf, 1, 1, np.inf, np.inf, 1],
        ],
    )
    MAG = 100  # Controls variance of alpha (larger = less variance)
    hyperprior = ObsParamsHyperPriorStructured(
        a_alpha=MAG * sparsity_pattern,
        b_alpha=MAG * np.ones_like(sparsity_pattern),
        a_phi=1.0,
        b_phi=1.0,
        beta_d=1.0,
    )

    config = GFASimConfig(
        n_samples=100,
        y_dims=np.array([10, 10, 10]),
        x_dim=7,
        snr=1.0,
        random_seed=0,
    )

    return gfa_sim.simulate(config, hyperprior)


@pytest.fixture(scope="module")
def fitted_model_converged(simulation_result):
    """Fit a GFA model to convergence.

    Uses tight tolerance (fit_tol=1e-8) for numerical accuracy tests.
    Fixed fitting seed for reproducibility.
    """
    config = GFAFitConfig(
        x_dim_init=10,
        fit_tol=1e-8,
        max_iter=20000,
        random_seed=0,
        min_var_frac=0.001,
        prune_x=True,
        prune_tol=1e-7,
        save_x=True,
        save_c_cov=True,
        save_fit_progress=True,
    )

    model = GFAModel(config=config)
    model.fit(simulation_result.observations)

    return model


@pytest.fixture
def fitted_model_copy(fitted_model_converged):
    """Fresh deep copy of converged model for tests that mutate state.

    Function-scoped (default), so each test gets its own isolated copy.
    The original fitted_model_converged is never modified.
    """
    return copy.deepcopy(fitted_model_converged)
