"""Shared fixtures for GFA tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import latents.gfa.simulation as gfa_sim
from latents.gfa import GFAFitConfig, GFAModel
from latents.observation import ObsParamsHyperPriorStructured


@pytest.fixture(scope="module")
def simulation_data():
    """Generate simulated GFA data with known ground truth.

    Uses fixed seeds for reproducibility. Returns data and ground truth
    parameters for regression testing.
    """
    random_seed = 0
    rng = np.random.default_rng(random_seed)
    n_samples = 100
    y_dims = np.array([10, 10, 10])
    n_groups = len(y_dims)
    x_dim = 7
    snr = 1.0 * np.ones(n_groups)

    # Sparsity pattern: rows=groups, cols=latents. np.inf means latent absent.
    sparsity_pattern = np.array(
        [
            [1, 1, 1, np.inf, 1, np.inf, np.inf],
            [1, 1, np.inf, 1, np.inf, 1, np.inf],
            [1, np.inf, 1, 1, np.inf, np.inf, 1],
        ],
    )
    MAG = 100  # Controls variance of alpha (larger = less variance)
    sim_priors = ObsParamsHyperPriorStructured(
        a_alpha=MAG * sparsity_pattern,
        b_alpha=MAG * np.ones_like(sparsity_pattern),
        a_phi=1.0,
        b_phi=1.0,
        beta_d=1.0,
    )

    Y, X_true, obs_params_true = gfa_sim.simulate(
        n_samples, y_dims, x_dim, sim_priors, snr, rng=rng
    )

    return {
        "Y": Y,
        "X_true": X_true,
        "obs_params_true": obs_params_true,
        "y_dims": y_dims,
        "x_dim": x_dim,
    }


@pytest.fixture(scope="module")
def fitted_model_converged(simulation_data):
    """Fit a GFA model to convergence.

    Uses tight tolerance (fit_tol=1e-8) for numerical accuracy tests.
    Fixed fitting seed for reproducibility.
    """
    Y = simulation_data["Y"]

    config = GFAFitConfig(
        x_dim_init=10,
        fit_tol=1e-8,
        max_iter=20000,
        verbose=False,
        random_seed=0,
        min_var_frac=0.001,
        prune_x=True,
        prune_tol=1e-7,
        save_x=True,
        save_c_cov=True,
        save_fit_progress=True,
    )

    model = GFAModel(config=config)
    model.fit(Y)

    return {"model": model}


@pytest.fixture
def fitted_model_copy(fitted_model_converged):
    """Fresh deep copy of converged model for tests that mutate state.

    Function-scoped (default), so each test gets its own isolated copy.
    The original fitted_model_converged is never modified.
    """
    return {"model": copy.deepcopy(fitted_model_converged["model"])}
