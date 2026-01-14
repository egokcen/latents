"""Pytest configuration and shared test utilities."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import latents.gfa.simulation as gfa_sim
from latents._internal.numerics import stability_floor
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.observation import ObsParamsHyperPriorStructured


def testing_tols(dtype) -> dict:
    """Return dtype-aware tolerances for np.testing.assert_allclose.

    Parameters
    ----------
    dtype
        NumPy dtype to determine precision. Defaults to float64.

    Returns
    -------
    dict
        Dictionary with 'rtol' and 'atol' keys suitable for unpacking
        into np.testing.assert_allclose.

    Examples
    --------
    >>> tols = testing_tols(np.float64)
    >>> np.testing.assert_allclose(actual, expected, **tols)
    """
    eps = np.finfo(dtype).eps
    floor = stability_floor(dtype)
    return {
        "rtol": float(np.sqrt(eps)),  # ~1.5e-8 (f64), ~3.5e-4 (f32)
        "atol": floor,  # 1e-10 (f64), 1e-6 (f32)
    }


# -----------------------------------------------------------------------------
# GFA Fixtures
# -----------------------------------------------------------------------------


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


@pytest.fixture
def fit_context(fitted_model_converged):
    """Real GFAFitContext from a fitted model for callback tests."""
    from latents.gfa.tracking import GFAFitContext

    model = fitted_model_converged
    return GFAFitContext(
        config=model.config,
        obs_hyperprior=model.obs_hyperprior,
        obs_posterior=model.obs_posterior,
        latents_posterior=model.latents_posterior,
        tracker=model.tracker,
        flags=model.flags,
    )
