"""Test the gfa.core module."""

from __future__ import annotations

import numpy as np
import pytest

import latents.gfa.simulation as gfa_sim
from latents.gfa import GFAFitConfig, GFAModel
from latents.observation import ObsParamsHyperPriorStructured


# --- Fixtures ---


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
def fitted_model(simulation_data):
    """Fit a GFA model to simulated data.

    Uses fixed fitting seed for reproducibility. Saves all outputs needed
    for regression tests (X, C covariance, fit progress).
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


# --- Tests ---


def test_fit(fitted_model):
    """Test basic fitting: convergence flags and iteration count."""
    model = fitted_model["model"]

    assert model.flags.converged
    assert not model.flags.decreasing_lb
    assert not model.flags.private_var_floor
    # Regression baselines for fixed seeds (simulation_seed=1, fitting_seed=0).
    # x_dim_init=10, true x_dim=7, so 3 latents pruned.
    assert model.flags.x_dims_removed == 3
    # Iteration count for convergence with fit_tol=1e-8.
    assert len(model.tracker.iter_time) == 2530


def test_elbo_monotonicity(fitted_model):
    """Test that ELBO is monotonically non-decreasing.

    Uses sqrt(machine epsilon) as tolerance to account for floating-point
    accumulation errors while remaining precision-aware.
    """
    from tests.conftest import testing_tols

    model = fitted_model["model"]
    lb = np.array(model.tracker.lb)

    # Use rtol from testing_tols (sqrt(eps)) as absolute tolerance for differences
    tols = testing_tols(lb.dtype)
    tol = tols["rtol"]

    lb_diff = np.diff(lb)
    assert np.all(lb_diff >= -tol), (
        f"ELBO decreased by more than tolerance. "
        f"Min diff: {lb_diff.min():.2e}, tolerance: {-tol:.2e}"
    )


def test_parameter_recovery(simulation_data, fitted_model):
    """Test that fitted parameters recover ground truth loading matrix.

    Compares estimated C matrix against ground truth using per-column
    correlation. Columns are reordered and sign-flipped to align with
    ground truth (ordering is arbitrary in factor models).

    Thresholds calibrated from baseline run with fixed seeds:
    - Min per-column correlation: 0.77 (column 0, group-specific latent)
    - Mean correlation: 0.94
    """
    model = fitted_model["model"]
    obs_params_true = simulation_data["obs_params_true"]

    # Column reordering and sign flips for this seed combination
    # (simulation_seed=1, fitting_seed=0)
    reorder = np.array([1, 6, 4, 0, 3, 5, 2])
    rescale = np.array([-1, -1, 1, 1, 1, -1, 1])

    C_true = obs_params_true.C
    C_est = model.obs_posterior.C.mean[:, reorder] * rescale

    # Compute per-column correlation
    n_cols = C_true.shape[1]
    correlations = np.array(
        [np.corrcoef(C_true[:, j], C_est[:, j])[0, 1] for j in range(n_cols)]
    )

    min_corr = correlations.min()
    mean_corr = correlations.mean()

    # Thresholds with margin below baseline values
    assert min_corr > 0.70, (
        f"Min column correlation {min_corr:.3f} below threshold 0.70"
    )
    assert mean_corr > 0.90, (
        f"Mean column correlation {mean_corr:.3f} below threshold 0.90"
    )
