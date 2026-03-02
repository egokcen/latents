"""Test GFA inference routines."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.fit
def test_fit(fitted_model_converged):
    """Test basic fitting: convergence flags and iteration count."""
    model = fitted_model_converged

    assert model.flags.converged
    assert not model.flags.decreasing_lb
    assert not model.flags.private_var_floor
    # Regression baselines for fixed seeds (simulation_seed=0, fitting_seed=0).
    # x_dim_init=10, true x_dim=7, so 3 latents pruned.
    assert model.flags.x_dims_removed == 3
    # Regression baseline: iteration count from examples gallery fit (fit_tol=1e-8).
    assert len(model.tracker.iter_time) == 2493


@pytest.mark.fit
def test_elbo_monotonicity(fitted_model_converged):
    """Test that ELBO is monotonically non-decreasing.

    Uses sqrt(machine epsilon) as tolerance to account for floating-point
    accumulation errors while remaining precision-aware.
    """
    model = fitted_model_converged
    lb = np.array(model.tracker.lb)

    # sqrt(machine epsilon) as tolerance for floating-point accumulation
    tol = float(np.sqrt(np.finfo(lb.dtype).eps))

    lb_diff = np.diff(lb)
    assert np.all(lb_diff >= -tol), (
        f"ELBO decreased by more than tolerance. "
        f"Min diff: {lb_diff.min():.2e}, tolerance: {-tol:.2e}"
    )


@pytest.mark.fit
def test_derived_quantities(fitted_model_converged):
    """Test that derived posterior quantities are consistent with simulation.

    Simulation uses snr=1.0 and a sparsity pattern with 4 active latents
    per group (7 total across 3 groups). Checks that the fitted model's
    SNR, dimensionalities, and squared norms reflect this structure.
    """
    model = fitted_model_converged
    n_groups = len(model.obs_posterior.y_dims)

    # SNR: simulation used snr=1.0 for all groups.
    # Bayesian shrinkage and finite samples cause deviation.
    snr = model.obs_posterior.compute_snr()
    assert snr.shape == (n_groups,)
    assert np.all(snr > 0.5), f"SNR too low: {snr}"
    assert np.all(snr < 2.0), f"SNR too high: {snr}"

    # Dimensionalities: 4 active latents per group from sparsity pattern.
    num_dim, sig_dims, var_exp, _dim_types = (
        model.obs_posterior.compute_dimensionalities()
    )
    n_dim_types = 2**n_groups
    assert num_dim.shape == (n_dim_types,)
    assert sig_dims.shape == (n_groups, model.obs_posterior.x_dim)
    assert var_exp.shape == (n_groups, n_dim_types)
    # Each group should have 4 significant latents
    active_per_group = sig_dims.sum(axis=1)
    np.testing.assert_array_equal(active_per_group, [4, 4, 4])

    # Squared norms: active dimensions should dominate pruned ones.
    norms = model.obs_posterior.C.compute_squared_norms(model.obs_posterior.y_dims)
    assert norms.shape == (n_groups, model.obs_posterior.x_dim)
    # For each group, the 4 largest norms should be >> the rest
    for g in range(n_groups):
        sorted_norms = np.sort(norms[g])[::-1]
        assert sorted_norms[3] > 1.0, f"Group {g}: 4th largest norm too small"
        assert sorted_norms[4] < 0.1, f"Group {g}: 5th largest norm too large"


@pytest.mark.fit
def test_parameter_recovery(simulation_result, fitted_model_converged):
    """Test that fitted parameters recover ground truth loading matrix.

    Compares estimated C matrix against ground truth using per-column
    correlation. Columns are reordered and sign-flipped to align with
    ground truth (ordering is arbitrary in factor models).

    Thresholds calibrated from baseline run with fixed seeds:
    - Min per-column correlation: 0.77 (column 0, group-specific latent)
    - Mean correlation: 0.94
    """
    model = fitted_model_converged
    obs_params_true = simulation_result.obs_params

    # Regression baselines from examples gallery fit: column reordering and
    # sign flips for this seed combination (simulation_seed=0, fitting_seed=0).
    reorder = np.array([1, 4, 3, 6, 0, 2, 5])
    rescale = np.array([-1, -1, 1, 1, 1, 1, 1])

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
