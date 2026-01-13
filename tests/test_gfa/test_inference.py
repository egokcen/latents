"""Test GFA inference routines."""

from __future__ import annotations

import numpy as np


# --- Tests ---


def test_fit(fitted_model_converged):
    """Test basic fitting: convergence flags and iteration count."""
    model = fitted_model_converged

    assert model.flags.converged
    assert not model.flags.decreasing_lb
    assert not model.flags.private_var_floor
    # Regression baselines for fixed seeds (simulation_seed=0, fitting_seed=0).
    # x_dim_init=10, true x_dim=7, so 3 latents pruned.
    assert model.flags.x_dims_removed == 3
    # Iteration count for convergence with fit_tol=1e-8.
    assert len(model.tracker.iter_time) == 2297


def test_elbo_monotonicity(fitted_model_converged):
    """Test that ELBO is monotonically non-decreasing.

    Uses sqrt(machine epsilon) as tolerance to account for floating-point
    accumulation errors while remaining precision-aware.
    """
    from tests.conftest import testing_tols

    model = fitted_model_converged
    lb = np.array(model.tracker.lb)

    # Use rtol from testing_tols (sqrt(eps)) as absolute tolerance for differences
    tols = testing_tols(lb.dtype)
    tol = tols["rtol"]

    lb_diff = np.diff(lb)
    assert np.all(lb_diff >= -tol), (
        f"ELBO decreased by more than tolerance. "
        f"Min diff: {lb_diff.min():.2e}, tolerance: {-tol:.2e}"
    )


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

    # Column reordering and sign flips for this seed combination
    # (simulation_seed=0, fitting_seed=0)
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
