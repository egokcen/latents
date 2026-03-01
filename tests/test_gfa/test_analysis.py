"""Test GFA analysis functions."""

from __future__ import annotations

import numpy as np
import pytest

from latents.gfa.analysis import predictive_performance
from latents.observation.realizations import ObsParamsRealization


def _expected_predictive_performance(
    obs_params: ObsParamsRealization,
) -> tuple[float, float]:
    """Compute analytical leave-group-out R2 and MSE from ground truth parameters.

    For each left-out group j, the posterior covariance of X given the
    remaining groups is:

        Lambda_{-j} = I + sum_{g != j} C_g^T diag(phi_g) C_g
        Sigma_{X|{-j}} = inv(Lambda_{-j})

    The expected squared prediction error for group j has two sources:
    uncertainty in the inferred X, and observation noise:

        residual_j = tr(C_j Sigma_{X|{-j}} C_j^T) + sum(1/phi_j)

    The total variance of group j is:

        total_j = tr(C_j C_j^T) + sum(1/phi_j)

    Aggregating across groups:

        R2  = 1 - sum_j(residual_j) / sum_j(total_j)
        MSE = sum_j(residual_j) / y_dim
    """
    y_dims = obs_params.y_dims
    n_groups = len(y_dims)
    x_dim = obs_params.x_dim
    y_dim = int(y_dims.sum())

    # Split parameters by group
    C_groups = np.split(obs_params.C, np.cumsum(y_dims)[:-1])
    phi_groups = np.split(obs_params.phi, np.cumsum(y_dims)[:-1])

    sum_residual = 0.0
    sum_total = 0.0

    for j in range(n_groups):
        # Precision of X posterior given all groups except j
        # Lambda_{-j} = I + sum_{g != j} C_g^T diag(phi_g) C_g
        Lambda = np.eye(x_dim)
        for g in range(n_groups):
            if g == j:
                continue
            # C_g: (y_dim_g, x_dim), phi_g: (y_dim_g,)
            Lambda += C_groups[g].T @ np.diag(phi_groups[g]) @ C_groups[g]

        Sigma = np.linalg.inv(Lambda)  # (x_dim, x_dim)

        # residual_j: prediction uncertainty + observation noise
        noise_j = np.sum(1.0 / phi_groups[j])
        residual_j = np.trace(C_groups[j] @ Sigma @ C_groups[j].T) + noise_j

        # total_j: signal variance + observation noise
        total_j = np.trace(C_groups[j] @ C_groups[j].T) + noise_j

        sum_residual += residual_j
        sum_total += total_j

    R2 = 1.0 - sum_residual / sum_total
    MSE = sum_residual / y_dim

    return R2, MSE


@pytest.mark.fit
class TestPredictivePerformance:
    """Tests for leave-group-out predictive performance.

    Uses the module-scoped fitted model from conftest. The simulation has
    SNR=1.0 (signal variance = noise variance), 3 groups x 10 dims,
    7 latents with structured sparsity, and 100 samples.

    The tolerances here are primarily regression tests. The analytical formula
    computes a population-level expectation; with n=100, finite-sample noise
    dominates the gap. Convergence verified empirically: the gap shrinks
    and is < 0.001 at n=100,000.
    """

    def test_r2_matches_analytical(self, simulation_result, fitted_model_converged):
        """Fitted R2 should be close to the analytical expectation.

        The analytical value uses ground truth parameters (point values).
        The fitted model uses posterior means, so the Bayesian subtlety
        (E[C C^T] != E[C] E[C]^T in infer_latents) causes small differences.
        With n=100, finite-sample noise contributes ~0.04 absolute error.
        """
        R2_expected, _ = _expected_predictive_performance(
            simulation_result.obs_params,
        )
        R2_fitted, _ = predictive_performance(
            simulation_result.observations,
            fitted_model_converged.obs_posterior,
        )
        np.testing.assert_allclose(R2_fitted, R2_expected, atol=0.05)

    def test_mse_matches_analytical(self, simulation_result, fitted_model_converged):
        """Fitted MSE should be close to the analytical expectation.

        With n=100, finite-sample noise contributes ~6% relative error.
        """
        _, MSE_expected = _expected_predictive_performance(
            simulation_result.obs_params,
        )
        _, MSE_fitted = predictive_performance(
            simulation_result.observations,
            fitted_model_converged.obs_posterior,
        )
        np.testing.assert_allclose(MSE_fitted, MSE_expected, rtol=0.10)

    def test_returns_float_tuple(self, simulation_result, fitted_model_converged):
        """Return type is a tuple of two floats."""
        R2, MSE = predictive_performance(
            simulation_result.observations,
            fitted_model_converged.obs_posterior,
        )
        assert isinstance(R2, float)
        assert isinstance(MSE, float)
