"""Test the mdlag.simulation module."""

import numpy as np
import pytest

from latents.mdlag.simulation import generate_latents, generate_observations
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD
from latents.state_model.gaussian_process import (
    GPParams,
    construct_gp_covariance_matrix,
)
from latents.state_model.latents import StateParamsDelayed


@pytest.fixture
def simulation_params():
    """Fixture to set up shared simulation parameters."""
    # Dataset characteristics:
    T = 25  # Number of samples per sequence
    y_dims = np.array([10, 11, 12])
    num_groups = len(y_dims)

    x_dim = 7
    binWidth = 20
    snr = 10.0 * np.ones(len(y_dims))

    # Define hyperpriors:
    MAG = 100.0
    sparsity_pattern = np.array(
        [
            [1, 1, 1, np.inf, 1, np.inf, np.inf],
            [1, 1, np.inf, 1, np.inf, 1, np.inf],
            [1, np.inf, 1, 1, np.inf, np.inf, 1],
        ]
    )
    hyper_priors = HyperPriorParams(
        a_alpha=MAG * sparsity_pattern,
        b_alpha=MAG * np.ones_like(sparsity_pattern),
        a_phi=1.0,
        b_phi=1.0,
        d_beta=1.0,
    )

    # Define GP parameters:
    tau = np.array([30, 80, 50, 120, 100, 40, 70])  # GP timescales
    eps = 1e-4 * np.ones(x_dim)  # GP noise variances
    D = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [15, -30, 0, 0, 0, 0, 0], [30, 0, -25, 40, 0, 0, 0]]
    )
    D = D / binWidth
    gamma = (binWidth / tau) ** 2

    obs_params = ObsParamsARD.generate(
        y_dims, x_dim, hyper_priors, snr, np.random.default_rng(seed=42)
    )
    gp_params = GPParams(gamma=gamma, eps=eps, D=D)

    state_params_delayed = StateParamsDelayed(x_dim, num_groups, T)

    return gp_params, state_params_delayed, obs_params


def test_generate_latents(simulation_params):
    """Test latents covariance matrix."""
    gp_params, state_params_delayed, _ = simulation_params
    rng = np.random.default_rng(seed=42)

    # Generate latents
    N = int(1e4)  # Number of samples
    latents = generate_latents(gp_params, N=N, T=state_params_delayed.T, rng=rng)

    # Compute empirical covariance matrix
    latents_flat = latents.reshape(-1, N, order="F")
    empirical_cov = np.cov(latents_flat)

    # Construct theoretical covariance matrix
    K_big = construct_gp_covariance_matrix(
        gp_params, T=state_params_delayed.T, return_tensor=False, order="F"
    )

    # Check similarity of empirical covariance and K_big
    assert np.allclose(
        empirical_cov, K_big, atol=1e-1
    ), "Empirical covariance does not match theoretical covariance matrix K_big"


def test_generate_observations(simulation_params):
    """Test observations covariance matrix."""
    gp_params, state_params_delayed, obs_params = simulation_params
    rng = np.random.default_rng(seed=42)

    # Generate latents
    N = int(1e5)  # Number of samples
    X = generate_latents(gp_params, N=N, T=state_params_delayed.T, rng=rng)

    # Generate observations
    Y = generate_observations(X, obs_params, rng)
    Ys = Y.get_groups()

    # Split observation model parameters according to observed groups
    y_dims = obs_params.y_dims
    ds, _ = obs_params.d.get_groups(y_dims)
    phis, _ = obs_params.phi.get_groups(y_dims)
    Cs, _, _ = obs_params.C.get_groups(y_dims)

    for group_idx, y_dim in enumerate(y_dims):
        observations_g = Ys[group_idx].data

        # Compute empirical mean
        empirical_mean_g = np.mean(observations_g, axis=(1, 2))
        d_g = ds[group_idx]

        assert np.allclose(
            d_g, empirical_mean_g, atol=1e-1
        ), f"Empirical mean for group {group_idx} does not match expected mean"

        # Compute empirical covariance
        observations_g_flat = np.reshape(
            observations_g, (y_dims[group_idx], -1)
        )  # Shape: (N_samples, yDims_g)
        empirical_cov_g = np.cov(observations_g_flat)

        # Compute theoretical covariance
        C_g = Cs[group_idx]
        Phi_g_inv = np.diag(1.0 / phis[group_idx])
        Cov_x_t = np.eye(
            state_params_delayed.x_dim
        )  # Simplification: assume Cov[x_t] is identity
        theoretical_cov_g = C_g @ Cov_x_t @ C_g.T + Phi_g_inv

        assert np.allclose(
            empirical_cov_g, theoretical_cov_g, atol=1e-1
        ), f"Empirical and theoretical covariance don't match for group {group_idx}"
