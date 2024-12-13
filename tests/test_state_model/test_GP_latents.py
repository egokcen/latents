"""Test the GPlatents submodule."""


def test_GP_covariance_matrix():
    """Test the construction of the GP covariance matrix match matlab."""
    import os

    import numpy as np
    from scipy.io import loadmat

    from latents.state_model.gaussian_process import (
        GPParams,
        construct_gp_covariance_matrix,
    )

    # Dataset characteristics:
    T = 25  # Number of samples per sequence
    x_dim = 7
    binWidth = 20  # Sample period of ground truth data

    # Define GP parameters:
    tau = np.array([30, 80, 50, 120, 100, 40, 70])  # GP timescales
    eps = 1e-1 * np.ones(x_dim)  # GP noise variances
    D = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [15, -30, 0, 0, 0, 0, 0], [30, 0, -25, 40, 0, 0, 0]]
    )
    D = D / binWidth
    gamma = (binWidth / tau) ** 2

    gp_params = GPParams(eps=eps, gamma=gamma, D=D)

    K_big = construct_gp_covariance_matrix(
        gp_params, T=T, return_tensor=False, order="F"
    )

    fixture_path = os.path.join(os.path.dirname(__file__), "K_big.mat")
    matlab_data = loadmat(fixture_path)
    K_matlab = matlab_data["K_big"]

    assert np.allclose(K_big, K_matlab)
