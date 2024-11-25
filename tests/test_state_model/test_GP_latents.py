"""Test the GPlatents submodule."""


def test_GP_covariance_matrix():
    """Test the construction of the GP covariance matrix match matlab."""
    import os

    import numpy as np
    from scipy.io import loadmat

    from latents.state_model.GP_latents import (
        RBF_GP_Params,
        construct_K_mdlag,
        construct_K_mdlag_fast,
    )

    # Dataset characteristics:
    T = 25  # Number of samples per sequence
    y_dims = np.array([10, 11, 12])  # Dimensionalities of each observed group
    num_groups = len(y_dims)  # Total number of groups
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

    gp_params = RBF_GP_Params(
        x_dim=x_dim, num_groups=num_groups, T=T, eps=eps, gamma=gamma, D=D
    )

    K_big_fast = construct_K_mdlag_fast(gp_params, return_matrix=True)
    K_big = construct_K_mdlag(gp_params, return_matrix=True)

    fixture_path = os.path.join(os.path.dirname(__file__), "K_big.mat")
    matlab_data = loadmat(fixture_path)
    K_matlab = matlab_data["K_big"]

    assert np.allclose(K_big_fast, K_matlab)
    assert np.allclose(K_big, K_matlab)
    assert np.allclose(K_big_fast, K_big)
