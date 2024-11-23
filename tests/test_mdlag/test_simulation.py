"""Test the mdlag.simulation module."""


def test_generate_latents():
    """Test basic fitting."""
    import numpy as np
    from latents.state_model.GP_latents import RBF_GP_Params
    from latents.state_model.GP_latents import construct_K_mdlag_fast
    from latents.state_model.latents import StateParamsGP    
    from latents.mdlag.simulation import generate_latents

    rng = np.random.default_rng(seed=42)

    # Dataset characteristics:
    T = 25  # Number of samples per sequence
    y_dims = np.array([10, 11, 12])  # Dimensionalities of each observed group
    num_groups = len(y_dims)  # Total number of groups
    x_dim = 7
    binWidth = 20  # Sample period of ground truth data
    
    # Define GP parameters:
    tau = np.array([30, 80, 50, 120, 100, 40, 70])  # GP timescales
    eps = 1e-4 * np.ones(x_dim)  # GP noise variances
    D = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [15, -30, 0, 0, 0, 0, 0], [30, 0, -25, 40, 0, 0, 0]]
    )
    D = D / binWidth
    gamma = (binWidth / tau) ** 2

    gp_params = RBF_GP_Params(
        x_dim=x_dim, num_groups=num_groups, T=T, eps=eps, gamma=gamma, D=D
    )
    state_params = StateParamsGP(x_dim=x_dim, num_groups=num_groups, T=T, gp_params=gp_params)

    N = int(1e6)
    latents = generate_latents(state_params, N, rng=rng)

    latents_flat = latents.reshape(N, -1, order="F")
    empirical_cov = np.cov(latents_flat.T)

    # Construct K_big for the given parameters and T
    K_big = construct_K_mdlag_fast(gp_params, return_matrix=True)

    ## Check similarity of empirical covariance and K_big
    assert np.allclose(empirical_cov, K_big, atol=1e-2)

