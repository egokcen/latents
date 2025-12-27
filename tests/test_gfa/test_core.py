"""Test the gfa.core module."""


def test_fit():
    """Test basic fitting."""
    import numpy as np

    import latents.gfa.simulation as gfa_sim
    from latents.gfa import GFAFitConfig, GFAModel
    from latents.observation_model.probabilistic import SimulationHyperPriors

    # Set a random seed, for reproducibility
    random_seed = 1  # Set to None for no seeding

    # Dataset characteristics
    N = 100  # Total number of samples
    y_dims = np.array([10, 10, 10])  # Dimensionality of each observed group
    num_groups = len(y_dims)  # Total number of groups
    x_dim = 7  # Latent dimensionality
    snr = 1.0 * np.ones(num_groups)  # Signal-to-noise ratio of each group

    # Build up the desired sparsity pattern of the loading matrices, a
    # (num_groups x x_dim) array. Row i corresponds to group i. Column j
    # corresponds to latent j. A value of np.inf indicates that a latent is
    # NOT present in a group. The corresponding loadings will be 0 for that
    # group. The remaining hyperparameters are not very important, and can
    # be left alone.
    sparsity_pattern = np.array(
        [
            [1, 1, 1, np.inf, 1, np.inf, np.inf],
            [1, 1, np.inf, 1, np.inf, 1, np.inf],
            [1, np.inf, 1, 1, np.inf, np.inf, 1],
        ],
    )
    MAG = 100  # Control the variance of alpha parameters (larger = less var.)
    sim_priors = SimulationHyperPriors(
        a_alpha=MAG * sparsity_pattern,
        b_alpha=MAG * np.ones_like(sparsity_pattern),
        a_phi=1.0,
        b_phi=1.0,
        d_beta=1.0,
    )

    # Simulate data
    Y, _, _ = gfa_sim.simulate(
        N,
        y_dims,
        x_dim,
        sim_priors,
        snr,
        random_seed=random_seed,
    )

    # Configure fitting
    config = GFAFitConfig(
        x_dim_init=10,  # Set to larger than the hypothesized latent dimensionality
        fit_tol=1e-8,  # Tolerance to determine fitting convergence
        max_iter=20000,  # Maximum number of fitting iterations
        verbose=True,  # Print fitting progress
        random_seed=0,  # Set to None for no seeding
        min_var_frac=0.001,  # Private variance floor
        prune_x=True,  # For speed-up, remove latents that become inactive
        prune_tol=1e-7,  # Tolerance for pruning inactive latents
        save_x=False,  # Set False to save memory when saving final results
        save_c_cov=False,  # Set False to save memory when saving final results
        save_fit_progress=True,  # Save lower bound, runtime each iteration
    )

    # Instantiate a GFA model with config
    model = GFAModel(config=config)

    # Initialize the model
    model.init(Y)

    # Fit the model
    model.fit(Y)

    # Check model fit flags
    assert model.flags.converged
    assert not model.flags.decreasing_lb
    assert not model.flags.private_var_floor
    assert model.flags.x_dims_removed == 3

    # Check the number of iterations
    assert len(model.tracker.iter_time) == 2521
