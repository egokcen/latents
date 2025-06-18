"""Test the mdlag observation inference."""

import numpy as np

from latents.mdlag.core import (
    infer_ard,
    infer_latents,
    infer_loadings,
    infer_obs_mean,
    infer_obs_prec,
    init as init_mdlag,
)
from latents.mdlag.simulation import generate_latents, generate_observations
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD
from latents.state_model.gaussian_process import GPParams
from latents.state_model.latents import StateParamsDelayed


def generate_params():
    """Generate simulation parameters."""
    # Dataset characteristics
    T = 25  # Number of samples per sequence
    y_dims = np.array([10, 11, 12])
    x_dim = 7
    bin_width = 20
    snr = 10.0 * np.ones(len(y_dims))

    # Define hyperpriors
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

    # Define GP parameters
    tau = np.array([30, 80, 50, 120, 100, 40, 70])  # GP timescales
    eps = 1e-4 * np.ones(x_dim)  # GP noise variances
    D = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [15, -30, 0, 0, 0, 0, 0], [30, 0, -25, 40, 0, 0, 0]]
    )
    D = D / bin_width
    gamma = (bin_width / tau) ** 2

    # Generate parameters
    obs_params = ObsParamsARD.generate(
        y_dims, x_dim, hyper_priors, snr, np.random.default_rng(seed=42)
    )
    gp_params = GPParams(gamma=gamma, eps=eps, D=D)
    state_params = StateParamsDelayed(x_dim, len(y_dims), T)

    return gp_params, state_params, obs_params


def custom_r2_score(y_true, y_pred):
    """Calculate R² score between true and predicted values."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else 1


def test_inference_observation(capsys):
    """Test the inference of observations achieves high R² with ground truth."""
    # Generate ground truth parameters and data
    gp_params_true, state_params_true, obs_params_true = generate_params()
    rng = np.random.default_rng(seed=42)

    # Generate latents and observations
    N = int(1e3)  # Number of samples
    X_true = generate_latents(gp_params_true, N=N, T=state_params_true.T, rng=rng)
    Y = generate_observations(X_true, obs_params_true, rng)

    # Initialize model with true GP parameters
    mdlag_params_true = init_mdlag(
        Y=Y, gp_params_init=gp_params_true, hyper_priors=HyperPriorParams()
    )
    mdlag_params_true.state_params = state_params_true
    mdlag_params_true.obs_params = obs_params_true

    # Initialize test model with slightly perturbed C
    mdlag_params = init_mdlag(
        Y=Y, gp_params_init=gp_params_true, hyper_priors=HyperPriorParams()
    )
    mdlag_params.obs_params.C.mean = (
        mdlag_params_true.obs_params.C.mean
        + 0.4 * rng.standard_normal(mdlag_params_true.obs_params.C.mean.shape)
    )

    # Track R² scores during inference
    num_iterations = 500
    for iter_idx in range(num_iterations):
        infer_latents(Y, mdlag_params, in_place=True)
        infer_obs_mean(Y, mdlag_params, HyperPriorParams(), in_place=True)
        infer_loadings(Y, mdlag_params, XY=None, in_place=True)
        infer_obs_prec(Y, mdlag_params, hyper_priors=HyperPriorParams(), in_place=True)
        infer_ard(mdlag_params, HyperPriorParams(), in_place=True)

    # Calculate R² scores
    final_r2_scores = {
        "C": custom_r2_score(
            obs_params_true.C.mean.flatten(), mdlag_params.obs_params.C.mean.flatten()
        ),
        "d": custom_r2_score(
            obs_params_true.d.mean.flatten(), mdlag_params.obs_params.d.mean.flatten()
        ),
        "phi": custom_r2_score(
            obs_params_true.phi.mean.flatten(),
            mdlag_params.obs_params.phi.mean.flatten(),
        ),
        "X": custom_r2_score(
            X_true.flatten(), mdlag_params.state_params.X.mean.flatten()
        ),
    }
    with capsys.disabled():
        print("\nFinal R² scores:")
        for param_name, score in final_r2_scores.items():
            print(f"{param_name}: {score:.4f}")
            assert score > 0.96, (
                f"R² score for {param_name} is {score:.4f}, expected > 0.96"
            )
