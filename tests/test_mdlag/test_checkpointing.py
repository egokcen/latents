"""Checkpointing tests for mDLAG model fitting."""

import os
from pathlib import Path

import numpy as np
import pytest

from latents.mdlag.core import mDLAGModel
from latents.mdlag.gp.fit_config import GPFitConfig
from latents.mdlag.gp.gp_model import mDLAGGP
from latents.mdlag.gp.kernels.rbf.rbf_kernel import RBFKernel
from latents.mdlag.simulation import generate_latents, generate_observations
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD


@pytest.mark.fast
def test_checkpoint_create_and_load(tmp_path: Path) -> None:
    """Checkpointing smoke test.

    - Fit a tiny synthetic model with checkpointing enabled
    - Assert that at least one checkpoint file is created
    - Load the last checkpoint and sanity-check a couple of fields
    """
    rng = np.random.RandomState(0)

    # Tiny synthetic problem to keep runtime low
    T = 8
    N = 5
    y_dims = np.array([3, 3, 3])
    x_dim = 2
    bin_width = 20

    # Hyperpriors and GP params (very small)
    MAG = 10.0
    sparsity_pattern = np.ones((len(y_dims), x_dim))
    hyper_priors = HyperPriorParams(
        a_alpha=MAG * sparsity_pattern,
        b_alpha=MAG * np.ones_like(sparsity_pattern),
        a_phi=1.0,
        b_phi=1.0,
        d_beta=1.0,
    )

    tau = np.array([30, 60])[:x_dim]
    eps = 1e-3 * np.ones(x_dim)
    D = np.zeros((len(y_dims), x_dim))
    gamma = (bin_width / tau) ** 2

    obs_params_true = ObsParamsARD.generate(
        y_dims, x_dim, hyper_priors, np.ones(len(y_dims)), rng
    )
    gp_true = mDLAGGP(gamma=gamma, delays=D / bin_width, eps=eps, kernel=RBFKernel())

    X_true = generate_latents(gp_true, T=T, N=N, rng=rng)
    Y = generate_observations(X_true, obs_params_true, rng)

    # Configure model and enable checkpointing frequently
    model = mDLAGModel()
    gp_fit_config = GPFitConfig(
        max_iter=10, tol=1e-6, grad_mode="autodiff", verbose=False
    )

    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.fit_args.set_args(
        gp_fit_config=gp_fit_config,
        hyper_priors=HyperPriorParams(),
        fit_tol=1e-7,
        max_iter=30,
        verbose=False,
        random_seed=0,
        prune_X=True,
        prune_tol=1e-7,
        save_X_cov=False,
        save_C_cov=False,
        save_fit_progress=True,
        checkpoint_interval=5,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Initialize and fit
    model.init(Y, x_dim_init=x_dim, bin_width=bin_width, kernel=RBFKernel(), eps=1e-3)
    model.fit(Y)

    # Check that checkpoints exist
    ckpt_files = sorted(
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_iter_")
    )
    assert len(ckpt_files) >= 1, "Expected at least one checkpoint file to be created"

    # Load last checkpoint and sanity-check a couple of attributes
    last_ckpt = ckpt_files[-1]
    loaded = mDLAGModel.load_from_checkpoint(os.path.join(checkpoint_dir, last_ckpt))

    # Basic structural checks
    assert loaded.params.state_params.x_dim == x_dim
    assert loaded.params.obs_params.C.mean.shape[1] == x_dim
    assert loaded.params.gp.params.gamma.shape[0] == x_dim
