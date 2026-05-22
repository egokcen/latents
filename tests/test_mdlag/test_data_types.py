"""Tests for mDLAG data types."""

import numpy as np

from latents.mdlag.data_types import mDLAGParams
from latents.mdlag.gp.gp_model import mDLAGGP
from latents.observation_model.probabilistic import HyperPriorParams, ObsParamsARD
from latents.state_model.latents import StateParamsDelayed


def test_mdlag_params_get_subset_dims_out_of_place() -> None:
    """Return a valid subset object when `in_place=False`."""
    rng = np.random.default_rng(seed=0)
    y_dims = np.array([3, 2], dtype=int)
    x_dim = 3
    T = 4
    hyper_priors = HyperPriorParams(
        a_alpha=1.0,
        b_alpha=1.0,
        a_phi=1.0,
        b_phi=1.0,
        d_beta=1.0,
    )

    obs_params = ObsParamsARD.generate(
        y_dims=y_dims,
        x_dim=x_dim,
        hyper_priors=hyper_priors,
        snr=np.ones(len(y_dims)),
        rng=rng,
    )
    gp = mDLAGGP.generate(x_dim=x_dim, num_groups=len(y_dims), rng=rng)
    state_params = StateParamsDelayed(x_dim=x_dim, num_groups=len(y_dims), T=T)
    state_params.X.mean = np.zeros((x_dim, len(y_dims), T, 1))
    state_params.X.cov = np.zeros((x_dim, len(y_dims), T, x_dim, len(y_dims), T))
    state_params.X.moment = np.zeros((len(y_dims), x_dim, x_dim))

    params = mDLAGParams(x_dim=x_dim, y_dims=y_dims.copy(), T=T, gp_init=gp)
    params.obs_params = obs_params
    params.state_params = state_params

    subset_dims = np.array([0, 2], dtype=int)
    subset = params.get_subset_dims(subset_dims, in_place=False)

    assert subset is not None
    assert subset is not params
    assert subset.obs_params.x_dim == len(subset_dims)
    assert subset.state_params.x_dim == len(subset_dims)
    assert subset.gp.params.x_dim == len(subset_dims)

    # Verify original params are not modified.
    assert params.obs_params.x_dim == x_dim
    assert params.state_params.x_dim == x_dim
    assert params.gp.params.x_dim == x_dim
