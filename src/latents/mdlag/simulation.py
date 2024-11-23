"""
Simulate data from the mDLAG generative model.

**Functions**

- :func:`simulate` -- Generate samples from the full mDLAG model.
- :func:`generate_latents` -- Generate latents via the state model.
- :func:`generate_observations` -- Generate observations via the observation model.

"""

from __future__ import annotations
from latents.state_model.latents import StateParamsGP
from latents.state_model.GP_latents import construct_K_mdlag_fast
import numpy as np

def simulate():
    """Generate samples from the full mDLAG model."""
    pass


def generate_latents(state_params: StateParamsGP, N: int, rng: np.random.Generator=np.random.default_rng()):
    """Generate latents via the mDLAG state model.
    
    Generates N independent sequences
    according to a zero-mean Gaussian Process defined by the mDLAG state model.
    Returns
    -------
        latent tensor (N, x_dim, num_groups, T)         
    """
    num_groups = state_params.num_groups
    x_dim = state_params.x_dim
    T = state_params.T

    # Construct the covariance matrix K_big for sequences of length T
    K_big = construct_K_mdlag_fast(state_params.gp_params, return_matrix=True)

    # Perform Cholesky decomposition for sampling
    K_big_size = K_big.shape[0]
    try:
        # Adding a small value to the diagonal for numerical stability
        L = np.linalg.cholesky(K_big + 1e-6 * np.eye(K_big_size))
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive-definite.")

    mean = np.zeros(K_big_size)
    #xsm = np.zeros((N,num_groups*x_dim, T))
    latents = np.zeros((N,x_dim,num_groups,T))
    # x_dim, num_groups, T,
    for n in range(N):
        # Sample latent variables
        u = rng.standard_normal(K_big_size)
        x_n = L @ u + mean

        # Reshape to (numGroups * xDim, Tj)
        #xsm[n, :, :] = x_n.reshape((num_groups * x_dim, T), order="F")
        latents[n, :, :, :] = x_n.reshape((x_dim, num_groups, T), order="F")

    return latents

def generate_observations():
    """Generate observations via the mDLAG observation model."""
    pass
