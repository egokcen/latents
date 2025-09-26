"""Simple test to verify automatic and manual gradients are close enough."""

import numpy as np
import jax.numpy as jnp

from latents.mdlag.gp.kernels.rbf.rbf_kernel import RBFKernel
from latents.mdlag.gp.multigroup_params import (
    MultiGroupGPHyperParams,
    MultiGroupGPParams,
)


def test_gradient_consistency():
    """Test that automatic and manual gradients are close enough."""
    # Simple test setup
    T = 10  # Number of time points
    M = 2  # Number of groups
    N = 50  # Number of samples

    # Create hyperparameters
    hyper_params = MultiGroupGPHyperParams(max_delay=5.0, min_gamma=0.01)

    # Create synthetic moment data
    rng = np.random.default_rng(42)
    X_moment = rng.standard_normal((M * T, M * T))
    X_moment = X_moment @ X_moment.T  # Make it positive semi-definite

    # Create test parameters (single latent dimension)
    gamma = np.array([0.1])
    delays = np.array([[0.0], [1.0]])  # 2 groups, 1 latent dimension
    eps = np.array([1e-4])  # Must be an array

    # Create parameter object and pack parameters properly
    params = MultiGroupGPParams(gamma=gamma, delays=delays, eps=eps)
    var = params.pack_params_single_latent(gamma, delays, hyper_params, 0)

    # Convert to JAX arrays
    X_moment_jax = jnp.array(X_moment, dtype=jnp.float64)
    var_jax = jnp.array(var, dtype=jnp.float64)

    # Create kernel
    kernel = RBFKernel()

    # Compute automatic gradient
    f_auto, grad_auto = kernel._compute_objective_and_gradient_autodiff(
        var_jax, X_moment_jax, N, T, eps[0], hyper_params
    )

    # Compute manual gradient
    f_manual, grad_manual = kernel._compute_objective_and_gradient_manual(
        var_jax, X_moment_jax, N, T, eps[0], hyper_params
    )

    # Check that function values are close
    assert np.allclose(f_auto, f_manual, rtol=1e-10, atol=1e-10), (
        f"Function values differ: auto={f_auto}, manual={f_manual}"
    )

    # Check that gradients are close
    assert np.allclose(grad_auto, grad_manual, rtol=1e-6, atol=1e-6), (
        f"Gradients differ: auto={grad_auto}, manual={grad_manual}"
    )
