"""Tests for RBF kernel implementation."""

import numpy as np
import jax.numpy as jnp

from latents.mdlag.gp.kernels.rbf.rbf_kernel import RBFKernel
from latents.mdlag.gp.multigroup_params import (
    MultiGroupGPParams,
    MultiGroupGPHyperParams,
)
from latents.state_model.gaussian_process import (
    GPParams,
    construct_gp_covariance_matrix,
)


def test_rbf_kernel_against_original():
    """Test that the optimized RBF kernel matches the original numpy implementation."""
    # Create test parameters
    x_dim = 3
    num_groups = 4
    T = 50

    # Create parameters for both implementations
    hyper_params = MultiGroupGPHyperParams()
    params_new = MultiGroupGPParams.generate(
        x_dim=x_dim,
        num_groups=num_groups,
        hyper_params=hyper_params,
    )

    # Convert to old format for original implementation
    params_old = GPParams(
        gamma=np.array(params_new.gamma),
        eps=np.array(params_new.eps),
        D=np.array(params_new.delays),
    )

    # Create kernel
    kernel = RBFKernel()

    # Compute kernels
    K_new = kernel.K_full(params_new, T, return_tensor=False)
    K_old = construct_gp_covariance_matrix(params_old, T, return_tensor=False)

    # Compare results
    diff = jnp.abs(K_new - K_old).max()
    print(f"Maximum difference: {diff}")
    print(f"Results match: {diff < 1e-10}")

    assert diff < 1e-10, f"Kernels don't match! Max difference: {diff}"
    print("✅ RBF kernel implementation matches original!")


def test_single_latent_kernel():
    """Test single latent kernel construction."""
    # Test parameters
    gamma = 0.1
    delays = jnp.array([0.0, 1.0, -0.5, 2.0])
    eps = 0.01
    T = 30

    kernel = RBFKernel()

    # Build kernel
    K = kernel.build_kernel(gamma, delays, eps, T, return_tensor=False)

    # Basic checks
    assert K.shape == (len(delays) * T, len(delays) * T)
    assert jnp.allclose(K, K.T)  # Should be symmetric
    assert jnp.all(K >= 0)  # Should be positive semi-definite

    print("✅ Single latent kernel construction works!")


def test_kernel_properties():
    """Test that the kernel has expected mathematical properties."""
    # Test parameters
    x_dim = 2
    num_groups = 3
    T = 20

    hyper_params = MultiGroupGPHyperParams()
    params = MultiGroupGPParams.generate(
        x_dim=x_dim,
        num_groups=num_groups,
        hyper_params=hyper_params,
    )

    kernel = RBFKernel()
    K = kernel.K_full(params, T, return_tensor=False)

    # Check properties
    assert K.shape == (x_dim * num_groups * T, x_dim * num_groups * T)
    assert jnp.allclose(K, K.T)  # Symmetric
    assert jnp.all(jnp.linalg.eigvals(K) >= -1e-10)  # Positive semi-definite

    print("✅ Kernel has correct mathematical properties!")


if __name__ == "__main__":
    print("Running RBF kernel tests...")
    test_single_latent_kernel()
    test_kernel_properties()
    test_rbf_kernel_against_original()
    print("All tests passed! 🎉")
