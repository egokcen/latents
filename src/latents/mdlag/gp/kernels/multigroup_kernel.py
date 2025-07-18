"""Base multi-group GP kernel class for mDLAG."""

from functools import partial

import jax
import jax.numpy as jnp

from ..multigroup_params import MultiGroupGPHyperParams, MultiGroupGPParams

# Ensure JAX uses 64-bit precision
jax.config.update("jax_enable_x64", True)


class MultiGroupGPKernel:
    """Base class for multi-group GP kernels in mDLAG."""

    def compute_elbo_from_kernel_matrix(
        self, Ki: jnp.ndarray, X_moment_gp_i: jnp.ndarray, N: int
    ) -> float:
        """Compute ELBO from kernel matrix K. This is the same for all kernels."""
        Ki = Ki.astype(jnp.float64)
        X_moment_gp_i = X_moment_gp_i.astype(jnp.float64)
        L_chol = jnp.linalg.cholesky(Ki)
        diag_L_chol = jnp.diag(L_chol)
        logdet_val = 2.0 * jnp.sum(jnp.log(diag_L_chol))
        Kinv_X = jnp.linalg.solve(L_chol.T, jnp.linalg.solve(L_chol, X_moment_gp_i))
        trace_term_val = jnp.trace(Kinv_X)
        elbo_like_value = -0.5 * N * logdet_val - 0.5 * trace_term_val
        return -elbo_like_value

    def build_full_kernel_matrix(
        self,
        params: MultiGroupGPParams,
        T: int,
        return_tensor: bool = False,
        order: str = "F",
    ) -> jnp.ndarray:
        """Build full kernel matrix across all dimensions."""
        x_dim = params.x_dim
        num_groups = params.num_groups

        # Initialize full kernel
        K = jnp.zeros((x_dim, num_groups, T, x_dim, num_groups, T), dtype=jnp.float64)

        # Fill diagonal blocks
        for i in range(x_dim):
            gamma_i = params.gamma[i]
            delays_i = params.delays[:, i]
            # Convert JAX scalar to Python float for hashability in JIT
            eps_i = float(params.eps[i])

            K_i = self.build_single_latent_kernel(
                gamma_i, delays_i, eps_i, T, return_tensor=True
            )
            K = K.at[i, :, :, i, :, :].set(K_i)

        if return_tensor:
            return K
        return K.reshape(x_dim * num_groups * T, x_dim * num_groups * T, order=order)

    def compute_objective_and_gradient(
        self,
        var: jnp.ndarray,
        X: jnp.ndarray,
        N: int,
        T: int,
        eps: float,
        hyper_params: MultiGroupGPHyperParams,
        use_autodiff: bool = True,
    ) -> tuple[float, jnp.ndarray]:
        """Compute objective value and gradient for optimization.

        This is the main interface for optimizers - it returns both value and gradient
        in a single call to avoid duplicate computation.
        """
        if use_autodiff:
            return self._compute_objective_and_gradient_autodiff(
                var, X, N, T, eps, hyper_params
            )
        return self._compute_objective_and_gradient_manual(
            var, X, N, T, eps, hyper_params
        )

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
    def _compute_objective_and_gradient_autodiff(
        self,
        var: jnp.ndarray,
        X: jnp.ndarray,
        N: int,
        T: int,
        eps: float,
        hyper_params: MultiGroupGPHyperParams,
    ) -> tuple[float, jnp.ndarray]:
        """Automatic gradient computation using JAX autodiff."""

        def objective(var):
            gamma, delays = MultiGroupGPParams.unpack_params(var, hyper_params)
            K = self.build_single_latent_kernel(gamma, delays, eps, T)
            return self.compute_elbo_from_kernel_matrix(K, X, N)

        return jax.value_and_grad(objective)(var)

    def _compute_objective_and_gradient_manual(
        self,
        var: jnp.ndarray,
        X: jnp.ndarray,
        N: int,
        T: int,
        eps: float,
        hyper_params: MultiGroupGPHyperParams,
    ) -> tuple[float, jnp.ndarray]:
        """Manual gradient computation - kernel specific implementation."""
        msg = "Subclasses must implement _compute_objective_and_gradient_manual"
        raise NotImplementedError(msg)

    def build_single_latent_kernel(
        self,
        gamma: jnp.ndarray,
        delays: jnp.ndarray,
        eps: float,
        T: int,
        return_tensor: bool = False,
        order: str = "F",
    ) -> jnp.ndarray:
        """Build kernel matrix - kernel specific implementation.

        This method must be implemented by subclasses to provide kernel-specific
        behavior.
        """
        msg = "Subclasses must implement build_single_latent_kernel"
        raise NotImplementedError(msg)
