"""RBF kernel."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from ...optimize import generic_gp_elbo
from ..base_kernel import BaseKernel, GPKernelSpec
from .rbf_params import RBFHyperParams, RBFParams


class RBFKernel(BaseKernel):
    """RBF (Radial Basis Function) kernel implementation."""

    @classmethod
    def initialize(cls, x_dim: int, num_groups: int) -> GPKernelSpec:
        """Create a GPKernelSpec by delegating to RBFParams.generate."""
        initial_params = RBFParams.generate(x_dim, num_groups)
        return GPKernelSpec(kernel=cls(), params=initial_params)

    # Core Math:

    def K_single_latent(
        self,
        params_i: RBFParams | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        T: int,
        return_tensor: bool = False,
        order: str = "F",
    ) -> jnp.ndarray:
        """Construct delayed kernel matrix for a single latent dimension.

        This function is written to be differentiable with respect to the parameters x.

        Parameters
        ----------
        params_i : RBFParams or tuple
            Either an RBFParams object or a tuple of (gamma_i, D_i, eps_i) as JAX arrays
        """
        # Handle both RBFParams objects and JAX array tuples
        if isinstance(params_i, RBFParams):
            if not params_i.is_initialized():
                msg = "Parameters not initialized"
                raise ValueError(msg)
            # Convert to jax arrays for mathematical operations
            D_i = jnp.asarray(params_i.D)
            gamma_i = jnp.asarray(
                params_i.gamma[0] if params_i.gamma.ndim > 0 else params_i.gamma
            )
            eps_i = jnp.asarray(
                params_i.eps[0] if params_i.eps.ndim > 0 else params_i.eps
            )
        else:
            # Assume it's a tuple of (gamma_i, D_i, eps_i)
            gamma_i, D_i, eps_i = params_i

        M = D_i.shape[0]
        tgrid = jnp.arange(T)
        tdiff = tgrid[None, :] - tgrid[:, None]  # (T,T)
        Ki = jnp.zeros((M, T, M, T), dtype=jnp.float64)

        for m1 in range(M):
            for m2 in range(M):
                diff = tdiff - (D_i[m2] - D_i[m1])
                block = (1.0 - eps_i) * jnp.exp(-0.5 * gamma_i * diff**2)
                if m1 == m2:
                    block += eps_i * jnp.eye(T, dtype=jnp.float64)
                Ki = Ki.at[m1, :, m2, :].set(block)
        if return_tensor:
            return Ki
        return Ki.reshape(M * T, M * T, order=order)

    def K_full(
        self, params: RBFParams, T: int, return_tensor: bool = False, order: str = "F"
    ) -> jnp.ndarray:
        """Construct full delayed kernel matrix across all dimensions."""
        # Convert to jax arrays for mathematical operations
        D = jnp.asarray(params.D)
        gamma = jnp.asarray(params.gamma)
        eps = jnp.asarray(params.eps)

        num_groups = D.shape[0]
        x_dim = D.shape[1]
        K = jnp.zeros((x_dim, num_groups, T, x_dim, num_groups, T), dtype=jnp.float64)
        for dim in range(x_dim):
            params_i = RBFParams(
                gamma=np.array([gamma[dim]]),
                D=np.array(D[:, dim].reshape(-1, 1)),
                eps=np.array([eps[dim]]),
                hyperparams=params.hyperparams,
            )
            K = K.at[dim, :, :, dim, :, :].set(
                self.K_single_latent(params_i, T, return_tensor=True)
            )
        if return_tensor:
            return K
        return K.reshape(x_dim * num_groups * T, x_dim * num_groups * T, order=order)

    def pack_params_single_latent(self, params: RBFParams, i: int) -> jnp.ndarray:
        """Pack the kernel parameters for a single latent dimension."""
        if not params.is_initialized():
            msg = "Parameters not initialized"
            raise ValueError(msg)
        hyperparams = params.hyperparams
        # Convert to jax arrays for mathematical operations
        gamma_i = jnp.asarray(params.gamma[i])
        D_i = jnp.asarray(params.D[:, i])
        max_delay, min_gamma = hyperparams.max_delay, hyperparams.min_gamma
        log_gamma = jnp.log(gamma_i - min_gamma)
        delay = jnp.log(max_delay + D_i[1:]) - jnp.log(max_delay - D_i[1:])

        return jnp.concatenate(
            (jnp.array([log_gamma], jnp.float64), delay.astype(jnp.float64))
        )

    def unpack_params(
        self, var_i: jnp.ndarray, hyperparams: RBFHyperParams
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Unpack the kernel parameters from a numpy array."""
        gamma = hyperparams.min_gamma + jnp.exp(var_i[0])
        beta = jnp.concatenate((jnp.zeros(1, var_i.dtype), var_i[1:]))
        D = hyperparams.max_delay * jnp.tanh(beta / 2.0)
        return gamma, D

    def get_objective_single_latent(
        self,
        params: RBFParams,
        i: int,
        X_moment_i: jnp.ndarray,
        N: int,
        T: int,
    ) -> Callable[[jnp.ndarray], float]:
        """Return a function to compute the ELBO term for a single latent GP."""
        hyperparams = params.hyperparams
        # Convert to jax array for mathematical operations
        eps_i = jnp.asarray(params.eps[i])

        def objective_fn(var_i: jnp.ndarray) -> float:
            var_i = var_i.astype(jnp.float64)
            gamma_i, D_i = self.unpack_params(var_i, hyperparams)
            # Use the unified kernel computation with JAX arrays
            K_i = self.K_single_latent((gamma_i, D_i, eps_i), T)
            return generic_gp_elbo(K_i, X_moment_i, N)

        return objective_fn

    def compute_loss_all_latents(
        self, params: RBFParams, X_moment: np.ndarray, N: int, T: int
    ) -> float:
        """Compute the objective function for all latent dimensions."""
        L = 0
        for i in range(params.x_dim):
            X_moment_i = X_moment[i, :, :]
            objective_fn = self.get_objective_single_latent(params, i, X_moment_i, N, T)
            var_i = self.pack_params_single_latent(params, i)
            L += objective_fn(var_i)
        return L

    def update_params_from_variables(
        self,
        params: RBFParams,
        i: int,
        var_i_opt: jnp.ndarray,
    ) -> RBFParams:
        """Update the parameters from the i-th variables."""
        gamma_i_opt, D_i_opt = self.unpack_params(var_i_opt, params.hyperparams)
        # Convert to jax arrays for mathematical operations
        new_gamma = params.gamma.copy()
        new_gamma[i] = gamma_i_opt

        new_D = params.D.copy()
        new_D[:, i] = D_i_opt

        return RBFParams(
            gamma=new_gamma,
            eps=params.eps,
            D=new_D,
            hyperparams=params.hyperparams,
        )
