"""Base kernel class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

# Ensure JAX uses 64-bit precision
jax.config.update("jax_enable_x64", True)

# TO DO TO SIMPLIFY:
# Base_kernel should contain the parameters that are common to all kernels: gamma and delay from RBFParams and the eps
# Base_kernel should contain pack_params, unpack_params, update_params

# rbf_kernel should contain K_single_latent,K_full, (that contain math specific to rbf)


@dataclass(slots=True, frozen=True, unsafe_hash=True)
class DelayedGPHyperParams:
    """Base hyperparameters common to all kernels."""

    min_gamma: float = 1e-3
    max_delay: float = 5.0


@jit
def compute_elbo_from_K_single_latent(
    Ki: jnp.ndarray, X_moment_gp_i: jnp.ndarray, N: int
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


@dataclass
class BaseDelayedGP:
    """Base parameters common to all kernels."""

    gamma: jnp.ndarray | None = None
    delays: jnp.ndarray | None = None  # delays
    eps: jnp.ndarray | None = None
    hyper_params: DelayedGPHyperParams = None

    def is_initialized(self) -> bool:
        """Check if parameters have been initialized."""
        return (
            self.gamma is not None and self.eps is not None and self.delays is not None
        )

    def __post_init__(self) -> None:
        """Validate dimensions and set derived attributes."""
        if self.is_initialized():
            # Check positivity constraints
            if not np.all(self.gamma > 0):
                msg = "gamma must contain positive values"
                raise ValueError(msg)
            if not np.all(self.eps > 0):
                msg = "eps must contain positive values"
                raise ValueError(msg)

            # Check dimension consistency
            if not (len(self.gamma) == len(self.eps) == self.delays.shape[1]):
                msg = (
                    f"Dimension mismatch: gamma {self.gamma.shape}, "
                    f"eps {self.eps.shape}, delays {self.delays.shape[1]}"
                )
                raise ValueError(msg)

            self.num_groups = self.delays.shape[0]
            self.x_dim = self.delays.shape[1]

    def _set_derived_attributes(self):
        """Set derived attributes without validation (for performance)."""
        if self.is_initialized():
            self.num_groups = self.delays.shape[0]
            self.x_dim = self.delays.shape[1]

    @classmethod
    def generate(
        cls,
        x_dim: int,
        num_groups: int,
        delay_lim: tuple[float, float] = (-5.0, 5.0),
        eps_lim: tuple[float, float] = (1e-4, 0.1),
        gamma_lim: tuple[float, float] = (0.01, 0.5),
        rng: np.random.Generator | None = None,
    ) -> BaseDelayedGP:
        """Generate random GP parameters.

        Parameters
        ----------
        x_dim : int
            Number of latent dimensions
        num_groups : int
            Number of groups
        gamma_lim : tuple[float, float], optional
            Limits for timescale parameter. Defaults to (0.01, 0.5)
        eps_lim : tuple[float, float], optional
            Limits for noise parameter. Defaults to (1e-4, 0.1)
        delay_lim : tuple[float, float], optional
            Limits for delay parameter. Defaults to (-5.0, 5.0)
        rng : np.random.Generator, optional
            Random number generator

        Returns
        -------
        BaseGPParams
            Randomly generated GP parameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        gamma = rng.uniform(gamma_lim[0], gamma_lim[1], size=x_dim)
        eps = rng.uniform(eps_lim[0], eps_lim[1], size=x_dim)
        delays = rng.uniform(delay_lim[0], delay_lim[1], size=(num_groups, x_dim))
        delays[0, :] = 0
        return cls(
            gamma=gamma, eps=eps, delays=delays, hyper_params=DelayedGPHyperParams()
        )

    @staticmethod
    def pack_params_single_latent(gamma, delays, hyper_params, i: int) -> jnp.ndarray:
        """Pack the kernel parameters for a single latent dimension.

        This is common to all kernels since they all use gamma, delay, and eps.
        """
        gamma_i = gamma[i]
        delays_i = delays[:, i]

        # Transform to unconstrained space
        log_gamma = jnp.log(gamma_i - hyper_params.min_gamma)
        # Use inverse of tanh transformation: beta = 2 * atanh(delay / max_delay)
        beta_params = 2.0 * jnp.arctanh(delays_i[1:] / hyper_params.max_delay)

        return jnp.concatenate(
            [jnp.array([log_gamma], dtype=jnp.float64), beta_params.astype(jnp.float64)]
        )

    @staticmethod
    def unpack_params(
        var_i: jnp.ndarray, hyper_params: DelayedGPHyperParams
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Unpack parameters from optimization variables.

        This is common to all kernels since they all use gamma, delay, and eps.
        """
        h = hyper_params
        gamma = h.min_gamma + jnp.exp(var_i[0])
        delays = jnp.concatenate(
            [jnp.zeros(1, var_i.dtype), h.max_delay * jnp.tanh(var_i[1:] / 2.0)]
        )
        return gamma, delays

    def update_params_from_variables(
        self, i: int, var_i_opt: jnp.ndarray
    ) -> BaseDelayedGP:
        """Update parameters from optimized variables.

        This is common to all kernels since they all use gamma, delay, and eps.
        """
        gamma_i_opt, delays_i_opt = self.unpack_params(var_i_opt, self.hyper_params)

        new_gamma = self.gamma.copy()
        new_gamma[i] = gamma_i_opt

        new_delays = self.delays.copy()
        new_delays[:, i] = delays_i_opt

        return BaseDelayedGP(
            gamma=new_gamma,
            delays=new_delays,
            eps=self.eps,
            hyper_params=self.hyper_params,
        )

    @staticmethod
    def K_single_latent(
        gamma,
        delays,
        eps,
        hyper_params,
        T: int,
        return_tensor: bool = False,
        order: str = "F",
    ) -> jnp.ndarray:
        """Construct delayed kernel matrix for a single latent dimension."""
        raise NotImplementedError

    @staticmethod
    def K_full(
        gamma,
        delays,
        eps,
        hyper_params,
        T: int,
        return_tensor: bool = False,
        order: str = "F",
    ) -> jnp.ndarray:
        """Construct full delayed kernel matrix across all dimensions."""
        raise NotImplementedError

    @staticmethod
    def K_from_variables(variables: jnp.ndarray, T: int, i: int, params) -> jnp.ndarray:
        """Compute kernel matrix K directly from optimization variables."""
        raise NotImplementedError

    @staticmethod
    def elbo(
        variables: jnp.ndarray,
        X_moment_gp_i: jnp.ndarray,
        N: int,
        T: int,
        i: int,
        params,
    ) -> float:
        """Compute ELBO using the shared ELBO computation."""
        K = BaseDelayedGP.K_from_variables(variables, T, i, params)
        return compute_elbo_from_K_single_latent(K, X_moment_gp_i, N)

    @staticmethod
    def compute_elbo_from_K(
        K: jnp.ndarray, X_moment_gp_i: jnp.ndarray, N: int
    ) -> float:
        """Compute ELBO from kernel matrix K (direct access to avoid inheritance)."""
        return compute_elbo_from_K_single_latent(K, X_moment_gp_i, N)


'''    # -----------------------------------------------------------------------------
    # Optimized JAX math functions integrated into base class
    # -----------------------------------------------------------------------------

    def validate_inputs(self, X_moment, D_init, T: int) -> None:
        """Quick shape sanity check."""
        if D_init.ndim != 1:
            raise ValueError(f"D_init must be 1‑D, got {D_init.shape}")
        if not jnp.isclose(D_init[0], 0.0): # Robust float comparison
            raise ValueError(f"D_init[0] must be zero, got {D_init[0]}")
        M = D_init.shape[0]
        if X_moment.shape != (M * T, M * T):
            raise ValueError(
                f"X_moment shape {X_moment.shape} inconsistent with M={M}, T={T}"
            )

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def lower_bound(self, var_i, X, N: int, T: int, eps: float, min_gamma: float, max_delay: float):
        var_i = var_i.astype(jnp.float64)
        X = X.astype(jnp.float64)
        gamma = min_gamma + jnp.exp(var_i[0])
        beta_params = var_i[1:]
        D_transformed = max_delay * jnp.tanh(beta_params / 2.0)
        D = jnp.concatenate((jnp.zeros(1, dtype=var_i.dtype), D_transformed))
        K = self._kernel(gamma, D, T, eps, min_gamma, max_delay)
        L_chol = jnp.linalg.cholesky(K)
        diag_L_chol = jnp.diag(L_chol)
        logdet_val = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L_chol, 1e-30)))
        Kinv_X = jnp.linalg.solve(L_chol.T, jnp.linalg.solve(L_chol, X))
        trace_term_val = jnp.trace(Kinv_X)
        elbo_like_value = -0.5 * N * logdet_val - 0.5 * trace_term_val
        return -elbo_like_value

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def value_and_grad(self, var_i, X, N: int, T: int, eps: float, min_gamma: float, max_delay: float):
        return jax.value_and_grad(self.lower_bound, argnums=0)(var_i, X, N, T, eps, min_gamma, max_delay)

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def lower_bound_manual_grad(self, var_i, X, N: int, T: int, eps: float, min_gamma: float, max_delay: float):
        current_dtype = jnp.float64
        var_i = var_i.astype(current_dtype)
        X = X.astype(current_dtype)
        gamma = min_gamma + jnp.exp(var_i[0])
        beta_params = var_i[1:]
        D_transformed = max_delay * jnp.tanh(beta_params / 2.0)
        D = jnp.concatenate((jnp.zeros(1, dtype=var_i.dtype), D_transformed))
        M = D.shape[0]
        MT = M * T
        t_grid = jnp.arange(T, dtype=current_dtype)
        tdiff_TT = t_grid[None, :] - t_grid[:, None]
        delay_offset_MM = D[None, :] - D[:, None]
        effective_deltaT_MMTT = tdiff_TT[None, None, :, :] - delay_offset_MM[:, :, None, None]
        Kj_temp_values_no_eps_MMTT = (1.0 - eps) * jnp.exp(-0.5 * gamma * effective_deltaT_MMTT**2)
        m_diag_mask_MM = jnp.eye(M, dtype=current_dtype)
        eye_T_TT = jnp.eye(T, dtype=current_dtype)
        eps_term_MMTT = eps * m_diag_mask_MM[:,:,None,None] * eye_T_TT[None,None,:,:]
        Kj_MMTT = Kj_temp_values_no_eps_MMTT + eps_term_MMTT
        Kj_equiv_for_reshape = Kj_MMTT.transpose(0,2,1,3)
        K = Kj_equiv_for_reshape.reshape(MT, MT, order="F")
        effective_deltaT_for_K = effective_deltaT_MMTT.transpose(0,2,1,3).reshape(MT, MT, order="F")
        effective_temp_no_eps_for_K = Kj_temp_values_no_eps_MMTT.transpose(0,2,1,3).reshape(MT, MT, order="F")
        L_chol = jnp.linalg.cholesky(K)
        diag_L_chol = jnp.diag(L_chol)
        logdet_K = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L_chol, 1e-30)))
        Kinv_X = jnp.linalg.solve(L_chol.T, jnp.linalg.solve(L_chol, X))
        val_L_obj = -0.5 * N * logdet_K - 0.5 * jnp.trace(Kinv_X)
        Linv_eye = jnp.linalg.solve(L_chol, jnp.eye(MT, dtype=current_dtype))
        Kinv_mat = Linv_eye.T @ Linv_eye
        Kinv_X_Kinv = Kinv_X @ Kinv_mat
        A_grad_Lobj_dK = -0.5 * (N * Kinv_mat - Kinv_X_Kinv)
        dK_dgamma_matrix = effective_temp_no_eps_for_K * (-0.5 * effective_deltaT_for_K**2)
        dLobj_dgamma = jnp.sum(A_grad_Lobj_dK * dK_dgamma_matrix)
        var_i_beta_params = var_i[1:]
        tanh_val = jnp.tanh(var_i_beta_params / 2.0)
        dD_dvar_iBeta = (max_delay * 0.5 * (1.0 - tanh_val**2)).astype(current_dtype)
        glob_r_coords = jnp.arange(MT, dtype=jnp.int32)[:, None]
        glob_c_coords = jnp.arange(MT, dtype=jnp.int32)[None, :]
        flat_idx = glob_r_coords + glob_c_coords * MT
        m1_map_for_K = flat_idx % M
        m2_map_for_K = (flat_idx // (M*T)) % M
        deriv_temp_wrt_eff_deltaT = effective_temp_no_eps_for_K * (-gamma * effective_deltaT_for_K)
        k_param_indices = jnp.arange(1, M, dtype=jnp.int32)
        m1_is_k_param_stack = (m1_map_for_K[None, :, :] == k_param_indices[:, None, None])
        m2_is_k_param_stack = (m2_map_for_K[None, :, :] == k_param_indices[:, None, None])
        d_eff_deltaT_dDk_stack = -(m2_is_k_param_stack.astype(current_dtype) - m1_is_k_param_stack.astype(current_dtype))
        dK_dDk_stack = deriv_temp_wrt_eff_deltaT[None, :, :] * d_eff_deltaT_dDk_stack
        dLobj_dD_all_k = jnp.sum(A_grad_Lobj_dK[None, :, :] * dK_dDk_stack, axis=(1,2))
        grads_Lobj_var_i_beta = dLobj_dD_all_k * dD_dvar_iBeta
        grad_Lobj_var_i_gamma_part = dLobj_dgamma * jnp.exp(var_i[0])
        grad_Lobj_var_i = jnp.concatenate(
            [jnp.array([grad_Lobj_var_i_gamma_part], dtype=current_dtype), 
             grads_Lobj_var_i_beta.astype(current_dtype)]
        )
        return -val_L_obj, -grad_Lobj_var_i

    def _kernel(self, gamma, D, T: int, eps: float, min_gamma: float, max_delay: float):
        """Vectorized kernel construction - to be implemented by subclasses."""
        raise NotImplementedError
'''
