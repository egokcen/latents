"""RBF kernel implementation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
from functools import partial
from ..base_kernel import BaseDelayedGP, DelayedGPHyperParams

# Ensure 64-bit precision, critical for numerical stability in GP models
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnums=(2, 3))
def build_rbf_kernel(gamma, D, eps, T: int):
    """Vectorized kernel construction."""
    M = D.shape[0]
    current_dtype = D.dtype

    tgrid = jnp.arange(T, dtype=current_dtype)
    tdiff_TT = tgrid[None, :] - tgrid[:, None]  # (T,T)

    delay_offset_MM = D[None, :] - D[:, None]  # (M,M), D[m2]-D[m1]

    effective_deltaT_MMTT = (
        tdiff_TT[None, None, :, :] - delay_offset_MM[:, :, None, None]
    )  # (M,M,T,T)

    # Kj_MMTT[m1, m2, t_row, t_col] is the (t_row, t_col) element of the block between stream m1 and m2
    Kj_no_eps_MMTT = (1.0 - eps) * jnp.exp(
        -0.5 * gamma * effective_deltaT_MMTT**2
    )  # (M,M,T,T)

    m_diag_mask_MM = jnp.eye(M, dtype=current_dtype)
    eye_T_TT = jnp.eye(T, dtype=current_dtype)
    eps_term_MMTT = eps * m_diag_mask_MM[:, :, None, None] * eye_T_TT[None, None, :, :]

    Kj_MMTT = Kj_no_eps_MMTT + eps_term_MMTT  # Shape (M,M,T,T)

    # To match the original code's reshape from an (M,T,M,T) tensor using order="F",
    # we need to transpose Kj_MMTT from (m1,m2,t_row,t_col) to (m1,t_row,m2,t_col).
    Kj_equiv_for_reshape = Kj_MMTT.transpose(0, 2, 1, 3)  # Shape (M,T,M,T)
    # Now, Kj_equiv_for_reshape[m1, t_row, m2, t_col] is the element.
    # This matches the indexing of the original `Kj` variable.
    return Kj_equiv_for_reshape.reshape(M * T, M * T, order="F")


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def lower_bound(var, X, N: int, T: int, eps, hyper_params):
    """ELBO term (target for maximization, or its negative for minimization) – autodiff friendly."""
    var = var.astype(jnp.float64)
    X = X.astype(jnp.float64)

    gamma, D_params = BaseDelayedGP.unpack_params(var, hyper_params)

    K = build_rbf_kernel(gamma, D_params, eps, T)

    L_chol = jnp.linalg.cholesky(K)
    diag_L_chol = jnp.diag(L_chol)
    # Add a small epsilon inside log for stability, though Cholesky should ensure positive diag
    logdet_val = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L_chol, 1e-30)))

    Kinv_X = jnp.linalg.solve(L_chol.T, jnp.linalg.solve(L_chol, X))
    trace_term_val = jnp.trace(Kinv_X)

    elbo_like_value = -0.5 * N * logdet_val - 0.5 * trace_term_val

    # Returns -L_obj, suitable for minimization if L_obj is to be maximized
    return -elbo_like_value


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def value_and_grad_kernel(var, X, N: int, T: int, eps, hyper_params):
    """Return (value, grad) via autodiff."""
    return jax.value_and_grad(lower_bound, argnums=0)(var, X, N, T, eps, hyper_params)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def lower_bound_manual_grad(var, X, N: int, T: int, eps, hyper_params):
    """Return (value, grad) using closed‑form derivatives, vectorized."""
    current_dtype = jnp.float64
    var = var.astype(current_dtype)
    X = X.astype(current_dtype)

    gamma, D_params = BaseDelayedGP.unpack_params(var, hyper_params)
    M = D_params.shape[0]
    MT = M * T

    # --- Construct K and helper derivative terms (Vectorized) ---
    # K construction now matches the fixed _kernel
    t_grid = jnp.arange(T, dtype=current_dtype)
    tdiff_TT = t_grid[None, :] - t_grid[:, None]
    delay_offset_MM = D_params[None, :] - D_params[:, None]

    effective_deltaT_MMTT = (
        tdiff_TT[None, None, :, :] - delay_offset_MM[:, :, None, None]
    )  # (M,M,T,T)
    Kj_temp_values_no_eps_MMTT = (1.0 - eps) * jnp.exp(
        -0.5 * gamma * effective_deltaT_MMTT**2
    )

    m_diag_mask_MM = jnp.eye(M, dtype=current_dtype)
    eye_T_TT = jnp.eye(T, dtype=current_dtype)
    eps_term_MMTT = eps * m_diag_mask_MM[:, :, None, None] * eye_T_TT[None, None, :, :]

    Kj_MMTT = Kj_temp_values_no_eps_MMTT + eps_term_MMTT  # (M,M,T,T)

    Kj_equiv_for_reshape = Kj_MMTT.transpose(0, 2, 1, 3)  # Shape (M,T,M,T)
    K = Kj_equiv_for_reshape.reshape(MT, MT, order="F")

    # For derivatives, we need effective_deltaT and effective_temp_no_eps
    # reshaped consistently with K.
    # effective_deltaT_MMTT was (m1,m2,t_row,t_col)
    # effective_temp_no_eps_MMTT was (m1,m2,t_row,t_col)
    # So, they need the same transpose and reshape as Kj_MMTT to align with K.
    effective_deltaT_for_K = effective_deltaT_MMTT.transpose(0, 2, 1, 3).reshape(
        MT, MT, order="F"
    )
    effective_temp_no_eps_for_K = Kj_temp_values_no_eps_MMTT.transpose(
        0, 2, 1, 3
    ).reshape(MT, MT, order="F")

    # --- End K and helper construction ---

    L_chol = jnp.linalg.cholesky(K)
    diag_L_chol = jnp.diag(L_chol)
    # Add a small epsilon inside log for stability
    logdet_K = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L_chol, 1e-30)))

    Kinv_X = jnp.linalg.solve(L_chol.T, jnp.linalg.solve(L_chol, X))
    val_L_obj = -0.5 * N * logdet_K - 0.5 * jnp.trace(Kinv_X)  # This is L_obj

    # Gradient d(L_obj)/dK (A_grad_Lobj_dK)
    Linv_eye = jnp.linalg.solve(L_chol, jnp.eye(MT, dtype=current_dtype))
    Kinv_mat = Linv_eye.T @ Linv_eye

    Kinv_X_Kinv = Kinv_X @ Kinv_mat
    A_grad_Lobj_dK = -0.5 * (N * Kinv_mat - Kinv_X_Kinv)

    # d(L_obj)/dgamma
    dK_dgamma_matrix = effective_temp_no_eps_for_K * (-0.5 * effective_deltaT_for_K**2)
    dLobj_dgamma = jnp.sum(A_grad_Lobj_dK * dK_dgamma_matrix)

    # d(L_obj)/dBeta_k (for k corresponding to theta[1:])
    beta_params = var[1:]
    tanh_val = jnp.tanh(beta_params / 2.0)
    dD_dthetaBeta = (hyper_params.max_delay * 0.5 * (1.0 - tanh_val**2)).astype(
        current_dtype
    )

    # The m1_map and m2_map need to correctly identify which original m1, m2 block indices
    # contributed to K[r,c] given the (M,T,M,T) source and order="F" reshape.
    # K[r,c] = Source_MTMT[r_idx, t_idx_in_r_block, c_idx, t_idx_in_c_block]
    # where Source_MTMT is Kj_equiv_for_reshape (shape M,T,M,T)
    # Fortran unraveling: K_flat[k] = Source_MTMT[k%M, (k//M)%T, (k//(M*T))%M, k//(M*T*M)]
    # K[r,c] = K_flat[r + c*MT]
    # So, for K[glob_r, glob_c], the linear Fortran index into Source_MTMT is `glob_r + glob_c * MT`.
    # Let flat_idx = glob_r_coords + glob_c_coords * MT
    # m1_map (block row index for Source_MTMT) = flat_idx % M
    # m2_map (block col index for Source_MTMT) = (flat_idx // (M*T)) % M
    # These mappings seem consistent with the corrected kernel structure.
    glob_r_coords = jnp.arange(MT, dtype=jnp.int32)[:, None]
    glob_c_coords = jnp.arange(MT, dtype=jnp.int32)[None, :]
    flat_idx = glob_r_coords + glob_c_coords * MT  # Using MT, not M*T*M*T

    m1_map_for_K = flat_idx % M
    m2_map_for_K = (flat_idx // (M * T)) % M

    deriv_temp_wrt_eff_deltaT = effective_temp_no_eps_for_K * (
        -gamma * effective_deltaT_for_K
    )

    k_param_indices = jnp.arange(1, M, dtype=jnp.int32)

    m1_is_k_param_stack = m1_map_for_K[None, :, :] == k_param_indices[:, None, None]
    m2_is_k_param_stack = m2_map_for_K[None, :, :] == k_param_indices[:, None, None]

    d_eff_deltaT_dDk_stack = -(
        m2_is_k_param_stack.astype(current_dtype)
        - m1_is_k_param_stack.astype(current_dtype)
    )

    dK_dDk_stack = deriv_temp_wrt_eff_deltaT[None, :, :] * d_eff_deltaT_dDk_stack

    dLobj_dD_all_k = jnp.sum(A_grad_Lobj_dK[None, :, :] * dK_dDk_stack, axis=(1, 2))

    grads_Lobj_theta_beta = dLobj_dD_all_k * dD_dthetaBeta

    grad_Lobj_theta_gamma_part = dLobj_dgamma * jnp.exp(var[0])

    grad_Lobj_theta = jnp.concatenate(
        [
            jnp.array([grad_Lobj_theta_gamma_part], dtype=current_dtype),
            grads_Lobj_theta_beta.astype(current_dtype),
        ]
    )
    return -val_L_obj, -grad_Lobj_theta
