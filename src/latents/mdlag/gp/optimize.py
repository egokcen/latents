"""Generic optimization for GP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxopt import LBFGSB

from .fit_config import GPFitConfig
from .kernels.base_kernel import GPKernelSpec

jax.config.update("jax_enable_x64", True)


def run_gp_optimizer(
    gp_spec: GPKernelSpec, X_moment: jnp.ndarray, N: int, T: int, cfg: GPFitConfig
) -> GPKernelSpec:
    """Run the GP optimizer."""
    kernel = gp_spec.kernel
    current_params = gp_spec.params

    # Ensure parameters are initialized
    if (
        not hasattr(current_params, "is_initialized")
        or not current_params.is_initialized()
    ):
        msg = "GP parameters must be initialized before optimization"
        raise ValueError(msg)

    x_dim = current_params.gamma.shape[0]

    for i in range(x_dim):
        X_moment_i = X_moment[:, i, :]
        objective_fn = kernel.get_objective_single_latent(
            current_params, i, X_moment_i, N, T
        )
        var_i = kernel.pack_params_single_latent(current_params, i)

        val_and_grad_fn = jax.value_and_grad(objective_fn)

        solver = LBFGSB(
            fun=lambda x: val_and_grad_fn(x)[0],
            value_and_grad=val_and_grad_fn,
            maxiter=cfg.max_iter,
            tol=cfg.tol,
            jit=True,
        )

        var_i_opt = solver.run(var_i).params

        current_params = kernel.update_params_from_variables(
            current_params, i, var_i_opt
        )
    return GPKernelSpec(kernel=kernel, params=current_params)


def generic_gp_elbo(K_i: jnp.ndarray, X_moment_i: jnp.ndarray, N: int) -> float:
    """ELBO term (positive) - **autodiff friendly** version."""
    X_moment_i = X_moment_i.astype(jnp.float64)
    K_i = K_i.astype(jnp.float64)
    L = jnp.linalg.cholesky(K_i)
    Y = jnp.linalg.solve(L, X_moment_i)
    trace_term = jnp.trace(jnp.linalg.solve(L.T, Y))
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return (-1.0) * (-0.5 * N * logdet - 0.5 * trace_term)
