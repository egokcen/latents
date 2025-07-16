"""Generic optimization for GP, highly optimized version with modular structure."""

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import numpy as np

import jax.numpy as jnp
from jax import value_and_grad
from jaxopt import LBFGS
import jax

# from latents.mdlag.data_types import GPParams
# from .kernels.base_kernel import BaseKernel, GPKernelSpec, BaseGPParams

from .fit_config import GPFitConfig
from .kernels.base_kernel import BaseDelayedGP, DelayedGPHyperParams
from .kernels.rbf.rbf_kernel import build_rbf_kernel, lower_bound
from .kernels.rbf.rbf_kernel import value_and_grad_kernel, lower_bound_manual_grad


def run_gp_optimizer(
    delayed_gp: BaseDelayedGP, X_moment: jnp.ndarray, N: int, T: int, cfg: GPFitConfig
) -> tuple[BaseDelayedGP, float]:
    """Run the GP optimizer with class-based approach using the integrated math."""

    if not delayed_gp.is_initialized():
        msg = "GP parameters must be initialized before optimization"
        raise ValueError(msg)

    x_dim = delayed_gp.x_dim
    total_loss = 0.0

    # Define return:
    updated_delayed_gp = BaseDelayedGP(
        gamma=np.copy(delayed_gp.gamma),
        delays=np.copy(delayed_gp.delays),
        eps=np.copy(delayed_gp.eps),
        hyper_params=delayed_gp.hyper_params,
    )

    hyper_params = delayed_gp.hyper_params

    for i in range(x_dim):
        # Extract data for this latent dimension
        X_moment_i = X_moment[i, :, :]

        # Get initial parameters for this latent
        var_i = delayed_gp.pack_params_single_latent(
            delayed_gp.gamma, delayed_gp.delays, delayed_gp.hyper_params, i
        )
        eps = delayed_gp.eps[i]

        # Create value-and-gradient function
        if cfg.grad_mode == "autodiff":
            val_and_grad = lambda var_i: value_and_grad_kernel(
                var_i, X_moment_i, N, T, eps, hyper_params
            )
        else:
            val_and_grad = lambda var_i: lower_bound_manual_grad(
                var_i, X_moment_i, N, T, eps, hyper_params
            )

        # Compute initial loss for display
        f0, g0 = val_and_grad(var_i)
        print(f"  Initial loss: {f0:.6f}")
        print(f"  Initial gradient: {g0}")

        # Optimize
        solver = LBFGS(
            fun=lambda var_i: val_and_grad(var_i)[0],
            value_and_grad=val_and_grad,
            maxiter=cfg.max_iter,
            tol=cfg.tol,
            # maxls=10,
            # linesearch="hager-zhang",
            jit=True,
        )

        res_i = solver.run(var_i)
        total_loss += res_i.state.value

        gamma_opt, delays_opt = delayed_gp.unpack_params(
            res_i.params, delayed_gp.hyper_params
        )

        # Update the parameters object
        updated_delayed_gp.gamma[i] = gamma_opt
        updated_delayed_gp.delays[:, i] = delays_opt

    return updated_delayed_gp, total_loss
