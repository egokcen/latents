"""Generic optimization for multi-group GP kernels in mDLAG."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .fit_config import GPFitConfig
from .kernels.multigroup_kernel import MultiGroupGPKernel
from .multigroup_params import MultiGroupGPHyperParams, MultiGroupGPParams


def run_gp_optimizer(
    params: MultiGroupGPParams,
    kernel: MultiGroupGPKernel,
    X_moment: jnp.ndarray,
    N: int,
    T: int,
    cfg: GPFitConfig,
    hyper_params: MultiGroupGPHyperParams,
) -> tuple[MultiGroupGPParams, float]:
    """Run the GP optimizer with the new multi-group kernel structure.

    Parameters
    ----------
    params : MultiGroupGPParams
        Initial GP parameters.
    kernel : MultiGroupGPKernel
        Kernel instance (e.g., RBFKernel).
    X_moment : jnp.ndarray
        Moment data, shape (x_dim, num_groups*T, num_groups*T).
    N : int
        Number of samples.
    T : int
        Number of time points.
    cfg : GPFitConfig
        Optimization configuration.
    hyper_params : MultiGroupGPHyperParams
        Hyperparameters for parameter constraints.

    Returns
    -------
    tuple[MultiGroupGPParams, float]
        Updated parameters and total loss.
    """
    if not params.is_initialized():
        msg = "GP parameters must be initialized before optimization"
        raise ValueError(msg)

    x_dim = params.x_dim
    total_loss = 0.0

    # Create a copy of parameters to update
    updated_params = MultiGroupGPParams(
        gamma=jnp.array(params.gamma),
        delays=jnp.array(params.delays),
        eps=jnp.array(params.eps),
    )

    for i in range(x_dim):
        # Extract data for this latent dimension
        X_moment_i = X_moment[i, :, :]

        # Get initial parameters for this latent
        var_i = MultiGroupGPParams.pack_params_single_latent(
            params.gamma, params.delays, hyper_params, i
        )
        eps = float(params.eps[i])

        # Create value-and-gradient function
        use_autodiff = cfg.grad_mode == "autodiff"

        def val_and_grad(var_i):
            return kernel.compute_objective_and_gradient(
                var_i, X_moment_i, N, T, eps, hyper_params, use_autodiff
            )

        var_i_np = np.array(var_i)

        result = fmin_l_bfgs_b(
            func=val_and_grad,
            x0=var_i_np,
            fprime=None,
            maxiter=cfg.max_iter,
            maxfun=cfg.max_iter * 10,
            factr=1e7,
            pgtol=1e-4, 
            m=15,
        )
        var_i_opt = result[0]
        f_opt = result[1]

        var_i_opt = jnp.array(var_i_opt)
        total_loss += f_opt

        # Update the parameters object
        updated_params = updated_params.update_params_from_variables(
            i, var_i_opt, hyper_params
        )

    return updated_params, total_loss
