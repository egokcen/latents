"""ELBO computation for GP optimization."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def generic_gp_elbo(K_i: jnp.ndarray, X_moment_i: jnp.ndarray, N: int) -> float:
    """ELBO term, autodiff friendly."""
    X_moment_i = X_moment_i.astype(jnp.float64)
    K_i = K_i.astype(jnp.float64)
    L = jnp.linalg.cholesky(K_i)
    # Use jax.scipy.linalg.solve_triangular for better performance
    Y = jax.scipy.linalg.solve_triangular(L, X_moment_i, lower=True)
    trace_term = jnp.trace(jax.scipy.linalg.solve_triangular(L.T, Y, lower=False))
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return (-1.0) * (-0.5 * N * logdet - 0.5 * trace_term)
