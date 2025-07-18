"""Multi-group GP parameters for mDLAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(slots=True, frozen=True, unsafe_hash=True)
class MultiGroupGPHyperParams:
    """Base hyperparameters common to all kernels."""

    min_gamma: float = 1.6e-05
    max_delay: float = 125


@dataclass
class MultiGroupGPParams:
    """Parameters for multi-group GP kernels in mDLAG.

    Attributes
    ----------
    gamma : jnp.ndarray
        Length scale parameters, shape (x_dim,).
    delays : jnp.ndarray
        Delay parameters, shape (num_groups, x_dim).
    eps : jnp.ndarray
        Noise parameters, shape (x_dim,).
    """

    gamma: jnp.ndarray | None = None
    delays: jnp.ndarray | None = None
    eps: jnp.ndarray | None = None

    def is_initialized(self) -> bool:
        """Check if parameters have been initialized."""
        return (
            self.gamma is not None and self.eps is not None and self.delays is not None
        )

    def __post_init__(self) -> None:
        """Validate dimensions and set derived attributes."""
        if self.is_initialized():
            if not jnp.all(self.gamma > 0):
                msg = "gamma must contain positive values"
                raise ValueError(msg)
            if not jnp.all(self.eps > 0):
                msg = "eps must contain positive values"
                raise ValueError(msg)
            if not (len(self.gamma) == len(self.eps) == self.delays.shape[1]):
                msg = (
                    f"Dimension mismatch: gamma {self.gamma.shape}, "
                    f"eps {self.eps.shape}, delays {self.delays.shape[1]}"
                )
                raise ValueError(msg)
            self.num_groups = self.delays.shape[0]
            self.x_dim = self.delays.shape[1]

    @classmethod
    def generate(
        cls,
        x_dim: int,
        num_groups: int,
        delay_lim: tuple[float, float] = (-5.0, 5.0),
        eps_lim: tuple[float, float] = (0.001, 0.001),  # Default to 0.001
        gamma_lim: tuple[float, float] = (0.01, 0.5),
        rng: Any = None,
        hyper_params: MultiGroupGPHyperParams | None = None,
    ) -> MultiGroupGPParams:
        """Generate random multi-group GP parameters."""
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        # Use hyper_params limits if provided
        if hyper_params is not None:
            gamma_lim = (
                hyper_params.min_gamma,
                gamma_lim[1],
            )  # Use min_gamma from hyper_params
            delay_lim = (
                -hyper_params.max_delay,
                hyper_params.max_delay,
            )  # Use max_delay from hyper_params

        gamma = jnp.array(
            rng.uniform(gamma_lim[0], gamma_lim[1], size=x_dim), dtype=jnp.float64
        )
        eps = jnp.array(
            rng.uniform(eps_lim[0], eps_lim[1], size=x_dim), dtype=jnp.float64
        )
        delays = jnp.array(
            rng.uniform(delay_lim[0], delay_lim[1], size=(num_groups, x_dim)),
            dtype=jnp.float64,
        )
        delays = delays.at[0, :].set(0)
        return cls(gamma=gamma, eps=eps, delays=delays)

    @staticmethod
    def pack_params_single_latent(
        gamma: jnp.ndarray,
        delays: jnp.ndarray,
        hyper_params: MultiGroupGPHyperParams,
        i: int,
    ) -> jnp.ndarray:
        """Pack the kernel parameters for a single latent dimension."""
        gamma_i = gamma[i]
        delays_i = delays[:, i]
        log_gamma = jnp.log(gamma_i - hyper_params.min_gamma)
        beta_params = 2.0 * jnp.arctanh(delays_i[1:] / hyper_params.max_delay)
        return jnp.concatenate(
            [
                jnp.array([log_gamma], dtype=jnp.float64),
                beta_params.astype(jnp.float64),
            ]
        )

    @staticmethod
    def unpack_params(
        var_i: jnp.ndarray, hyper_params: MultiGroupGPHyperParams
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Unpack parameters from optimization variables."""
        gamma = hyper_params.min_gamma + jnp.exp(var_i[0])
        delays = jnp.concatenate(
            [
                jnp.zeros(1, var_i.dtype),
                hyper_params.max_delay * jnp.tanh(var_i[1:] / 2.0),
            ]
        )
        return gamma, delays

    def update_params_from_variables(
        self,
        i: int,
        var_i_opt: jnp.ndarray,
        hyper_params: MultiGroupGPHyperParams,
    ) -> MultiGroupGPParams:
        """Update parameters from optimized variables."""
        gamma_i_opt, delays_i_opt = self.unpack_params(var_i_opt, hyper_params)
        new_gamma = self.gamma.at[i].set(gamma_i_opt)
        new_delays = self.delays.at[:, i].set(delays_i_opt)
        return MultiGroupGPParams(
            gamma=new_gamma,
            delays=new_delays,
            eps=self.eps,
        )
