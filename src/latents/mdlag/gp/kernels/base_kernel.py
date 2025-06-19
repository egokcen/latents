"""Base kernel class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass
class GPKernelSpec:
    """A container for the complete specification of a Gaussian Process kernel."""

    kernel: BaseKernel
    params: BaseParams


class BaseParams:
    """Base class for kernel parameters."""

    pass


class BaseKernel:
    """Base class for kernels."""

    @classmethod
    def initialize(cls, x_dim: int, num_groups: int) -> GPKernelSpec:
        """Create a fully configured GPKernelSpec."""
        raise NotImplementedError

    def K_single_latent(
        self, params_i, T: int, return_tensor: bool = False, order: str = "F"
    ) -> jnp.ndarray:
        """Construct delayed kernel matrix for a single latent dimension."""
        raise NotImplementedError

    def K_full(
        self, params, T: int, return_tensor: bool = False, order: str = "F"
    ) -> jnp.ndarray:
        """Construct full delayed kernel matrix across all dimensions."""
        raise NotImplementedError

    def unpack_params(self, variables: jnp.ndarray, hyperparams) -> tuple:
        """Unpack the kernel parameters from a numpy array."""
        raise NotImplementedError

    def get_objective_single_latent(
        self, params, i: int, X_moment_i: jnp.ndarray, N: int, T: int
    ) -> Callable:
        """Return a function to compute the ELBO term for a single latent GP."""
        raise NotImplementedError

    def update_params_from_variables(
        self, params, i: int, variables_i_opt: jnp.ndarray
    ):
        """Update the parameters from the i-th variables."""
        raise NotImplementedError
