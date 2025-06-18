"""Base kernel class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    def initialize(cls, hyperparams, x_dim: int, num_groups: int) -> GPKernelSpec:
        """Create a fully configured GPKernelSpec."""
        raise NotImplementedError

    def fit(self) -> None:
        """Learn the kernel's parameters."""
        raise NotImplementedError

    def K_single(self) -> np.ndarray:
        """Compute the covariance matrix for a single group."""
        raise NotImplementedError

    def K_full(self) -> np.ndarray:
        """Compute the covariance matrix for all groups."""
        raise NotImplementedError

    def pack_params(self) -> np.ndarray:
        """Pack the kernel parameters into a numpy array."""
        raise NotImplementedError

    def unpack_params(self, params: np.ndarray) -> BaseParams:
        """Unpack the kernel parameters from a numpy array."""
        raise NotImplementedError

    def get_params_for_latent(self, i: int) -> BaseParams:
        """Get the kernel parameters for a specific latent."""
        raise NotImplementedError

    def update_params_for_latent(self, i: int, params: BaseParams) -> None:
        """Update the kernel parameters for a specific latent."""
        raise NotImplementedError

    def grad_log_mll(self, params_single: BaseParams, T: int) -> np.ndarray:
        """Compute the gradient of loss with respect to the kernel parameters."""
        raise NotImplementedError
