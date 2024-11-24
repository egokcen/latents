"""
Building blocks for Gaussian Process latent state models.

**Classes**

- :class:`GP_Params` -- Base class for GP kernel parameters.
- :class:`RBF_GP_Params` -- Parameters for RBF kernel GP state model.

**Functions**
- :func:`construct_K_mdlag` -- Construct GP covariance tensor for mDLAG model.
- :func:`construct_K_mdlag_fast` -- Construct GP covariance matrix for mDLAG model.
- :func:`tensor_to_matrix` -- Convert covariance tensor to matrix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GP_Params(ABC):
    """Abstract base class for Gaussian Process parameters.

    Parameters
    ----------
    x_dim : int
        Number of latent dimensions
    num_groups : int
        Number of groups
    T : int
        Number of time points
    """

    x_dim: int
    num_groups: int
    T: int

    def check_indices(self, t1: int, t2: int, j1: int, j2: int, m1: int, m2: int):
        """Validate that the provided indices are within allowed ranges.

        Parameters
        ----------
        t1, t2 : int
            Time indices
        j1, j2 : int
            Latent dimension indices
        m1, m2 : int
            Group indices

        Raises
        ------
        AssertionError
            If any index is out of its valid range
        """
        assert 0 <= j1 < self.x_dim, f"Index j1={j1} out of range [0, {self.x_dim - 1}]"
        assert 0 <= j2 < self.x_dim, f"Index j2={j2} out of range [0, {self.x_dim - 1}]"
        assert (
            0 <= m1 < self.num_groups
        ), f"Group m1={m1} out of range [0, {self.num_groups - 1}]"
        assert (
            0 <= m2 < self.num_groups
        ), f"Group m2={m2} out of range [0, {self.num_groups - 1}]"
        assert 0 <= t1 < self.T, f"Time t1={t1} out of range [0, {self.T - 1}]"
        assert 0 <= t2 < self.T, f"Time t2={t2} out of range [0, {self.T - 1}]"

    @abstractmethod
    def kernel_func(
        self, t1: int, t2: int, j1: int, j2: int, m1: int, m2: int
    ) -> float:
        """Compute kernel function value for given indices."""
        pass


@dataclass
class RBF_GP_Params(GP_Params):
    """RBF (Radial Basis Function) Gaussian Process parameters.

    Parameters
    ----------
    eps : ndarray, shape (x_dim,)
        Noise variances
    gamma : ndarray, shape (x_dim,)
        Inverse squared GP timescales
    D : ndarray, shape (num_groups, x_dim)
        Delay matrix
    """

    name: str = field(default="rbf", init=False)  # Class-level constant
    eps: np.ndarray
    gamma: np.ndarray
    D: np.ndarray

    def __post_init__(self):
        # Verify shapes
        assert self.gamma.shape == (
            self.x_dim,
        ), f"gamma shape mismatch: expected ({self.x_dim},), got {self.gamma.shape}"
        assert self.eps.shape == (
            self.x_dim,
        ), f"eps shape mismatch: expected ({self.x_dim},), got {self.eps.shape}"
        msg = (
            f"D shape mismatch: expected ({self.num_groups}, {self.x_dim}), "
            f"got {self.D.shape}"
        )
        assert self.D.shape == (self.num_groups, self.x_dim), msg

    def kernel_func(
        self, t1: int, t2: int, j1: int, j2: int, m1: int, m2: int
    ) -> float:
        """Compute kernel function value for given indices."""
        self.check_indices(t1, t2, j1, j2, m1, m2)
        if j1 != j2:
            return 0.0  # Covariance is zero between different latent dimensions
        j = j1
        # Compute difference accounting for delays
        diff = (t1 - self.D[m1, j]) - (t2 - self.D[m2, j])
        base_kernel = (1 - self.eps[j]) * np.exp(-0.5 * self.gamma[j] * diff**2)

        if m1 == m2 and t1 == t2:
            base_kernel += self.eps[j]

        return base_kernel

    @classmethod
    def generate(
        cls,
        x_dim: int,
        num_groups: int,
        tau_lim: tuple[float, float],
        eps_lim: tuple[float, float],
        delay_lim: tuple[float, float],
        rng: np.random.Generator | None = None,
    ) -> RBF_GP_Params:
        """Generate random mDLAG GP parameters.

        Parameters
        ----------
        x_dim : int
            Number of latent dimensions
        num_groups : int
            Number of groups
        tau_lim : tuple[float, float]
            Limits for timescale parameter
        eps_lim : tuple[float, float]
            Limits for noise parameter
        delay_lim : tuple[float, float]
            Limits for delay parameter
        rng : np.random.Generator, optional
            Random number generator

        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        ERROR_MSG = "generate_GPparams_mdlag.m is not yet implemented"
        raise NotImplementedError(ERROR_MSG)


def construct_K_mdlag(params: GP_Params, return_matrix: bool = False) -> np.ndarray:
    """Construct the GP covariance tensor using explicit loops.

    Parameters
    ----------
    params : GP_Params
        GP parameters
    return_matrix : bool, optional
        If True, returns flattened covariance matrix, by default False

    Returns
    -------
    ndarray
        Covariance tensor or matrix depending on return_matrix
    """
    x_dim, num_groups, T = params.x_dim, params.num_groups, params.T
    K_tensor = np.zeros((x_dim, num_groups, T, x_dim, num_groups, T))

    for m1 in range(num_groups):
        for j1 in range(x_dim):
            for j2 in range(x_dim):
                for m2 in range(num_groups):
                    for t1 in range(T):
                        for t2 in range(T):
                            K_tensor[j1, m1, t1, j2, m2, t2] = params.kernel_func(
                                t1, t2, j1, j2, m1, m2
                            )

    return tensor_to_matrix(K_tensor) if return_matrix else K_tensor


def construct_K_mdlag_fast(
    params: GP_Params, return_matrix: bool = False
) -> np.ndarray:
    """Construct the GP covariance tensor using vectorized operations.

    Parameters
    ----------
    params : GP_Params
        GP parameters
    return_matrix : bool, optional
        If True, returns flattened covariance matrix, by default False

    Returns
    -------
    ndarray
        If return_matrix is False:
            Covariance tensor with shape (x_dim, num_groups, T, x_dim, num_groups, T)
        If return_matrix is True:
            Flattened matrix with shape (x_dim*num_groups*T, x_dim*num_groups*T)

    Notes
    -----
    This is a faster implementation than construct_K_mdlag_tensor using broadcasting.
    """
    x_dim, num_groups, T = params.x_dim, params.num_groups, params.T
    eps, gamma, D = params.eps, params.gamma, params.D
    # t1,m1,t2,m2
    t1 = np.arange(T)[None, :, None, None]
    t2 = np.arange(T)[None, None, None, :]
    m1 = np.arange(num_groups)[:, None, None, None]
    m2 = np.arange(num_groups)[None, None, :, None]

    K_tensor = np.zeros((x_dim, num_groups, T, x_dim, num_groups, T))
    mask_diag = (m1 == m2) & (t1 == t2)

    for j in range(x_dim):
        D1 = D[m1, j]
        D2 = D[m2, j]

        diff = (t1 - t2) - (D1 - D2)

        base_kernel = (1 - eps[j]) * np.exp(-0.5 * gamma[j] * diff**2)

        K_tensor[j, :, :, j, :, :] = base_kernel + eps[j] * mask_diag

    if return_matrix:
        return K_tensor.reshape(
            x_dim * num_groups * T, x_dim * num_groups * T, order="F"
        )
    return K_tensor


def tensor_to_matrix(K_tensor: np.ndarray) -> np.ndarray:
    """Convert the 6D covariance tensor to a matrix.

    Parameters
    ----------
    K_tensor : ndarray
        Covariance tensor with shape (j1, m1, t1, j2, m2, t2)

    Returns
    -------
    ndarray
        Flattened matrix with shape (j1*m1*t1, j2*m2*t2)
    """
    # Get the shape of the tensor
    j1, m1, t1, j2, m2, t2 = K_tensor.shape

    # Fortran order: first index changing fastest,
    return K_tensor.reshape(j1 * m1 * t1, j2 * m2 * t2, order="F")
