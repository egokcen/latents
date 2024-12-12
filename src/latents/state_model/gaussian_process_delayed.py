"""
Gaussian Process kernel functions for delayed latent models.

**Classes**

- :class:`GPParams` -- Parameters for Gaussian Process kernel.

**Functions**

- :func:`construct_single_latent_delayed_kernel` -- \
    Construct kernel matrix for a single latent dimension.
- :func:`construct_delayed_kernel` -- Construct full delayed kernel matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GPParams:
    """
    Parameters for Gaussian Process kernel.

    Parameters
    ----------
    gamma
        `ndarray` of `float`, shape ``(x_dim,)``.
        Length scale parameters for each dimension.
    eps
        `ndarray` of `float`, shape ``(x_dim,)``.
        Noise variance parameters for each dimension.
    D
        `ndarray` of `float`, shape ``(num_groups, x_dim)``.
        Delay matrix for each group and dimension.

    Attributes
    ----------
    gamma
        Same as **gamma**, above.
    eps
        Same as **eps**, above.
    D
        Same as **D**, above.
    num_groups
        Number of groups, derived from D.shape[0].
    x_dim
        Number of dimensions, derived from D.shape[1].

    Raises
    ------
    ValueError
        If dimensions of gamma, eps, and D are not consistent.
    """

    gamma: np.ndarray
    eps: np.ndarray
    D: np.ndarray

    def __post_init__(self) -> None:
        """Validate dimensions and set derived attributes."""
        if not (len(self.gamma) == len(self.eps) == self.D.shape[1]):
            error_message = (
                f"Dimension mismatch: gamma {self.gamma.shape}, "
                f"eps {self.eps.shape}, D {self.D.shape[1]}"
            )
            raise ValueError(error_message)

        self.num_groups = self.D.shape[0]
        self.x_dim = self.D.shape[1]

    @classmethod
    def generate(
        cls,
        x_dim: int,
        num_groups: int,
        tau_lim: tuple[float, float],
        eps_lim: tuple[float, float],
        delay_lim: tuple[float, float],
        rng: np.random.Generator | None = None,
    ) -> GPParams:
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

        Returns
        -------
        GPParams
            Randomly generated GP parameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        gamma = rng.uniform(tau_lim[0], tau_lim[1], size=x_dim)
        eps = rng.uniform(eps_lim[0], eps_lim[1], size=x_dim)
        D = rng.uniform(delay_lim[0], delay_lim[1], size=(num_groups, x_dim))
        D[0, :] = 0
        return cls(gamma, eps, D)


def construct_single_latent_delayed_kernel(
    x: np.ndarray, T: int, return_tensor: bool = False, order: str = "F"
) -> np.ndarray:
    """
    Construct delayed kernel matrix for a single latent dimension.

    This function is written to be differentiable with respect to the parameters x.

    Parameters
    ----------
    x
        `ndarray` of `float`, shape ``(num_groups + 2,)``.
        Array containing [delays, noise_variance, length_scale] for the
        current latent dimension.
    T
        Number of time points.
    return_tensor
        If ``True``, return 4D tensor.
        If ``False``, return flattened matrix. Defaults to ``False``.
    order
        Order for reshaping ('F' for Fortran, 'C' for C). Defaults to ``'F'``.

    Returns
    -------
    ndarray
        If return_tensor is ``True``, shape ``(num_groups, T, num_groups, T)``.
        If return_tensor is ``False``, shape ``(num_groups*T, num_groups*T)``.
        Kernel matrix for the current latent dimension.
    """
    gamma_j = x[-1]
    eps_j = x[-2]
    D_j = x[:-2]
    num_groups = D_j.shape[0]

    Kj = np.zeros((num_groups, T, num_groups, T))
    t = np.arange(T)
    t2_minus_t1 = t[np.newaxis, :] - t[:, np.newaxis]

    for m1 in range(num_groups):
        for m2 in range(num_groups):
            diff = t2_minus_t1 - (D_j[m2] - D_j[m1])
            Kj[m1, :, m2, :] = (1 - eps_j) * np.exp(-0.5 * gamma_j * diff**2)
            if m1 == m2:
                Kj[m1, :, m2, :] += eps_j * np.eye(T)

    if return_tensor:
        return Kj
    return Kj.reshape(num_groups * T, num_groups * T, order=order)


def construct_delayed_kernel(
    gp_params: GPParams, T: int, return_tensor: bool = False, order: str = "F"
) -> np.ndarray:
    """
    Construct full delayed kernel matrix across all dimensions.

    Parameters
    ----------
    gp_params
        Gaussian Process kernel parameters.
    T
        Number of time points.
    return_tensor
        If ``True``, return 6D tensor.
        If ``False``, return flattened matrix. Defaults to ``False``.
    order
        Order for reshaping ('F' for Fortran, 'C' for C). Defaults to ``'F'``.

    Returns
    -------
    ndarray
        If return_tensor is ``True``, shape\
        ``(x_dim, num_groups, T, x_dim, num_groups, T)``.
        If return_tensor is ``False``, shape\
        ``(x_dim*num_groups*T, x_dim*num_groups*T)``.
        Full kernel matrix across all dimensions.

    Raises
    ------
    TypeError
        If gp_params is not an instance of GPParams.
    """
    if not isinstance(gp_params, GPParams):
        error_message = "First argument must be a GPParams instance"
        raise TypeError(error_message)

    D = gp_params.D
    gamma = gp_params.gamma
    eps = gp_params.eps

    num_groups = D.shape[0]
    x_dim = D.shape[1]
    K = np.zeros((x_dim, num_groups, T, x_dim, num_groups, T))

    for dim in range(x_dim):
        x = np.concatenate([D[:, dim], [eps[dim]], [gamma[dim]]])
        K[dim, :, :, dim, :, :] = construct_single_latent_delayed_kernel(
            x, T, return_tensor=True
        )

    if return_tensor:
        return K
    return K.reshape(x_dim * num_groups * T, x_dim * num_groups * T, order=order)
