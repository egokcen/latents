"""Concrete parameter values for observation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObsParamsRealization:
    """A single realization of observation model parameters.

    Sources: prior sampling, posterior means, posterior samples.

    Parameters
    ----------
    C
        Loading matrices, shape (y_dim, x_dim).
    d
        Observation means, shape (y_dim,).
    phi
        Observation precisions, shape (y_dim,).
    alpha
        ARD parameters, shape (n_groups, x_dim).
    y_dims
        Dimensionalities of each observed group, shape (n_groups,).
    x_dim
        Number of latent dimensions.
    """

    C: np.ndarray
    d: np.ndarray
    phi: np.ndarray
    alpha: np.ndarray
    y_dims: np.ndarray
    x_dim: int

    @property
    def n_groups(self) -> int:
        """Number of observed groups."""
        return len(self.y_dims)

    @property
    def y_dim(self) -> int:
        """Total observed dimensionality."""
        return int(self.y_dims.sum())


@dataclass
class ObsParamsPoint:
    """Point estimates of observation model parameters.

    Source: Non-Bayesian fitting (FA, GPFA, etc.)

    Semantically distinct from ObsParamsRealization—represents "the" optimized
    answer, not "a" sample from a distribution. Does not include alpha
    (non-Bayesian methods do not use ARD).

    Parameters
    ----------
    C
        Loading matrices, shape (y_dim, x_dim).
    d
        Observation means, shape (y_dim,).
    phi
        Observation precisions, shape (y_dim,).
    y_dims
        Dimensionalities of each observed group, shape (n_groups,).
    x_dim
        Number of latent dimensions.
    """

    C: np.ndarray
    d: np.ndarray
    phi: np.ndarray
    y_dims: np.ndarray
    x_dim: int

    @property
    def n_groups(self) -> int:
        """Number of observed groups."""
        return len(self.y_dims)

    @property
    def y_dim(self) -> int:
        """Total observed dimensionality."""
        return int(self.y_dims.sum())


def adjust_snr(
    realization: ObsParamsRealization,
    snr: float | np.ndarray,
    y_dims: np.ndarray | None = None,
) -> ObsParamsRealization:
    """Scale observation precisions to achieve target signal-to-noise ratios.

    SNR is defined as var(signal) / var(noise), where signal variance comes
    from the loading matrices C and noise variance from observation precisions
    phi. This function scales phi to achieve the target SNR per group.

    Parameters
    ----------
    realization
        Observation parameters to adjust.
    snr
        Target SNR. Either a scalar (broadcast to all groups) or per-group
        array of shape ``(n_groups,)``.
    y_dims
        Dimensionalities of each group, shape (n_groups,). If None, uses
        realization.y_dims.

    Returns
    -------
    ObsParamsRealization
        New realization with adjusted phi values. Other parameters unchanged.
    """
    if y_dims is None:
        y_dims = realization.y_dims

    n_groups = len(y_dims)

    # Normalize snr to array
    snr = np.atleast_1d(snr)
    if snr.size == 1:
        snr = np.broadcast_to(snr, (n_groups,))

    # Split C and phi by group
    C_split = np.split(realization.C, np.cumsum(y_dims)[:-1], axis=0)
    phi_adjusted = realization.phi.copy()
    phi_split = np.split(phi_adjusted, np.cumsum(y_dims)[:-1], axis=0)

    for group_idx in range(n_groups):
        # Signal variance: sum of squared loadings
        var_signal = np.sum(C_split[group_idx] ** 2)
        # Desired noise variance to achieve target SNR
        var_noise_desired = var_signal / snr[group_idx]
        # Current noise variance: sum of 1/phi
        var_noise_current = np.sum(1 / phi_split[group_idx])
        # Scale phi to achieve desired noise variance
        phi_split[group_idx] *= var_noise_current / var_noise_desired

    return ObsParamsRealization(
        C=realization.C.copy(),
        d=realization.d.copy(),
        phi=phi_adjusted,
        alpha=realization.alpha.copy(),
        y_dims=realization.y_dims.copy(),
        x_dim=realization.x_dim,
    )
