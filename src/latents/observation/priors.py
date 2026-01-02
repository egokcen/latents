"""Hyperpriors and prior distributions for observation model parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latents.observation.realizations import ObsParamsRealization


@dataclass(frozen=True, slots=True, kw_only=True)
class ObsParamsHyperPrior:
    """Homogeneous hyperpriors for observation model parameters.

    Scalar values broadcast to all groups/latents. Typical for inference
    with uninformative priors.

    Parameters
    ----------
    a_alpha
        Shape parameter of the ARD prior (Gamma). Must be > 0.
    b_alpha
        Rate parameter of the ARD prior (Gamma). Must be > 0.
    a_phi
        Shape parameter of the observation precision prior (Gamma). Must be > 0.
    b_phi
        Rate parameter of the observation precision prior (Gamma). Must be > 0.
    beta_d
        Precision of the observation mean prior (Gaussian). Must be > 0.

    Examples
    --------
    >>> priors = ObsParamsHyperPrior()  # Use defaults (uninformative)
    >>> priors = ObsParamsHyperPrior(a_alpha=1e-6, b_alpha=1e-6)
    """

    a_alpha: float = 1e-12
    b_alpha: float = 1e-12
    a_phi: float = 1e-12
    b_phi: float = 1e-12
    beta_d: float = 1e-12

    def __post_init__(self) -> None:
        """Validate all parameters are positive."""
        for name in ("a_alpha", "b_alpha", "a_phi", "b_phi", "beta_d"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                msg = f"{name} must be a positive number, got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObsParamsHyperPriorStructured:
    """Structured hyperpriors with per-group, per-latent control.

    Enables sparsity constraints (np.inf in a_alpha forces zero loadings)
    and incorporation of prior knowledge.

    Parameters
    ----------
    a_alpha
        Shape parameters for ARD priors, shape (n_groups, x_dim).
        Use np.inf to force zero loadings (sparsity pattern).
    b_alpha
        Rate parameters for ARD priors, shape (n_groups, x_dim).
        Typically ones or matched to a_alpha.
    a_phi
        Shape parameter of observation precision prior. Must be > 0.
    b_phi
        Rate parameter of observation precision prior. Must be > 0.
    beta_d
        Precision of observation mean prior. Must be > 0.

    Examples
    --------
    >>> # 3 groups, 4 latents, with sparsity pattern
    >>> sparsity = np.array([
    ...     [1, 1, np.inf, 1],      # Group 0: latents 0,1,3
    ...     [1, np.inf, 1, 1],      # Group 1: latents 0,2,3
    ...     [np.inf, 1, 1, 1],      # Group 2: latents 1,2,3
    ... ])
    >>> priors = ObsParamsHyperPriorStructured(
    ...     a_alpha=100 * sparsity,
    ...     b_alpha=100 * np.ones((3, 4)),
    ... )
    """

    a_alpha: np.ndarray
    b_alpha: np.ndarray
    a_phi: float = 1.0
    b_phi: float = 1.0
    beta_d: float = 1.0

    def __post_init__(self) -> None:
        """Validate array shapes and scalar positivity."""
        # Validate a_alpha is ndarray
        if not isinstance(self.a_alpha, np.ndarray):
            msg = f"a_alpha must be a numpy array, got {type(self.a_alpha).__name__}"
            raise TypeError(msg)

        # Validate b_alpha is ndarray
        if not isinstance(self.b_alpha, np.ndarray):
            msg = f"b_alpha must be a numpy array, got {type(self.b_alpha).__name__}"
            raise TypeError(msg)

        # Validate shapes match
        if self.a_alpha.shape != self.b_alpha.shape:
            msg = (
                f"a_alpha shape {self.a_alpha.shape} "
                f"must match b_alpha shape {self.b_alpha.shape}"
            )
            raise ValueError(msg)

        # Validate 2D
        if self.a_alpha.ndim != 2:
            msg = f"a_alpha must be 2D (n_groups, x_dim), got {self.a_alpha.ndim}D"
            raise ValueError(msg)

        # Validate b_alpha values are positive (a_alpha can have np.inf)
        if np.any(self.b_alpha <= 0):
            msg = "b_alpha values must all be > 0"
            raise ValueError(msg)

        # Validate finite a_alpha values are positive
        finite_mask = np.isfinite(self.a_alpha)
        if np.any(self.a_alpha[finite_mask] <= 0):
            msg = "Finite a_alpha values must be > 0 (use np.inf for sparsity)"
            raise ValueError(msg)

        # Validate scalars
        for name in ("a_phi", "b_phi", "beta_d"):
            value = getattr(self, name)
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

    @property
    def n_groups(self) -> int:
        """Number of observed groups."""
        return self.a_alpha.shape[0]

    @property
    def x_dim(self) -> int:
        """Number of latent dimensions."""
        return self.a_alpha.shape[1]


@dataclass
class ObsParamsPrior:
    """Prior distributions over observation model parameters.

    Encapsulates p(C, d, phi, alpha) and handles correct sampling order
    (alpha must be sampled before C, since C|alpha ~ N(0, alpha^-1)).

    Parameters
    ----------
    hyperprior
        Hyperprior parameters controlling the prior distributions.
    """

    hyperprior: ObsParamsHyperPrior | ObsParamsHyperPriorStructured

    def sample(
        self,
        y_dims: np.ndarray,
        x_dim: int,
        rng: np.random.Generator,
    ) -> ObsParamsRealization:
        """Sample from joint prior p(alpha)p(C|alpha)p(d)p(phi).

        Parameters
        ----------
        y_dims
            Dimensionalities of each observed group, shape (n_groups,).
        x_dim
            Number of latent dimensions.
        rng
            Random number generator.

        Returns
        -------
        ObsParamsRealization
            Sampled parameter values.
        """
        n_groups = len(y_dims)
        y_dim = int(y_dims.sum())

        # Get hyperprior values, expanding scalars to arrays if needed
        if isinstance(self.hyperprior, ObsParamsHyperPriorStructured):
            a_alpha = self.hyperprior.a_alpha
            b_alpha = self.hyperprior.b_alpha
        else:
            a_alpha = np.full((n_groups, x_dim), self.hyperprior.a_alpha)
            b_alpha = np.full((n_groups, x_dim), self.hyperprior.b_alpha)

        # Sample observation mean: d ~ N(0, 1/beta_d)
        d = rng.normal(0, 1 / np.sqrt(self.hyperprior.beta_d), size=y_dim)

        # Sample observation precision: phi ~ Gamma(a_phi, b_phi)
        phi = rng.gamma(
            shape=self.hyperprior.a_phi,
            scale=1 / self.hyperprior.b_phi,
            size=y_dim,
        )

        # Sample ARD parameters and loadings group by group
        alpha = np.zeros((n_groups, x_dim))
        C = np.zeros((y_dim, x_dim))

        # Split C by group for in-place assignment
        y_boundaries = np.cumsum(y_dims)[:-1]
        C_split = np.split(C, y_boundaries, axis=0)

        for group_idx in range(n_groups):
            for x_idx in range(x_dim):
                a = a_alpha[group_idx, x_idx]
                b = b_alpha[group_idx, x_idx]

                if np.isinf(a):
                    # Infinite shape parameter forces zero loadings
                    alpha[group_idx, x_idx] = np.inf
                    C_split[group_idx][:, x_idx] = 0.0
                else:
                    # Sample alpha ~ Gamma(a, b)
                    alpha[group_idx, x_idx] = rng.gamma(shape=a, scale=1 / b)
                    # Sample C|alpha ~ N(0, 1/alpha)
                    C_split[group_idx][:, x_idx] = rng.normal(
                        0,
                        1 / np.sqrt(alpha[group_idx, x_idx]),
                        size=y_dims[group_idx],
                    )

        return ObsParamsRealization(
            C=C,
            d=d,
            phi=phi,
            alpha=alpha,
            y_dims=y_dims.copy(),
            x_dim=x_dim,
        )
