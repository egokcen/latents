"""RBF kernel parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base_kernel import BaseParams


@dataclass
class RBFParams(BaseParams):
    """Parameters for the RBF kernel.

    Parameters
    ----------
    gamma
        `ndarray` of `float`, shape ``(x_dim,)``.
        Length scale parameters for each dimension. Must be positive.
        Defaults to ``None``.
    eps
        `ndarray` of `float`, shape ``(x_dim,)``.
        Noise variance parameters for each dimension. Must be positive.
        Defaults to ``None``.
    D
        `ndarray` of `float`, shape ``(num_groups, x_dim)``.
        Delay matrix for each group and dimension.
        Defaults to ``None``.

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
        If dimensions of gamma, eps, and D are not consistent, or if gamma
        or eps contain non-positive values.
    """

    gamma: np.ndarray | None = None
    eps: np.ndarray | None = None
    D: np.ndarray | None = None

    learn_gamma: bool = True
    learn_D: bool = True
    learn_eps: bool = False

    def __post_init__(self) -> None:
        """Validate dimensions and set derived attributes."""
        if self.is_initialized():
            # Check positivity constraints
            if not np.all(self.gamma > 0):
                msg = "gamma must contain positive values"
                raise ValueError(msg)
            if not np.all(self.eps > 0):
                msg = "eps must contain positive values"
                raise ValueError(msg)

            # Check dimension consistency
            if not (len(self.gamma) == len(self.eps) == self.D.shape[1]):
                msg = (
                    f"Dimension mismatch: gamma {self.gamma.shape}, "
                    f"eps {self.eps.shape}, D {self.D.shape[1]}"
                )
                raise ValueError(msg)

            self.num_groups = self.D.shape[0]
            self.x_dim = self.D.shape[1]

    def is_initialized(self) -> bool:
        """Check if parameters have been initialized.

        Returns
        -------
        bool
            ``True`` if all parameters (gamma, eps, D) are not None.
        """
        return self.gamma is not None and self.eps is not None and self.D is not None

    @classmethod
    def generate(
        cls,
        x_dim: int,
        num_groups: int,
        delay_lim: tuple[float, float] = (-5.0, 5.0),
        eps_lim: tuple[float, float] = (1e-4, 0.1),
        gamma_lim: tuple[float, float] = (0.01, 0.5),
        rng: np.random.Generator | None = None,
    ) -> RBFParams:
        """Generate random mDLAG GP parameters.

        Parameters
        ----------
        x_dim : int
            Number of latent dimensions
        num_groups : int
            Number of groups
        gamma_lim : tuple[float, float], optional
            Limits for timescale parameter. Defaults to (0.01, 0.5)
        eps_lim : tuple[float, float], optional
            Limits for noise parameter. Defaults to (1e-4, 0.1)
        delay_lim : tuple[float, float], optional
            Limits for delay parameter. Defaults to (-5.0, 5.0)
        rng : np.random.Generator, optional
            Random number generator

        Returns
        -------
        GPParams
            Randomly generated GP parameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        gamma = rng.uniform(gamma_lim[0], gamma_lim[1], size=x_dim)
        eps = rng.uniform(eps_lim[0], eps_lim[1], size=x_dim)
        D = rng.uniform(delay_lim[0], delay_lim[1], size=(num_groups, x_dim))
        D[0, :] = 0
        return cls(gamma=gamma, eps=eps, D=D)
