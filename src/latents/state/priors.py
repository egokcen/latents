"""Prior distributions for state model parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from latents.state.realizations import LatentsRealization


class LatentsPriorStatic:
    """Static latent prior: X ~ N(0, I).

    The standard GFA prior assumes independent standard normal latents.

    Examples
    --------
    >>> prior = LatentsPriorStatic()
    >>> rng = np.random.default_rng(42)
    >>> realization = prior.sample(x_dim=5, n_samples=100, rng=rng)
    >>> realization.X.shape
    (5, 100)
    """

    def sample(
        self,
        x_dim: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> LatentsRealization:
        """Sample X ~ N(0, I).

        Parameters
        ----------
        x_dim
            Number of latent dimensions.
        n_samples
            Number of samples to generate.
        rng
            Random number generator.

        Returns
        -------
        LatentsRealization
            Sampled latent values.
        """
        X = rng.normal(size=(x_dim, n_samples))
        return LatentsRealization(X=X)


@dataclass
class LatentsHyperPriorGP:
    """GP kernel hyperpriors.

    Stub for GPFA/mDLAG. GP hyperparameters are learnable.

    Parameters
    ----------
    kernel
        Kernel type (e.g., "rbf").
    timescale
        Characteristic timescale of the GP kernel.
    variance
        Signal variance of the GP kernel.
    """

    kernel: str = "rbf"
    timescale: float = 50.0
    variance: float = 1.0


@dataclass
class LatentsPriorGP:
    """GP latent prior.

    Stub for GPFA/mDLAG.

    Parameters
    ----------
    hyperprior
        GP kernel hyperpriors.
    """

    hyperprior: LatentsHyperPriorGP = field(default_factory=LatentsHyperPriorGP)

    def sample(
        self,
        x_dim: int,
        n_samples: int,
        n_timepoints: int,
        rng: np.random.Generator,
    ) -> LatentsRealization:
        """Sample from GP prior.

        Parameters
        ----------
        x_dim
            Number of latent dimensions.
        n_samples
            Number of samples (trials).
        n_timepoints
            Number of time points per sample.
        rng
            Random number generator.

        Returns
        -------
        LatentsRealization
            Sampled latent trajectories.

        Raises
        ------
        NotImplementedError
            GP prior sampling not yet implemented.
        """
        msg = "GP prior sampling not yet implemented"
        raise NotImplementedError(msg)
