"""GFAModel class for Group Factor Analysis."""

from __future__ import annotations

import sys

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.gfa.inference import (
    GFAFitFlags,
    GFAFitTracker,
    fit,
    infer_latents,
    init_posteriors,
)
from latents.observation import (
    ObsParamsHyperPrior,
    ObsParamsPosterior,
    ObsParamsPrior,
)
from latents.state import LatentsPosteriorStatic, LatentsPriorStatic

jsonpickle_numpy.register_handlers()


class GFAModel:
    """High-level interface for Group Factor Analysis.

    Parameters
    ----------
    config
        Fitting configuration. If None, uses default GFAFitConfig().
    obs_hyperprior
        Prior hyperparameters for observation model. If None, uses default.

    Attributes
    ----------
    config : GFAFitConfig
        Fitting configuration (immutable after construction).
    obs_posterior : ObsParamsPosterior | None
        Posterior over observation model parameters. None until fit.
    latents_posterior : LatentsPosteriorStatic | None
        Posterior over latent variables. None until fit.
    tracker : GFAFitTracker | None
        Fitting progress tracker. None until fit.
    flags : GFAFitFlags | None
        Fitting status flags. None until fit.

    Examples
    --------
    >>> from latents.gfa import GFAModel, GFAFitConfig
    >>> config = GFAFitConfig(x_dim_init=10, verbose=True)
    >>> model = GFAModel(config=config)
    >>> model.fit(Y)
    >>> X_new = model.infer_latents(Y_new)
    """

    def __init__(
        self,
        config: GFAFitConfig | None = None,
        obs_hyperprior: ObsParamsHyperPrior | None = None,
    ):
        # Configuration (immutable)
        self.config = config or GFAFitConfig()

        # Prior specification (private, fixed at construction)
        self._obs_prior = ObsParamsPrior(
            hyperprior=obs_hyperprior or ObsParamsHyperPrior()
        )
        self._latents_prior = LatentsPriorStatic()

        # Posterior estimates (populated by fit)
        self.obs_posterior: ObsParamsPosterior | None = None
        self.latents_posterior: LatentsPosteriorStatic | None = None

        # Fitting state (populated by fit)
        self.tracker: GFAFitTracker | None = None
        self.flags: GFAFitFlags | None = None

    def __repr__(self) -> str:
        fitted = self.obs_posterior is not None
        return f"GFAModel(fitted={fitted}, config={self.config})"

    @property
    def obs_hyperprior(self) -> ObsParamsHyperPrior:
        """Observation model hyperprior parameters."""
        return self._obs_prior.hyperprior

    @property
    def obs_prior(self) -> ObsParamsPrior:
        """Observation model prior."""
        return self._obs_prior

    @property
    def latents_prior(self) -> LatentsPriorStatic:
        """Latent variable prior."""
        return self._latents_prior

    def fit(self, Y: ObsStatic) -> Self:
        """Fit model to data via variational inference.

        Resets tracker and flags. Warm-starts from posteriors if present,
        otherwise initializes from scratch.

        Parameters
        ----------
        Y
            Observed data.

        Returns
        -------
        Self
            The fitted model (for method chaining).
        """
        # Initialize posteriors if not present (cold start)
        if self.obs_posterior is None:
            if self.config.verbose:
                print("Initializing posteriors...")
            self._init_posteriors(Y)

        # Fit (resets tracker/flags for fresh convergence tracking)
        self.obs_posterior, self.latents_posterior, self.tracker, self.flags = fit(
            Y,
            self.obs_posterior,
            self.latents_posterior,
            config=self.config,
            obs_hyperprior=self.obs_hyperprior,
        )
        return self

    def resume_fit(self, Y: ObsStatic, max_iter: int | None = None) -> Self:
        """Resume an interrupted fit.

        Appends to tracker, preserves convergence baseline.

        Parameters
        ----------
        Y
            Observed data.
        max_iter
            Maximum iterations for this resume run. If None, uses config.max_iter.

        Returns
        -------
        Self
            The model (for method chaining).

        Raises
        ------
        ValueError
            If no fit to resume (posteriors not initialized).
        """
        if self.obs_posterior is None:
            msg = "No fit to resume. Use fit() instead."
            raise ValueError(msg)
        if self.tracker is None or self.flags is None:
            msg = "No tracking state to resume. Use fit() instead."
            raise ValueError(msg)
        if self.flags.converged:
            # Already converged, nothing to do
            return self

        self.obs_posterior, self.latents_posterior, self.tracker, self.flags = fit(
            Y,
            self.obs_posterior,
            self.latents_posterior,
            config=self.config,
            obs_hyperprior=self.obs_hyperprior,
            tracker=self.tracker,
            flags=self.flags,
            max_iter=max_iter,
        )
        return self

    def clear_fit(self) -> Self:
        """Clear fit results for fresh initialization on next fit().

        Returns
        -------
        Self
            The model (for method chaining).
        """
        self.obs_posterior = None
        self.latents_posterior = None
        self.tracker = None
        self.flags = None
        return self

    def infer_latents(self, Y: ObsStatic) -> LatentsPosteriorStatic:
        """Infer latent posterior for new data given fitted parameters.

        Does not modify the model's stored latents_posterior.

        Parameters
        ----------
        Y
            Observed data.

        Returns
        -------
        LatentsPosteriorStatic
            Posterior over latent variables for the given data.

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self.obs_posterior is None:
            msg = "Model must be fitted before inferring latents."
            raise ValueError(msg)
        return infer_latents(Y, self.obs_posterior)

    def _init_posteriors(self, Y: ObsStatic) -> None:
        """Initialize posteriors from data."""
        self.obs_posterior, self.latents_posterior = init_posteriors(
            Y, config=self.config, obs_hyperprior=self.obs_hyperprior
        )

    def save(self, filename: str, indent: int = 2) -> None:
        """Save model to a JSON file.

        Uses jsonpickle for serialization.

        Parameters
        ----------
        filename
            Path to JSON file.
        indent
            Number of spaces to indent. Defaults to 2.
        """
        with open(filename, "w") as f:
            f.write(jsonpickle.encode(self, indent=indent))

    @staticmethod
    def load(filename: str) -> GFAModel:
        """Load model from a JSON file.

        Uses jsonpickle for deserialization. Only load files from trusted
        sources.

        Parameters
        ----------
        filename
            Path to JSON file.

        Returns
        -------
        GFAModel
            Loaded model.
        """
        with open(filename) as f:
            return jsonpickle.decode(f.read())
