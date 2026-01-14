"""GFAModel class for Group Factor Analysis."""

from __future__ import annotations

import os
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.gfa.inference import fit, infer_latents, infer_loadings, init_posteriors
from latents.gfa.tracking import (
    GFAFitFlags,
    GFAFitTracker,
    load_gfa_state,
    save_gfa_state,
)
from latents.observation import (
    ObsParamsHyperPrior,
    ObsParamsPosterior,
    ObsParamsPrior,
)
from latents.state import LatentsPosteriorStatic, LatentsPriorStatic


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
    >>> from latents.callbacks import ProgressCallback
    >>> config = GFAFitConfig(x_dim_init=10)
    >>> model = GFAModel(config=config)
    >>> model.fit(Y, callbacks=[ProgressCallback()])
    >>> X_new = model.infer_latents(Y_new)
    >>> model.save("fitted_model.safetensors")
    >>> loaded = GFAModel.load("fitted_model.safetensors")
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

    def fit(self, Y: ObsStatic, callbacks: list | None = None) -> Self:
        """Fit model to data via variational inference.

        Resets tracker and flags. Warm-starts from posteriors if present,
        otherwise initializes from scratch.

        Parameters
        ----------
        Y
            Observed data.
        callbacks
            List of callback objects for progress, logging, checkpointing, etc.
            See latents.callbacks module.

        Returns
        -------
        Self
            The fitted model (for method chaining).
        """
        # Initialize posteriors if not present (cold start)
        if self.obs_posterior is None:
            self._init_posteriors(Y)

        # Fit (resets tracker/flags for fresh convergence tracking)
        self.obs_posterior, self.latents_posterior, self.tracker, self.flags = fit(
            Y,
            self.obs_posterior,
            self.latents_posterior,
            config=self.config,
            obs_hyperprior=self.obs_hyperprior,
            callbacks=callbacks,
        )
        return self

    def resume_fit(
        self,
        Y: ObsStatic,
        max_iter: int | None = None,
        callbacks: list | None = None,
    ) -> Self:
        """Resume an interrupted fit.

        Appends to tracker, preserves convergence baseline.

        Parameters
        ----------
        Y
            Observed data.
        max_iter
            Maximum iterations for this resume run. If None, uses config.max_iter.
        callbacks
            List of callback objects for progress, logging, checkpointing, etc.

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
            callbacks=callbacks,
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

    def recompute_latents(self, Y: ObsStatic) -> Self:
        """Recompute latents from data. Updates self.latents_posterior.

        Use this to restore latents after loading a model saved with
        save_x=False.

        Parameters
        ----------
        Y
            Observed data (typically the training data).

        Returns
        -------
        Self
            The model (for method chaining).

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self.obs_posterior is None:
            msg = "Model must be fitted before recomputing latents."
            raise ValueError(msg)
        if self.latents_posterior is None:
            self.latents_posterior = LatentsPosteriorStatic()
        infer_latents(Y, self.obs_posterior, self.latents_posterior)
        return self

    def recompute_loadings(self, Y: ObsStatic) -> Self:
        """Recompute loading posterior from data. Updates self.obs_posterior.C.

        Use this to restore C.cov after loading a model saved with
        save_c_cov=False.

        Note: At convergence, the recomputed values are essentially identical
        to the original fitted values. Pre-convergence, there may be non-negligible
        differences because the reconstruction uses the final latents posterior rather
        than the latents posterior from the previous iteration.

        Parameters
        ----------
        Y
            Observed data (typically the training data).

        Returns
        -------
        Self
            The model (for method chaining).

        Raises
        ------
        ValueError
            If model has not been fitted, or if latents are not available.
        """
        if self.obs_posterior is None:
            msg = "Model must be fitted before recomputing loadings."
            raise ValueError(msg)
        if (
            self.latents_posterior is None
            or not self.latents_posterior.is_initialized()
        ):
            msg = "Latents must be available. Call recompute_latents(Y) first."
            raise ValueError(msg)
        infer_loadings(Y, self.obs_posterior, self.latents_posterior)
        return self

    def _init_posteriors(self, Y: ObsStatic) -> None:
        """Initialize posteriors from data."""
        self.obs_posterior, self.latents_posterior = init_posteriors(
            Y, config=self.config, obs_hyperprior=self.obs_hyperprior
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save model to a safetensors file.

        Uses safetensors format for secure serialization (no arbitrary code
        execution on load). Arrays are stored as tensors; scalars and config
        are stored as JSON in metadata.

        Parameters
        ----------
        path
            Output file path (conventionally ends in .safetensors).
        """
        save_gfa_state(
            path,
            config=self.config,
            obs_hyperprior=self.obs_hyperprior,
            obs_posterior=self.obs_posterior,
            latents_posterior=self.latents_posterior,
            tracker=self.tracker,
            flags=self.flags,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> GFAModel:
        """Load model from a safetensors file.

        Parameters
        ----------
        path
            Path to .safetensors file.

        Returns
        -------
        GFAModel
            Loaded model, ready for inference or continued fitting.
        """
        config, obs_hyperprior, obs_posterior, latents_posterior, tracker, flags = (
            load_gfa_state(path)
        )

        # Create model with loaded config and hyperprior
        model = cls(config=config, obs_hyperprior=obs_hyperprior)

        # Restore posteriors and tracking state
        model.obs_posterior = obs_posterior
        model.latents_posterior = latents_posterior
        model.tracker = tracker
        model.flags = flags

        return model
