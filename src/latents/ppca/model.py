"""pPCAModel class for probabilistic Principal Component Analysis."""

from __future__ import annotations

import os
from typing import Self

from latents.data import ObsStatic
from latents.observation import ObsParamsPoint
from latents.ppca.config import pPCAFitConfig
from latents.ppca.inference import fit, infer_latents, infer_loadings, init_posteriors
from latents.ppca.tracking import (
    load_ppca_state,
    pPCAFitFlags,
    pPCAFitTracker,
    save_ppca_state,
)
from latents.state import LatentsPosteriorStatic


class pPCAModel:
    """High-level interface for pPCA.

    Parameters
    ----------
    config : pPCAFitConfig or None, default None
        Fitting configuration. If None, uses default pPCAFitConfig()

    Attributes
    ----------
    config : pPCAFitConfig or None
        Fitting configuration (immutable after construction).
    obs_params : ObsParamsPoint or None
        Point estimates of observation model parameters
    latents_posterior : LatentsPosteriorStatic or None
        Posterior over latent variables. None until fit.
    tracker : pPCAFitTracker or None
        Fitting progress tracker. None until fit.
    flags : pPCAFitFlags or None
        Fitting status flags. None until fit.

    Examples
    --------
    **Basic usage: fit and infer**

    >>> from latents.ppca import pPCAModel, pPCAFitConfig
    >>> from latents.callbacks import ProgressCallback
    >>> config = pPCAFitConfig(x_dim=10)
    >>> model = pPCAModel(config=config)
    >>> model.fit(Y, callbacks=[ProgressCallback()])
    >>> X_new = model.infer_latents(Y_new)

    **Persistence: save and load**

    >>> model.save("fitted_model.safetensors")
    >>> loaded = pPCAModel.load("fitted_model.safetensors")
    """

    def __init__(
        self,
        config: pPCAFitConfig | None = None,
        obs_params: ObsParamsPoint | None = None,
    ):
        # Configuration (immutable)
        self.config = config or pPCAFitConfig()

        # Posterior estimates (populated by fit)
        if obs_params is not None:
            self.obs_params = obs_params
        else:
            self.obs_params = None

        self.latents_posterior: LatentsPosteriorStatic | None = None

        # Fitting state (populated by fit)
        self.tracker: pPCAFitTracker | None = None
        self.flags: pPCAFitFlags | None = None

    def __repr__(self) -> str:
        fitted = self.obs_params is not None
        return f"pPCAModel(fitted={fitted}, config={self.config})"

    def fit(self, Y: ObsStatic, callbacks: list | None = None) -> Self:
        """Fit model to data via Expectation-Maximization.

        Resets tracker and flags. Warm-starts from posteriors if present,
        otherwise initializes from scratch.

        Parameters
        ----------
        Y : ObsStatic
            Observed data.
        callbacks : list of Callback or None, default None
            List of callback objects for progres, logging, checkpointing, etc.
            See :mod:`~latents.callbacks` module.

        Returns
        -------
        Self
            The fitted model (for method chaining).

        Examples
        --------
        >>> from latents.ppca import pPCAModel, pPCAFitConfig
        >>> from latents.callbacks import ProgressCallback
        >>> config = pPCAFitConfig(x_dim=10, max_iter=100)
        >>> model = pPCAModel(config=config)
        >>> model.fit(Y, callbacks=[ProgressCallback()])
        """
        # Initialize obs_params if not present (cold start)
        if self.obs_params is None:
            self._init_posteriors(Y)

        # Fit (resets tracker/flags for fresh convergence tracking)
        self.obs_params, self.latents_posterior, self.tracker, self.flags = fit(
            Y,
            self.obs_params,
            self.latents_posterior,
            config=self.config,
            callbacks=callbacks,
        )
        return self

    def resume_fit(
        self,
        Y: ObsStatic,
        max_iter: int | None = None,
        callbacks: list | None = None,
    ) -> Self:
        """Resue an interrupted fit.

        Appends to tracker, preserves convergence baseline.

        Parameters
        ----------
        Y : ObsStatic
          Observed data.
        max_iter : int or None, default None
            Maximum iterations for this resume run. If None, uses config.max_iter.
        callbacks : list of Callback or None, default none
            List of callback objects for progress, logging, checkpointing, etc

        Returns
        -------
        Self
            The model (for method chaining).

        Raises
        ------
        ValueError
            if no fit to resume (params not initialized)
        """
        if self.obs_params is None:
            msg = "No fit to resume. Use fit() instead."
            raise ValueError(msg)
        if self.tracker is None or self.flags is None:
            msg = "No tracking state to resume. Use fit() instead."
            raise ValueError(msg)
        if self.flags.converged:
            # Already converged, nothing to do
            return self

        self.obs_params, self.latents_posterior, self.tracker, self.flags = fit(
            Y,
            self.obs_params,
            self.latents_posterior,
            config=self.config,
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
        self.obs_params = None
        self.latents_posterior = None
        self.tracker = None
        self.flags = None
        return self

    def infer_latents(self, Y: ObsStatic) -> LatentsPosteriorStatic:
        """Infer latent posterior for new data given fitted parameters.

        Does not modify the model's stored latents_posterior.

        Parameters
        ----------
        Y : ObsStatic
            Observed data.

        Returns
        -------
        LatentsPosteriorStatic
            Posterior over latent variables for the given data.

        Raises
        ------
        ValueError
            If model has not been fitted.

        Examples
        --------
        >>> model.fit(Y_train)
        >>> X_test = model.infer_latents(Y_test)
        >>> X_test.mean.shape
        (5, 50)
        """
        if self.obs_params is None:
            msg = "Model must be fitted before inferring latents."
            raise ValueError(msg)
        return infer_latents(Y, self.obs_params)

    def recompute_latents(self, Y: ObsStatic) -> Self:
        """Recompute latents from data. Updates self.latents_posterior.

        Use this to restore latents after loading a model saved with
        save_x=False.

        Parameters
        ----------
        Y : ObsStatic
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
        if self.obs_params is None:
            msg = "Model must be fitted before recomputing latents."
            raise ValueError(msg)
        if self.latents_posterior is None:
            self.latents_posterior = LatentsPosteriorStatic()
        infer_latents(Y, self.obs_params, self.latents_posterior)
        return self

    def recompute_loadings(self, Y: ObsStatic) -> Self:
        """Recompute loading posterior from data. Updates self.obs_params.C.

        Use this to restore C.cov after loading a model saved with
        save_c_cov=False.

        Note: At convergence, the recomputed values are essentially identical
        to the original fitted values. Pre-convergence, there may be non-negligible
        differences because the reconstruction uses the final latents posterior rather
        than the latents posterior from the previous iteration.

        Parameters
        ----------
        Y : ObsStatic
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
        if self.obs_params is None:
            msg = "Model must be fitted before recomputing loadings."
            raise ValueError(msg)
        if (
            self.latents_posterior is None
            or not self.latents_posterior.is_initialized()
        ):
            msg = "Latents must be available. Call recompute_latents(Y) first."
            raise ValueError(msg)
        infer_loadings(Y, self.obs_params, self.latents_posterior)
        return self

    def _init_posteriors(self, Y: ObsStatic) -> None:
        """Initialize posteriors from data."""
        self.obs_params, self.latents_posterior = init_posteriors(
            Y, config=self.config, obs_params=self.obs_params
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save model to a safetensors file.

        Uses safetensors format for secure serialization (no arbitrary code
        execution on load). Arrays are stored as tensors; scalars and config
        are stored as JSON in metadata.

        Parameters
        ----------
        path : str or PathLike
            Output file path (conventionally ends in .safetensors).
        """
        save_ppca_state(
            path,
            config=self.config,
            obs_params=self.obs_params,
            latents_posterior=self.latents_posterior,
            tracker=self.tracker,
            flags=self.flags,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> pPCAModel:
        """Load model from a safetensors file.

        Parameters
        ----------
        path : str or PathLike
            Path to .safetensors file.

        Returns
        -------
        pPCAModel
            Loaded model, ready for inference or continued fitting.
        """
        config, obs_params, latents_posterior, tracker, flags = load_ppca_state(path)

        # Create model with loaded config
        model = cls(config=config)

        # Restore posteriors and tracking state
        model.obs_params = obs_params
        model.latents_posterior = latents_posterior
        model.tracker = tracker
        model.flags = flags

        return model
