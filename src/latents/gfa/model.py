"""GFAModel class for Group Factor Analysis."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

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
    infer_loadings,
    init_posteriors,
)
from latents.observation import (
    ARDPosterior,
    LoadingPosterior,
    ObsMeanPosterior,
    ObsParamsHyperPrior,
    ObsParamsPosterior,
    ObsParamsPrior,
    ObsPrecPosterior,
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
    >>> config = GFAFitConfig(x_dim_init=10, verbose=True)
    >>> model = GFAModel(config=config)
    >>> model.fit(Y)
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

    def save(self, path: str) -> None:
        """Save model to a safetensors file.

        Uses safetensors format for secure serialization (no arbitrary code
        execution on load). Arrays are stored as tensors; scalars and config
        are stored as JSON in metadata.

        Parameters
        ----------
        path
            Output file path (conventionally ends in .safetensors).
        """
        tensors: dict[str, np.ndarray] = {}
        metadata: dict[str, str] = {}

        # Config and hyperprior (frozen dataclasses -> JSON)
        metadata["config"] = json.dumps(asdict(self.config))
        metadata["obs_hyperprior"] = json.dumps(asdict(self.obs_hyperprior))

        # Observation posterior
        if self.obs_posterior is not None:
            obs = self.obs_posterior
            metadata["obs_posterior.x_dim"] = str(obs.x_dim)
            tensors["obs_posterior.y_dims"] = obs.y_dims

            # Loading posterior (C)
            if obs.C.mean is not None:
                tensors["obs_posterior.C.mean"] = obs.C.mean
            if obs.C.cov is not None:
                tensors["obs_posterior.C.cov"] = obs.C.cov
            if obs.C.moment is not None:
                tensors["obs_posterior.C.moment"] = obs.C.moment

            # ARD posterior (alpha)
            if obs.alpha.a is not None:
                tensors["obs_posterior.alpha.a"] = obs.alpha.a
            if obs.alpha.b is not None:
                tensors["obs_posterior.alpha.b"] = obs.alpha.b
            if obs.alpha.mean is not None:
                tensors["obs_posterior.alpha.mean"] = obs.alpha.mean

            # Observation mean posterior (d)
            if obs.d.mean is not None:
                tensors["obs_posterior.d.mean"] = obs.d.mean
            if obs.d.cov is not None:
                tensors["obs_posterior.d.cov"] = obs.d.cov

            # Observation precision posterior (phi)
            # phi.a is a scalar, so it goes in metadata
            if obs.phi.a is not None:
                metadata["obs_posterior.phi.a"] = str(obs.phi.a)
            if obs.phi.b is not None:
                tensors["obs_posterior.phi.b"] = obs.phi.b
            if obs.phi.mean is not None:
                tensors["obs_posterior.phi.mean"] = obs.phi.mean

        # Latents posterior
        if self.latents_posterior is not None:
            lat = self.latents_posterior
            if lat.mean is not None:
                tensors["latents_posterior.mean"] = lat.mean
            if lat.cov is not None:
                tensors["latents_posterior.cov"] = lat.cov
            if lat.moment is not None:
                tensors["latents_posterior.moment"] = lat.moment

        # Tracker
        if self.tracker is not None:
            if self.tracker.lb is not None:
                tensors["tracker.lb"] = self.tracker.lb
            if self.tracker.iter_time is not None:
                tensors["tracker.iter_time"] = self.tracker.iter_time
            # lb_base is a scalar, so it goes in metadata
            if self.tracker.lb_base is not None:
                metadata["tracker.lb_base"] = str(self.tracker.lb_base)

        # Flags
        if self.flags is not None:
            metadata["flags"] = json.dumps(asdict(self.flags))

        save_file(tensors, path, metadata=metadata)

    @classmethod
    def load(cls, path: str) -> GFAModel:
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
        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata()
            tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118

        # Reconstruct config and hyperprior
        config = GFAFitConfig(**json.loads(metadata["config"]))
        obs_hyperprior = ObsParamsHyperPrior(**json.loads(metadata["obs_hyperprior"]))

        # Create model
        model = cls(config=config, obs_hyperprior=obs_hyperprior)

        # Reconstruct observation posterior
        if "obs_posterior.x_dim" in metadata:
            model.obs_posterior = ObsParamsPosterior(
                x_dim=int(metadata["obs_posterior.x_dim"]),
                y_dims=tensors["obs_posterior.y_dims"],
                C=LoadingPosterior(
                    mean=tensors.get("obs_posterior.C.mean"),
                    cov=tensors.get("obs_posterior.C.cov"),
                    moment=tensors.get("obs_posterior.C.moment"),
                ),
                alpha=ARDPosterior(
                    a=tensors.get("obs_posterior.alpha.a"),
                    b=tensors.get("obs_posterior.alpha.b"),
                    mean=tensors.get("obs_posterior.alpha.mean"),
                ),
                d=ObsMeanPosterior(
                    mean=tensors.get("obs_posterior.d.mean"),
                    cov=tensors.get("obs_posterior.d.cov"),
                ),
                phi=ObsPrecPosterior(
                    a=(
                        float(metadata["obs_posterior.phi.a"])
                        if "obs_posterior.phi.a" in metadata
                        else None
                    ),
                    b=tensors.get("obs_posterior.phi.b"),
                    mean=tensors.get("obs_posterior.phi.mean"),
                ),
            )

        # Reconstruct latents posterior
        if "latents_posterior.mean" in tensors:
            model.latents_posterior = LatentsPosteriorStatic(
                mean=tensors.get("latents_posterior.mean"),
                cov=tensors.get("latents_posterior.cov"),
                moment=tensors.get("latents_posterior.moment"),
            )

        # Reconstruct tracker
        if "tracker.lb" in tensors or "tracker.lb_base" in metadata:
            model.tracker = GFAFitTracker(
                lb=tensors.get("tracker.lb"),
                iter_time=tensors.get("tracker.iter_time"),
                lb_base=(
                    float(metadata["tracker.lb_base"])
                    if "tracker.lb_base" in metadata
                    else None
                ),
            )

        # Reconstruct flags
        if "flags" in metadata:
            model.flags = GFAFitFlags(**json.loads(metadata["flags"]))

        return model
