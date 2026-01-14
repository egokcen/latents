"""Fit tracking and serialization for GFA models."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from latents.gfa.config import GFAFitConfig
from latents.observation import (
    ARDPosterior,
    LoadingPosterior,
    ObsMeanPosterior,
    ObsParamsHyperPrior,
    ObsParamsPosterior,
    ObsPrecPosterior,
)
from latents.state import LatentsPosteriorStatic
from latents.tracking import FitFlags, FitTracker

# -----------------------------------------------------------------------------
# Tracker and Flags
# -----------------------------------------------------------------------------


class GFAFitTracker(FitTracker):
    """Quantities tracked during a GFA model fit.

    Attributes
    ----------
    lb : ndarray of float, shape (num_iter,)
        Variational lower bound at each iteration.
    iter_time : ndarray of float, shape (num_iter,)
        Runtime on each iteration.
    lb_base : float or None
        Baseline lower bound for convergence checking.
    """

    pass


@dataclass
class GFAFitFlags(FitFlags):
    """Status flags from a GFA model fit.

    Attributes
    ----------
    converged
        True if the lower bound converged before reaching max_iter.
    decreasing_lb
        True if lower bound decreased during fitting.
    private_var_floor
        True if the private variance floor was used on any values of phi.
    x_dims_removed
        Number of latent dimensions removed due to low variance.
    """

    x_dims_removed: int = 0

    def display(self) -> None:
        """Print the fit flags."""
        super().display()
        print(f"Latent dimensions removed: {self.x_dims_removed}")


# -----------------------------------------------------------------------------
# Fit Context
# -----------------------------------------------------------------------------


@dataclass
class GFAFitContext:
    """Context passed to callbacks during GFA fitting.

    Provides read-only access to fitting state and a save() method for
    checkpointing. Checkpoints can be loaded via ``GFAModel.load(path)``.

    Attributes
    ----------
    config
        Fitting configuration.
    obs_hyperprior
        Prior hyperparameters.
    obs_posterior
        Observation model posterior.
    latents_posterior
        Latent variable posterior.
    tracker
        Fitting progress tracker.
    flags
        Fitting status flags.
    """

    config: GFAFitConfig
    obs_hyperprior: ObsParamsHyperPrior
    obs_posterior: ObsParamsPosterior
    latents_posterior: LatentsPosteriorStatic
    tracker: GFAFitTracker
    flags: GFAFitFlags

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save current state to a checkpoint file.

        The checkpoint can be loaded as a GFAModel via ``GFAModel.load(path)``.

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


# -----------------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------------


def save_gfa_state(
    path: str | os.PathLike[str],
    config: GFAFitConfig,
    obs_hyperprior: ObsParamsHyperPrior,
    obs_posterior: ObsParamsPosterior | None = None,
    latents_posterior: LatentsPosteriorStatic | None = None,
    tracker: GFAFitTracker | None = None,
    flags: GFAFitFlags | None = None,
) -> None:
    """Save GFA model state to a safetensors file.

    Uses safetensors format for secure serialization (no arbitrary code
    execution on load). Arrays are stored as tensors; scalars and config
    are stored as JSON in metadata.

    Parameters
    ----------
    path
        Output file path (conventionally ends in .safetensors).
    config
        Fitting configuration.
    obs_hyperprior
        Prior hyperparameters.
    obs_posterior
        Observation model posterior (None if unfitted).
    latents_posterior
        Latent variable posterior (None if not saved).
    tracker
        Fitting progress tracker (None if not tracked).
    flags
        Fitting status flags (None if unfitted).
    """
    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {}

    # Config and hyperprior (frozen dataclasses -> JSON)
    metadata["config"] = json.dumps(asdict(config))
    metadata["obs_hyperprior"] = json.dumps(asdict(obs_hyperprior))

    # Observation posterior
    if obs_posterior is not None:
        obs = obs_posterior
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
    if latents_posterior is not None:
        lat = latents_posterior
        if lat.mean is not None:
            tensors["latents_posterior.mean"] = lat.mean
        if lat.cov is not None:
            tensors["latents_posterior.cov"] = lat.cov
        if lat.moment is not None:
            tensors["latents_posterior.moment"] = lat.moment

    # Tracker
    if tracker is not None:
        if tracker.lb is not None:
            tensors["tracker.lb"] = tracker.lb
        if tracker.iter_time is not None:
            tensors["tracker.iter_time"] = tracker.iter_time
        # lb_base is a scalar, so it goes in metadata
        if tracker.lb_base is not None:
            metadata["tracker.lb_base"] = str(tracker.lb_base)

    # Flags
    if flags is not None:
        metadata["flags"] = json.dumps(asdict(flags))

    save_file(tensors, path, metadata=metadata)


def load_gfa_state(
    path: str | os.PathLike[str],
) -> tuple[
    GFAFitConfig,
    ObsParamsHyperPrior,
    ObsParamsPosterior | None,
    LatentsPosteriorStatic | None,
    GFAFitTracker | None,
    GFAFitFlags | None,
]:
    """Load GFA model state from a safetensors file.

    Parameters
    ----------
    path
        Path to .safetensors file.

    Returns
    -------
    config
        Fitting configuration.
    obs_hyperprior
        Prior hyperparameters.
    obs_posterior
        Observation model posterior, or None if not present.
    latents_posterior
        Latent variable posterior, or None if not present.
    tracker
        Fitting progress tracker, or None if not present.
    flags
        Fitting status flags, or None if not present.
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118

    # Config and hyperprior (always present)
    config = GFAFitConfig(**json.loads(metadata["config"]))
    obs_hyperprior = ObsParamsHyperPrior(**json.loads(metadata["obs_hyperprior"]))

    # Observation posterior
    obs_posterior = None
    if "obs_posterior.x_dim" in metadata:
        obs_posterior = ObsParamsPosterior(
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

    # Latents posterior
    latents_posterior = None
    if "latents_posterior.mean" in tensors:
        latents_posterior = LatentsPosteriorStatic(
            mean=tensors.get("latents_posterior.mean"),
            cov=tensors.get("latents_posterior.cov"),
            moment=tensors.get("latents_posterior.moment"),
        )

    # Tracker
    tracker = None
    if "tracker.lb" in tensors or "tracker.lb_base" in metadata:
        tracker = GFAFitTracker(
            lb=tensors.get("tracker.lb"),
            iter_time=tensors.get("tracker.iter_time"),
            lb_base=(
                float(metadata["tracker.lb_base"])
                if "tracker.lb_base" in metadata
                else None
            ),
        )

    # Flags
    flags = None
    if "flags" in metadata:
        flags = GFAFitFlags(**json.loads(metadata["flags"]))

    return config, obs_hyperprior, obs_posterior, latents_posterior, tracker, flags
