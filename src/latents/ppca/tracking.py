"""Fit tracking and serialization for pPCA models."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from safetensors import safe_open

from latents._internal.serialization import save_tensors
from latents.observation import ObsParamsPoint
from latents.ppca.config import pPCAFitConfig
from latents.state import LatentsPosteriorStatic
from latents.tracking import FitFlags, FitTracker

# ----------------------------------------------------------------
# Tracker and Flags
# ----------------------------------------------------------------


class pPCAFitTracker(FitTracker):
    """Quantities tracked during a pPCA model fit.

    Attributes
    ----------
    lb : ndarray of float, shape (num_iter,)
        log-likelihood lower bound at each iteration
    iter_time : ndarray of float, shape (num_iter)
        Runtime on each iteration.
    lb_base : float or None
        Baseline lower bound for convergence checking
    """

    pass


@dataclass
class pPCAFitFlags(FitFlags):
    """Status flags from a pPCA model fit.

    Parameters
    ----------
    converged : bool, default False
        True if lower bound converged before reaching max_iter
    decrasing_lb : bool, default False
        True if lower bound decreased during fitting
    private_var_floor : bool, default False
        Stays false, non-applicable to pPCA
    """

    def display(self) -> None:
        """Print the fit flags."""
        super().display()
        print("private_var_floor non-applicable for pPCA")


# ----------------------------------------------------------------
# Fit Context
# ----------------------------------------------------------------


@dataclass
class pPCAFitContext:
    """Context passed to callbacks during pPCA fitting.

    Provides read-only access to fitting state and a save() method for
    checkpointing. Checkpoints can be loaded via :meth:`pPCAModel.load`.

    Parameters
    ----------
    config : pPCAFitConfig
        Fitting configuration.
    obs_params : ObsParamsPoint
        Observation model parameters.
    latents_posterior : LatentsPosteriorStatic
        Latent variable posterior.
    tracker : pPCAFitTracker
        Fitting progress tracker.
    flags : pPCAFitFlags
        Fitting status flags.
    """

    config: pPCAFitConfig
    obs_params: ObsParamsPoint
    latents_posterior: LatentsPosteriorStatic
    tracker: pPCAFitTracker
    flags: pPCAFitFlags

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save current state to a checkpoint file.

        The checkpoint can be loaded as a :class:`pPCAModel` via
        :meth:`pPCAModel.load`.

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


# -----------------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------------


def save_ppca_state(
    path: str | os.PathLike[str],
    config: pPCAFitConfig,
    obs_params: ObsParamsPoint | None = None,
    latents_posterior: LatentsPosteriorStatic | None = None,
    tracker: pPCAFitTracker | None = None,
    flags: pPCAFitFlags | None = None,
) -> None:
    """Save pPCA model state to a safetensors file.

    Uses safetensors format for secure serialization (no arbitrary code
    execution on load). Arrays are stored as tensors; scalars and config
    are stored as JSON in metadata.

    Parameters
    ----------
    path : str or PathLike
        Output file path (conventionally ends in .safetensors).
    config : pPCAFitConfig
        Fitting configuration.
    obs_params : ObsParamsPoint or None, default None
        Observation model parameters (None if unfitted).
    latents_posterior : LatentsPosteriorStatic or None, default None
        Latent variable posterior (None if not saved).
    tracker : pPCAFitTracker or None, default None
        Fitting progress tracker (None if not tracked).
    flags : pPCAFitFlags or None, default None
        Fitting status flags (None if unfitted).
    """
    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {}

    # Config (frozen dataclass -> JSON)
    metadata["config"] = json.dumps(asdict(config))

    # Observation parameters (point estimates)
    if obs_params is not None:
        metadata["obs_params.x_dim"] = str(obs_params.x_dim)
        tensors["obs_params.y_dims"] = obs_params.y_dims
        tensors["obs_params.C"] = obs_params.C
        tensors["obs_params.d"] = obs_params.d
        # phi: (y_dim,) precisions, all equal for isotropic noise
        tensors["obs_params.phi"] = obs_params.phi

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

    save_tensors(tensors, path, metadata)


def load_ppca_state(
    path: str | os.PathLike[str],
) -> tuple[
    pPCAFitConfig,
    ObsParamsPoint | None,
    LatentsPosteriorStatic | None,
    pPCAFitTracker | None,
    pPCAFitFlags | None,
]:
    """Load pPCA model state from a safetensors file.

    Parameters
    ----------
    path : str or PathLike
        Path to .safetensors file.

    Returns
    -------
    config : pPCAFitConfig
        Fitting configuration.
    obs_params : ObsParamsPoint or None
        Observation model parameters, or None if not present.
    latents_posterior : LatentsPosteriorStatic or None
        Latent variable posterior, or None if not present.
    tracker : pPCAFitTracker or None
        Fitting progress tracker, or None if not present.
    flags : pPCAFitFlags or None
        Fitting status flags, or None if not present.
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118

    # Config (always present)
    config = pPCAFitConfig(**json.loads(metadata["config"]))

    # Observation parameters
    obs_params = None
    if "obs_params.x_dim" in metadata:
        obs_params = ObsParamsPoint(
            C=tensors["obs_params.C"],
            d=tensors["obs_params.d"],
            phi=tensors["obs_params.phi"],
            y_dims=tensors["obs_params.y_dims"],
            x_dim=int(metadata["obs_params.x_dim"]),
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
        tracker = pPCAFitTracker(
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
        flags = pPCAFitFlags(**json.loads(metadata["flags"]))

    return config, obs_params, latents_posterior, tracker, flags
