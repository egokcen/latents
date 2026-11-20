"""Simulate data from the probabilistic PCA generative model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from safetensors import safe_open

from latents._internal.serialization import save_tensors
from latents.data import ObsStatic
from latents.observation import (
    ObsParamsPoint,
    adjust_snr,
)
from latents.ppca.config import pPCASimConfig
from latents.state import LatentsPriorStatic, LatentsRealization


@dataclass
class pPCASimulationResult:
    """Complete output of a pPCA simulation run.

    Bundles the specifications with the sampled outputs
    (obs_params, latents, observations). This is a run artifact analogous
    to a fitted :class:`pPCAModel`.

    Parameters
    ----------
    config : pPCASimConfig
        Experimental design parameters used for simulation.
    obs_params : ObsParamsPoint
        Sampled observation model parameters (C, d, phi).
    latents : LatentsRealization
        Latent variables.
    observations : ObsStatic
        Generated observed data.
    """

    config: pPCASimConfig
    obs_params: ObsParamsPoint
    latents: LatentsRealization
    observations: ObsStatic


def simulate(
    config: pPCASimConfig,
    obs_params: ObsParamsPoint,
) -> pPCASimulationResult:
    """Generate samples from the full probabilistic PCA model.

    Parameters
    ----------
    config : pPCASimConfig
        Experimental design parameters (n_samples, y_dim, x_dim, snr, seed).
    obs_params : ObsParamsPoint
        Observation model parameters. ``phi`` is rescaled to achieve
        ``config.snr``; the input is not modified.

    Returns
    -------
    pPCASimulationResult
        Complete simulation output including config, parameters, latents, observations.

    Examples
    --------
    >>> config = pPCASimConfig(
    ...     n_samples=100,
    ...     y_dim=20,
    ...     x_dim=5,
    ...     snr=1.0,
    ...     random_seed=42,
    ... )
    >>> result = simulate(config, obs_params)
    >>> result.observations.data.shape
    (20, 100)
    """
    rng = np.random.default_rng(config.random_seed)
    snr = config.snr

    # adjust snr
    obs_params = adjust_snr(obs_params, snr)

    # Sample latent data from the static
    latents_prior = LatentsPriorStatic()
    latents = latents_prior.sample(config.x_dim, config.n_samples, rng)

    # Generate observed_data
    observations = sample_observations(latents, obs_params, rng)

    return pPCASimulationResult(
        config=config,
        obs_params=obs_params,
        latents=latents,
        observations=observations,
    )


def sample_observations(
    latents: LatentsRealization,
    obs_params: ObsParamsPoint,
    rng: np.random.Generator,
) -> ObsStatic:
    """Generate osberved data via the pPCA observation model.

    Parameters
    ----------
    latents : LatentsRealization
        Sampled latent data.
    obs_params : ObsParamsPoint
        pPCA observation model parameters
    rng : numpy.random.Generator
        Numpy random number generator.

    Returns
    -------
    ObsStatic
        Generated observed data.
    """
    # Number of data points
    n_samples = latents.n_samples
    # Dimensionality observed data
    y_dim = obs_params.y_dim

    # Initialize observed data list
    Y = ObsStatic(data=np.zeros((y_dim, n_samples)), dims=obs_params.y_dims.copy())

    C = obs_params.C
    data = latents.data
    d = obs_params.d
    phi = obs_params.phi
    noise = rng.normal(size=(y_dim, n_samples)) / np.sqrt(phi)[:, np.newaxis]

    # Generate observed data
    Y.data = C @ data + d[:, np.newaxis] + noise

    return Y


# --- Serialization helpers ---


def _serialize_config(
    config: pPCASimConfig,
    metadata: dict[str, str],
) -> None:
    """Serialize pPCASimConfig to metadata dicts (in-place)."""
    # metadata
    metadata["config.n_samples"] = str(config.n_samples)
    metadata["config.x_dim"] = str(config.x_dim)
    metadata["config.y_dim"] = str(config.y_dim)
    metadata["config.snr"] = str(config.snr)

    if config.random_seed is not None:
        metadata["config.random_seed"] = json.dumps(config.random_seed)


def _deserialize_config(metadata: dict[str, str]) -> pPCASimConfig:
    """Deserialize pPCASimConfig from tensors and metadata dicts."""
    return pPCASimConfig(
        n_samples=int(metadata["config.n_samples"]),
        y_dim=int(metadata["config.y_dim"]),
        x_dim=int(metadata["config.x_dim"]),
        snr=float(metadata["config.snr"]),
        random_seed=(
            json.loads(metadata["config.random_seed"])
            if "config.random_seed" in metadata
            else None
        ),
    )


# --- Save/Load functions ---


def save_simulation(path: str | os.PathLike[str], result: pPCASimulationResult) -> None:
    """Save complete simulation result to safetensors file.

    Saves the full snapshot: config, obs_params, latents, and
    observations. Use :func:`~latents.ppca.simulation.load_simulation` to restore.

    Parameters
    ----------
    path : str or PathLike
        Output file path (conventionally ends in .safetensors).
    result : pPCASimulationResult
        Complete simulation result to save

    See Also
    --------
    save_simulation_recipe : Save only config (smaller file)
    """
    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {"file_type": "snapshot"}

    # Config
    _serialize_config(result.config, metadata)

    # obs_params
    tensors["obs_params.C"] = result.obs_params.C
    tensors["obs_params.d"] = result.obs_params.d
    tensors["obs_params.phi"] = result.obs_params.phi
    tensors["obs_params.y_dim"] = result.obs_params.y_dims
    metadata["obs_params.x_dim"] = str(result.obs_params.x_dim)

    # latents
    tensors["latents.data"] = result.latents.data

    # observations
    tensors["observations.data"] = result.observations.data
    tensors["observations.dims"] = result.observations.dims

    save_tensors(tensors, path, metadata)


def save_simulation_recipe(
    path: str | os.PathLike[str],
    config: pPCASimConfig,
) -> None:
    """Save simulation recipe (config).

    Saves a minimal file that can regenerate the full simulation when
    passed to "func"`~latents.ppca.simulation.simulate`. Requires ``config.random_seed``
    to be set.

    Parameters
    ----------
    path : str or PathLike
        Output file path (conventionally ends in .safetensors).
    config : pPCASimConfig
        Simulation configuration. Must have random_seed set.

    Raises
    ------
    ValueError
        If config.random_seed is None.

    See Also
    --------
    save_simulation : Save complete results (larger file).
    """
    if config.random_seed is None:
        msg = "random_seed required for reproducible recipe"
        raise ValueError(msg)

    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {"file_type": "recipe"}

    _serialize_config(config, metadata)

    save_tensors(tensors, path, metadata)


def load_simulation(path: str | os.PathLike[str]) -> pPCASimulationResult:
    """Load complete simulation from file.

    Parameters
    ----------
    path : str or PathLike
        Path to .safetensors file saved with
        :func:`~latents.ppca.simulation.save_simulation`.

    Returns
    -------
    pPCASimulationResult
        Complete simulation result.

    Raises
    ------
    ValueError
        If file contains only a recipe. Use
        :func:`~latents.ppca.simulation.load_simulation_recipe` and
        :func:`~latents.ppca.simulation.simulate` to regenerate.

    See Also
    --------
    load_simulation_recipe : Load recipe and regenerate via simulate().
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118

    if metadata.get("file_type") == "recipe":
        msg = (
            "File contains only a recipe (config). "
            "Use load_simulation_recipe() and simulate() to regenerate."
        )
        raise ValueError(msg)

    config = _deserialize_config(metadata)

    obs_params = ObsParamsPoint(
        C=tensors["obs_params.C"],
        d=tensors["obs_params.d"],
        phi=tensors["obs_params.phi"],
        y_dims=tensors["obs_params.y_dim"],
        x_dim=int(metadata["obs_params.x_dim"]),
    )

    latents = LatentsRealization(data=tensors["latents.data"])

    observations = ObsStatic(
        data=tensors["observations.data"],
        dims=tensors["observations.dims"],
    )

    return pPCASimulationResult(
        config=config,
        obs_params=obs_params,
        latents=latents,
        observations=observations,
    )


def load_simulation_recipe(
    path: str | os.PathLike[str],
) -> pPCASimConfig:
    """Load simulation recipe from file.

    Works for both recipe-only files and full snapshots (extracts just
    the specification).

    Parameters
    ----------
    path : str or PathLike
        Path to .safetensors file

    Returns
    -------
    config : pPCASimConfig
        Simulation configuration

    Examples
    --------
    >>> config = load_simulation_recipe("simulation.safetensors")
    >>> result = simulate(config) # Regenerate from recipe
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()

    return _deserialize_config(metadata)
