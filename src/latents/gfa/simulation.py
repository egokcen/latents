"""Simulate data from the group factor analysis (GFA) generative model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from latents.data import ObsStatic
from latents.gfa.config import GFASimConfig
from latents.observation import (
    ObsParamsHyperPrior,
    ObsParamsHyperPriorStructured,
    ObsParamsPoint,
    ObsParamsPrior,
    ObsParamsRealization,
    adjust_snr,
)
from latents.state import LatentsPriorStatic, LatentsRealization


@dataclass
class GFASimulationResult:
    """Complete output of a GFA simulation run.

    Bundles the specification (config + hyperprior) with the sampled outputs
    (obs_params, latents, observations). This is a run artifact analogous to
    a fitted :class:`GFAModel`.

    Parameters
    ----------
    config : GFASimConfig
        Experimental design parameters used for simulation.
    hyperprior : ObsParamsHyperPrior or ObsParamsHyperPriorStructured
        Probabilistic model specification.
    obs_params : ObsParamsRealization
        Sampled observation model parameters (C, d, phi, alpha).
    latents : LatentsRealization
        Sampled latent variables.
    observations : ObsStatic
        Generated observed data.
    """

    config: GFASimConfig
    hyperprior: ObsParamsHyperPrior | ObsParamsHyperPriorStructured
    obs_params: ObsParamsRealization
    latents: LatentsRealization
    observations: ObsStatic


def simulate(
    config: GFASimConfig,
    hyperprior: ObsParamsHyperPrior | ObsParamsHyperPriorStructured,
) -> GFASimulationResult:
    """Generate samples from the full group factor analysis model.

    Parameters
    ----------
    config : GFASimConfig
        Experimental design parameters (n_samples, y_dims, x_dim, snr, seed).
    hyperprior : ObsParamsHyperPrior or ObsParamsHyperPriorStructured
        Probabilistic model specification. For structured sparsity patterns,
        use :class:`ObsParamsHyperPriorStructured` where ``a_alpha`` and ``b_alpha``
        arrays specify group- and column-specific patterns. Use ``np.inf`` in
        ``a_alpha`` to force zero loadings.

    Returns
    -------
    GFASimulationResult
        Complete simulation output including config, hyperprior, sampled
        parameters, latents, and observations.

    Examples
    --------
    >>> config = GFASimConfig(
    ...     n_samples=100,
    ...     y_dims=np.array([10, 10]),
    ...     x_dim=5,
    ...     random_seed=42,
    ... )
    >>> hyperprior = ObsParamsHyperPrior(
    ...     a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
    ... )
    >>> result = simulate(config, hyperprior)
    >>> result.observations.data.shape
    (20, 100)
    """
    rng = np.random.default_rng(config.random_seed)

    # Normalize snr to array
    snr = np.atleast_1d(config.snr)
    if snr.size == 1:
        snr = np.broadcast_to(snr, (config.n_groups,))

    # Sample from the prior and adjust SNR
    prior = ObsParamsPrior(hyperprior=hyperprior)
    obs_params = prior.sample(config.y_dims, config.x_dim, rng)
    obs_params = adjust_snr(obs_params, snr)

    # Sample latent data from the static prior
    latents_prior = LatentsPriorStatic()
    latents = latents_prior.sample(config.x_dim, config.n_samples, rng)

    # Generate observed data
    observations = sample_observations(latents, obs_params, rng)

    return GFASimulationResult(
        config=config,
        hyperprior=hyperprior,
        obs_params=obs_params,
        latents=latents,
        observations=observations,
    )


def sample_observations(
    latents: LatentsRealization,
    obs_params: ObsParamsRealization | ObsParamsPoint,
    rng: np.random.Generator,
) -> ObsStatic:
    """Generate observed data via the GFA observation model.

    Parameters
    ----------
    latents : LatentsRealization
        Sampled latent data.
    obs_params : ObsParamsRealization or ObsParamsPoint
        GFA observation model parameters, either as a full realization
        or as point estimates.
    rng : numpy.random.Generator
        NumPy random number generator.

    Returns
    -------
    ObsStatic
        Generated observed data.
    """
    # Number of data points
    n_samples = latents.n_samples
    # Dimensionality of each observed group
    y_dims = obs_params.y_dims
    # Number of observed groups
    n_groups = len(y_dims)

    # Split d, phi, and C according to observed groups
    y_boundaries = np.cumsum(y_dims)[:-1]
    ds = np.split(obs_params.d, y_boundaries)
    phis = np.split(obs_params.phi, y_boundaries)
    Cs = np.split(obs_params.C, y_boundaries, axis=0)

    # Initialize observed data list
    Y = ObsStatic(data=np.zeros((y_dims.sum(), n_samples)), dims=y_dims)
    Ys = Y.get_groups()

    # Generate observed data group by group
    for group_idx in range(n_groups):
        Ys[group_idx][:] = (
            Cs[group_idx] @ latents.data
            + ds[group_idx][:, np.newaxis]
            + rng.multivariate_normal(
                np.zeros(y_dims[group_idx]),
                np.diag(1 / phis[group_idx]),
                size=n_samples,
            ).T
        )

    return Y


# --- Serialization helpers ---


def _serialize_hyperprior(
    hyperprior: ObsParamsHyperPrior | ObsParamsHyperPriorStructured,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> None:
    """Serialize hyperprior to tensors and metadata dicts (in-place)."""
    if isinstance(hyperprior, ObsParamsHyperPriorStructured):
        metadata["hyperprior_type"] = "ObsParamsHyperPriorStructured"
        tensors["hyperprior.a_alpha"] = hyperprior.a_alpha
        tensors["hyperprior.b_alpha"] = hyperprior.b_alpha
        metadata["hyperprior.a_phi"] = str(hyperprior.a_phi)
        metadata["hyperprior.b_phi"] = str(hyperprior.b_phi)
        metadata["hyperprior.beta_d"] = str(hyperprior.beta_d)
    else:
        metadata["hyperprior_type"] = "ObsParamsHyperPrior"
        metadata["hyperprior"] = json.dumps(asdict(hyperprior))


def _deserialize_hyperprior(
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> ObsParamsHyperPrior | ObsParamsHyperPriorStructured:
    """Deserialize hyperprior from tensors and metadata dicts."""
    hyperprior_type = metadata["hyperprior_type"]

    if hyperprior_type == "ObsParamsHyperPriorStructured":
        return ObsParamsHyperPriorStructured(
            a_alpha=tensors["hyperprior.a_alpha"],
            b_alpha=tensors["hyperprior.b_alpha"],
            a_phi=float(metadata["hyperprior.a_phi"]),
            b_phi=float(metadata["hyperprior.b_phi"]),
            beta_d=float(metadata["hyperprior.beta_d"]),
        )
    return ObsParamsHyperPrior(**json.loads(metadata["hyperprior"]))


def _serialize_config(
    config: GFASimConfig,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> None:
    """Serialize GFASimConfig to tensors and metadata dicts (in-place)."""
    # y_dims and snr (if array) go to tensors; scalars go to metadata
    tensors["config.y_dims"] = config.y_dims
    metadata["config.n_samples"] = str(config.n_samples)
    metadata["config.x_dim"] = str(config.x_dim)

    if config.random_seed is not None:
        metadata["config.random_seed"] = json.dumps(config.random_seed)

    # snr: scalar or array
    if isinstance(config.snr, np.ndarray):
        tensors["config.snr"] = config.snr
        metadata["config.snr_is_array"] = "true"
    else:
        metadata["config.snr"] = str(config.snr)
        metadata["config.snr_is_array"] = "false"


def _deserialize_config(
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> GFASimConfig:
    """Deserialize GFASimConfig from tensors and metadata dicts."""
    # snr: check if array or scalar
    if metadata.get("config.snr_is_array") == "true":
        snr = tensors["config.snr"]
    else:
        snr = float(metadata["config.snr"])

    return GFASimConfig(
        n_samples=int(metadata["config.n_samples"]),
        y_dims=tensors["config.y_dims"],
        x_dim=int(metadata["config.x_dim"]),
        snr=snr,
        random_seed=(
            json.loads(metadata["config.random_seed"])
            if "config.random_seed" in metadata
            else None
        ),
    )


# --- Save/Load functions ---


def save_simulation(path: str | os.PathLike[str], result: GFASimulationResult) -> None:
    """Save complete simulation result to safetensors file.

    Saves the full snapshot: config, hyperprior, obs_params, latents, and
    observations. Use :func:`~latents.gfa.simulation.load_simulation` to restore.

    Parameters
    ----------
    path : str or PathLike
        Output file path (conventionally ends in .safetensors).
    result : GFASimulationResult
        Complete simulation result to save.

    See Also
    --------
    save_simulation_recipe : Save only config and hyperprior (smaller file).
    """
    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {"file_type": "snapshot"}

    # Config and hyperprior
    _serialize_config(result.config, tensors, metadata)
    _serialize_hyperprior(result.hyperprior, tensors, metadata)

    # obs_params
    tensors["obs_params.C"] = result.obs_params.C
    tensors["obs_params.d"] = result.obs_params.d
    tensors["obs_params.phi"] = result.obs_params.phi
    tensors["obs_params.alpha"] = result.obs_params.alpha
    tensors["obs_params.y_dims"] = result.obs_params.y_dims
    metadata["obs_params.x_dim"] = str(result.obs_params.x_dim)

    # latents
    tensors["latents.data"] = result.latents.data

    # observations
    tensors["observations.data"] = result.observations.data
    tensors["observations.dims"] = result.observations.dims

    save_file(tensors, path, metadata=metadata)


def save_simulation_recipe(
    path: str | os.PathLike[str],
    config: GFASimConfig,
    hyperprior: ObsParamsHyperPrior | ObsParamsHyperPriorStructured,
) -> None:
    """Save simulation recipe (config + hyperprior only).

    Saves a minimal file that can regenerate the full simulation when
    passed to :func:`~latents.gfa.simulation.simulate`. Requires ``config.random_seed``
    to be set.

    Parameters
    ----------
    path : str or PathLike
        Output file path (conventionally ends in .safetensors).
    config : GFASimConfig
        Simulation configuration. Must have random_seed set.
    hyperprior : ObsParamsHyperPrior or ObsParamsHyperPriorStructured
        Probabilistic model specification.

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

    _serialize_config(config, tensors, metadata)
    _serialize_hyperprior(hyperprior, tensors, metadata)

    save_file(tensors, path, metadata=metadata)


def load_simulation(path: str | os.PathLike[str]) -> GFASimulationResult:
    """Load complete simulation from file.

    Parameters
    ----------
    path : str or PathLike
        Path to .safetensors file saved with
        :func:`~latents.gfa.simulation.save_simulation`.

    Returns
    -------
    GFASimulationResult
        Complete simulation result.

    Raises
    ------
    ValueError
        If file contains only a recipe. Use
        :func:`~latents.gfa.simulation.load_simulation_recipe` and
        :func:`~latents.gfa.simulation.simulate` to regenerate.

    See Also
    --------
    load_simulation_recipe : Load recipe and regenerate via simulate().
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118

    if metadata.get("file_type") == "recipe":
        msg = (
            "File contains only a recipe (config + hyperprior). "
            "Use load_simulation_recipe() and simulate() to regenerate."
        )
        raise ValueError(msg)

    config = _deserialize_config(tensors, metadata)
    hyperprior = _deserialize_hyperprior(tensors, metadata)

    obs_params = ObsParamsRealization(
        C=tensors["obs_params.C"],
        d=tensors["obs_params.d"],
        phi=tensors["obs_params.phi"],
        alpha=tensors["obs_params.alpha"],
        y_dims=tensors["obs_params.y_dims"],
        x_dim=int(metadata["obs_params.x_dim"]),
    )

    latents = LatentsRealization(data=tensors["latents.data"])

    observations = ObsStatic(
        data=tensors["observations.data"],
        dims=tensors["observations.dims"],
    )

    return GFASimulationResult(
        config=config,
        hyperprior=hyperprior,
        obs_params=obs_params,
        latents=latents,
        observations=observations,
    )


def load_simulation_recipe(
    path: str | os.PathLike[str],
) -> tuple[GFASimConfig, ObsParamsHyperPrior | ObsParamsHyperPriorStructured]:
    """Load simulation recipe from file.

    Works for both recipe-only files and full snapshots (extracts just
    the specification).

    Parameters
    ----------
    path : str or PathLike
        Path to .safetensors file.

    Returns
    -------
    config : GFASimConfig
        Simulation configuration.
    hyperprior : ObsParamsHyperPrior or ObsParamsHyperPriorStructured
        Probabilistic model specification.

    Examples
    --------
    >>> config, hyperprior = load_simulation_recipe("simulation.safetensors")
    >>> result = simulate(config, hyperprior)  # Regenerate from recipe
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118

    config = _deserialize_config(tensors, metadata)
    hyperprior = _deserialize_hyperprior(tensors, metadata)

    return config, hyperprior
