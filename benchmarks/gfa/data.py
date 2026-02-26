"""GFA ground truth generation and data subsetting for benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from benchmarks.gfa.config import BenchmarkConfig
from latents.data import ObsStatic
from latents.gfa.config import GFASimConfig
from latents.gfa.simulation import GFASimulationResult, sample_observations, simulate
from latents.observation import ObsParamsHyperPrior, ObsParamsRealization, adjust_snr
from latents.state import LatentsRealization

_MAG = 100.0  # Controls variance of alpha; larger = tighter concentration
BENCHMARK_HYPERPRIOR = ObsParamsHyperPrior(
    a_alpha=_MAG,
    b_alpha=_MAG,
    a_phi=1.0,
    b_phi=1.0,
    beta_d=1.0,
)
"""Default hyperprior for benchmark simulations.

Uses ``Gamma(1, 1)`` priors for precisions and unit precision for means.
These values are numerically stable for simulation, unlike the small
defaults in :class:`~latents.observation.ObsParamsHyperPrior` which
are designed for uninformative inference.
"""


def build_y_dims(
    config: BenchmarkConfig,
    sweep_name: str,
    sweep_value: int | float,
) -> np.ndarray:
    """Construct y_dims array for a specific experiment.

    Handles the special case where n_groups sweep holds total y_dim fixed
    while varying per-group dimensionality.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration with sweep specs and defaults.
    sweep_name : str
        Name of the factor being swept.
    sweep_value : int or float
        Current value of the swept factor.

    Returns
    -------
    ndarray of shape (n_groups,)
        Per-group observed dimensionalities.

    Raises
    ------
    ValueError
        If ``n_groups_total_y_dim`` is not divisible by ``n_groups`` for
        n_groups sweep.
    """
    if sweep_name == "n_groups":
        # n_groups sweep: total y_dim fixed, per-group varies
        n_groups = int(sweep_value)
        if config.n_groups_total_y_dim % n_groups != 0:
            msg = (
                f"n_groups_total_y_dim ({config.n_groups_total_y_dim}) "
                f"must be divisible by n_groups ({n_groups})"
            )
            raise ValueError(msg)
        y_dim_per_group = config.n_groups_total_y_dim // n_groups
    elif sweep_name == "y_dim_per_group":
        # y_dim_per_group sweep: use swept value
        n_groups = int(config.n_groups.default)
        y_dim_per_group = int(sweep_value)
    else:
        # All other sweeps: use defaults
        n_groups = int(config.n_groups.default)
        y_dim_per_group = int(config.y_dim_per_group.default)

    return np.full(n_groups, y_dim_per_group, dtype=np.intp)


def generate_ground_truth(
    n_samples: int,
    y_dims: np.ndarray,
    x_dim: int,
    snr: float,
    seed: int | Sequence[int],
) -> GFASimulationResult:
    """Generate ground truth data for benchmarks.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    y_dims : ndarray of shape (n_groups,)
        Per-group observed dimensionalities.
    x_dim : int
        Latent dimensionality.
    snr : float
        Signal-to-noise ratio.
    seed : int or sequence of int
        Random seed for reproducibility. Accepts sequences for structured
        seeding (e.g., from BenchmarkConfig.get_data_seed()).

    Returns
    -------
    GFASimulationResult
        Complete simulation result with config, hyperprior, obs_params,
        latents, and observations.
    """
    sim_config = GFASimConfig(
        n_samples=n_samples,
        y_dims=y_dims,
        x_dim=x_dim,
        snr=snr,
        random_seed=seed,
    )
    return simulate(sim_config, BENCHMARK_HYPERPRIOR)


def subset_by_samples(
    result: GFASimulationResult,
    n_samples: int,
) -> tuple[ObsStatic, LatentsRealization]:
    """Subset observations and latents to first n samples.

    Parameters
    ----------
    result : GFASimulationResult
        Full simulation result generated at maximum n_samples.
    n_samples : int
        Number of samples to keep.

    Returns
    -------
    observations : ObsStatic
        Observations with first n_samples columns.
    latents : LatentsRealization
        Latents with first n_samples columns.
    """
    # Y: (y_dim, n_samples_full) -> (y_dim, n_samples)
    Y_subset = ObsStatic(
        data=result.observations.data[:, :n_samples].copy(),
        dims=result.observations.dims.copy(),
    )
    # X: (x_dim, n_samples_full) -> (x_dim, n_samples)
    X_subset = LatentsRealization(data=result.latents.data[:, :n_samples].copy())

    return Y_subset, X_subset


def subset_by_y_dim(
    result: GFASimulationResult,
    y_dim_per_group: int,
) -> tuple[ObsStatic, ObsParamsRealization]:
    """Subset observations and obs_params to first y_dim dimensions per group.

    Parameters
    ----------
    result : GFASimulationResult
        Full simulation result generated at maximum y_dim.
    y_dim_per_group : int
        Number of observed dimensions to keep per group.

    Returns
    -------
    observations : ObsStatic
        Observations with first y_dim_per_group dimensions per group.
    obs_params : ObsParamsRealization
        Observation parameters (``C``, ``d``, ``phi``) subsetted to match.
    """
    n_groups = len(result.observations.dims)
    new_y_dims = np.full(n_groups, y_dim_per_group, dtype=np.intp)

    # Build index mask for dimensions to keep
    # For each group, keep first y_dim_per_group dimensions
    old_y_dims = result.observations.dims
    keep_indices = []
    offset = 0
    for old_dim in old_y_dims:
        keep_indices.extend(range(offset, offset + y_dim_per_group))
        offset += old_dim
    keep_indices = np.array(keep_indices)

    # Subset observations: Y[keep_indices, :]
    Y_subset = ObsStatic(
        data=result.observations.data[keep_indices, :].copy(),
        dims=new_y_dims,
    )

    # Subset obs_params: C[keep_indices, :], d[keep_indices], phi[keep_indices]
    obs_params_subset = ObsParamsRealization(
        C=result.obs_params.C[keep_indices, :].copy(),
        d=result.obs_params.d[keep_indices].copy(),
        phi=result.obs_params.phi[keep_indices].copy(),
        alpha=result.obs_params.alpha.copy(),  # (n_groups, x_dim) unchanged
        y_dims=new_y_dims,
        x_dim=result.obs_params.x_dim,
    )

    return Y_subset, obs_params_subset


def resample_at_snr(
    result: GFASimulationResult,
    snr: float,
    obs_seed: int | Sequence[int],
) -> tuple[ObsStatic, ObsParamsRealization]:
    """Adjust SNR and re-sample observations with independent noise.

    Scales observation precisions (``phi``) to achieve the target SNR, then
    generates new observations. The signal structure (``C``, ``d``, ``alpha``,
    latents) is unchanged; only the noise level differs.

    Parameters
    ----------
    result : GFASimulationResult
        Base simulation result with signal structure at default SNR.
    snr : float
        Target signal-to-noise ratio.
    obs_seed : int or sequence of int
        Random seed for observation noise sampling.

    Returns
    -------
    observations : ObsStatic
        Observations generated at the target SNR.
    obs_params : ObsParamsRealization
        Observation parameters with ``phi`` scaled to the target SNR.
        ``C``, ``d``, and ``alpha`` are unchanged from the original.
    """
    obs_params_snr = adjust_snr(result.obs_params, snr)
    rng = np.random.default_rng(obs_seed)
    Y_snr = sample_observations(result.latents, obs_params_snr, rng)
    return Y_snr, obs_params_snr
