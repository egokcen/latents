"""GFA benchmark configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Stream discriminators for independent RNG streams.
# These separate different sources of randomness within a single experiment.
_STREAM_DATA = 0  # obs_params + latents (signal structure)
_STREAM_OBS = 1  # observation noise realization
_STREAM_FIT = 2  # model fitting randomness

# Sweep names in fixed order for stable seed indexing.
# Order determines sweep_idx component of seeds.
_SWEEP_NAMES = ("n_samples", "y_dim_per_group", "x_dim", "n_groups", "snr")


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for a single scaling factor.

    Parameters
    ----------
    default : int or float
        Default value used when this factor is held fixed.
    values : tuple of int or float or None, default None
        Values to sweep over. ``None`` means this factor is not swept.

    Examples
    --------
    A swept factor:

    >>> n_samples = SweepConfig(default=200, values=(50, 100, 200, 500, 1000))
    >>> n_samples.is_swept
    True

    A fixed factor:

    >>> snr = SweepConfig(default=1.0)
    >>> snr.is_swept
    False
    """

    default: int | float
    values: tuple[int, ...] | tuple[float, ...] | None = None

    @property
    def is_swept(self) -> bool:
        """Whether this factor is swept (has values defined)."""
        return self.values is not None

    @property
    def max_value(self) -> int | float:
        """Maximum value for data generation at max scale.

        Returns default if not swept.
        """
        if self.values is None:
            return self.default
        return max(self.values)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for GFA benchmark experiments.

    Composes SweepConfig instances for each scaling factor. Factors with
    values=None are held at their default; factors with values are swept.

    Parameters
    ----------
    n_samples : SweepConfig
        Sample size configuration.
    y_dim_per_group : SweepConfig
        Per-group observed dimensionality configuration.
    x_dim : SweepConfig
        Latent dimensionality configuration.
    n_groups : SweepConfig
        Number of observed groups configuration.
    snr : SweepConfig
        Signal-to-noise ratio configuration.
    n_groups_total_y_dim : int, default 100
        Fixed total observed dimensionality for n_groups sweep.
        Per-group dims computed as ``n_groups_total_y_dim`` / ``n_groups``.
    n_runs : int, default 10
        Number of independent runs per sweep point.
    base_seed : int, default 42
        Base random seed for reproducibility.
    """

    n_samples: SweepConfig
    y_dim_per_group: SweepConfig
    x_dim: SweepConfig
    n_groups: SweepConfig
    snr: SweepConfig

    n_groups_total_y_dim: int = 100
    n_runs: int = 10
    base_seed: int = 42

    def get_data_seed(
        self,
        sweep_name: str,
        run_idx: int,
        sweep_value_idx: int | None = None,
    ) -> list[int]:
        """Get seed for data generation (obs_params and latents).

        Returns a sequence suitable for ``np.random.default_rng()``. NumPy's
        SeedSequence hashes the components into independent streams.

        Parameters
        ----------
        sweep_name : str
            Name of the sweep (n_samples, y_dim_per_group, x_dim, n_groups,
            snr).
        run_idx : int
            Index of the run (0 to n_runs - 1).
        sweep_value_idx : int or None, default None
            Index of the sweep value. Required for structural sweeps
            (x_dim, n_groups) where data structure differs per value.
            Omit for subsetting sweeps (n_samples, y_dim_per_group, snr)
            where signal structure is generated once per run.

        Returns
        -------
        list of int
            Seed sequence for np.random.default_rng().

        Raises
        ------
        ValueError
            If sweep_name is not recognized.
        """
        sweep_idx = self._get_sweep_idx(sweep_name)
        components = [sweep_idx, run_idx]
        if sweep_value_idx is not None:
            components.append(sweep_value_idx)
        components.extend([_STREAM_DATA, self.base_seed])
        return components

    def get_obs_seed(
        self,
        sweep_name: str,
        run_idx: int,
        sweep_value_idx: int,
    ) -> list[int]:
        """Get seed for observation sampling (noise realization).

        Used for sweeps like SNR where the signal structure is shared but
        each sweep value needs an independent noise realization.

        Parameters
        ----------
        sweep_name : str
            Name of the sweep.
        run_idx : int
            Index of the run.
        sweep_value_idx : int
            Index of the sweep value. Always required since each value
            gets an independent noise realization.

        Returns
        -------
        list of int
            Seed sequence for np.random.default_rng().

        Raises
        ------
        ValueError
            If sweep_name is not recognized.
        """
        sweep_idx = self._get_sweep_idx(sweep_name)
        return [sweep_idx, run_idx, sweep_value_idx, _STREAM_OBS, self.base_seed]

    def get_fit_seed(
        self,
        sweep_name: str,
        run_idx: int,
        sweep_value_idx: int,
    ) -> list[int]:
        """Get seed for model fitting.

        Parameters
        ----------
        sweep_name : str
            Name of the sweep.
        run_idx : int
            Index of the run.
        sweep_value_idx : int
            Index of the sweep value. Always required since each fit
            is independent.

        Returns
        -------
        list of int
            Seed sequence for np.random.default_rng().

        Raises
        ------
        ValueError
            If sweep_name is not recognized.
        """
        sweep_idx = self._get_sweep_idx(sweep_name)
        return [sweep_idx, run_idx, sweep_value_idx, _STREAM_FIT, self.base_seed]

    def _get_sweep_idx(self, sweep_name: str) -> int:
        """Get index of sweep name in _SWEEP_NAMES.

        Raises ValueError if sweep_name is not recognized.
        """
        try:
            return _SWEEP_NAMES.index(sweep_name)
        except ValueError:
            msg = f"Unknown sweep '{sweep_name}'. Valid: {list(_SWEEP_NAMES)}"
            raise ValueError(msg) from None

    def get_sweep_config(self, sweep_name: str) -> SweepConfig:
        """Get the SweepConfig for a given factor.

        Parameters
        ----------
        sweep_name : str
            Name of the sweep (n_samples, y_dim_per_group, x_dim, n_groups,
            snr).

        Returns
        -------
        SweepConfig
            Configuration for the requested factor.

        Raises
        ------
        ValueError
            If sweep_name is not recognized.
        """
        sweep_map = {
            "n_samples": self.n_samples,
            "y_dim_per_group": self.y_dim_per_group,
            "x_dim": self.x_dim,
            "n_groups": self.n_groups,
            "snr": self.snr,
        }
        if sweep_name not in sweep_map:
            valid = list(sweep_map.keys())
            msg = f"Unknown sweep '{sweep_name}'. Valid: {valid}"
            raise ValueError(msg)
        return sweep_map[sweep_name]

    def get_active_sweeps(self) -> list[str]:
        """Get list of factors that are actively swept.

        Returns
        -------
        list of str
            Names of factors where ``is_swept`` is ``True``.
        """
        return [name for name in _SWEEP_NAMES if self.get_sweep_config(name).is_swept]


RUNTIME_CONFIG = BenchmarkConfig(
    n_samples=SweepConfig(default=1_000, values=(100, 1_000, 10_000, 100_000)),
    y_dim_per_group=SweepConfig(default=100, values=(10, 30, 100, 300, 1_000)),
    x_dim=SweepConfig(default=3, values=(10, 30, 100, 200)),
    n_groups=SweepConfig(default=2, values=(1, 5, 10, 50, 100)),
    snr=SweepConfig(default=1.0),  # Fixed, not swept
    n_groups_total_y_dim=100,  # Fixed total y_dim for n_groups sweep
)
"""Runtime benchmark configuration. SNR is fixed (not a scaling factor)."""


RECOVERY_CONFIG = BenchmarkConfig(
    n_samples=SweepConfig(default=1_000, values=(100, 1_000, 10_000, 100_000)),
    y_dim_per_group=SweepConfig(default=30, values=(3, 10, 30, 100)),
    x_dim=SweepConfig(default=3, values=(1, 3, 10, 30)),
    n_groups=SweepConfig(default=2, values=(1, 3, 10, 30)),
    snr=SweepConfig(default=1.0, values=(0.1, 0.3, 1.0, 3.0, 10.0)),
    n_groups_total_y_dim=30,  # Fixed total y_dim for n_groups sweep
)
"""Parameter recovery benchmark configuration. All factors swept."""


@dataclass(frozen=True)
class DimensionalityConfig:
    """Configuration for dimensionality recovery benchmarks.

    Unlike :class:`BenchmarkConfig` which sweeps one factor at a time,
    this uses a full factorial design over ``n_samples`` x ``snr``.

    Parameters
    ----------
    n_samples_values : tuple of int, default (100, 300, 1_000, 3_000, 10_000)
        Sample sizes to sweep.
    snr_values : tuple of float, default (0.1, 0.3, 1.0, 3.0, 10.0)
        Signal-to-noise ratios to sweep.
    x_dim_true : int, default 10
        True latent dimensionality for data generation.
    x_dim_init : int, default 20
        Initial latent dimensionality for fitting (deliberately exceeds
        ``x_dim_true`` to test post-hoc selection).
    n_groups : int, default 1
        Number of observed groups.
    y_dim_per_group : int, default 50
        Observed dimensionality per group.
    n_runs : int, default 10
        Independent runs per grid point.
    base_seed : int, default 42
        Base random seed for reproducibility.
    """

    n_samples_values: tuple[int, ...] = (100, 300, 1_000, 3_000, 10_000)
    snr_values: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    x_dim_true: int = 10
    x_dim_init: int = 20
    n_groups: int = 1
    y_dim_per_group: int = 50
    n_runs: int = 10
    base_seed: int = 42

    @property
    def max_n_samples(self) -> int:
        """Maximum sample size (used for generating full dataset)."""
        return max(self.n_samples_values)

    @property
    def y_dims(self) -> np.ndarray:
        """Per-group dimensionalities array, shape (n_groups,)."""
        return np.full(self.n_groups, self.y_dim_per_group, dtype=np.intp)

    def get_data_seed(self, run_idx: int) -> list[int]:
        """Get seed for ground truth generation (signal structure).

        One per run; shared across all grid points within a run.

        Parameters
        ----------
        run_idx : int
            Index of the run (0 to n_runs - 1).

        Returns
        -------
        list of int
            Seed sequence for ``np.random.default_rng()``.
        """
        return [run_idx, _STREAM_DATA, self.base_seed]

    def get_obs_seed(self, run_idx: int, snr_idx: int) -> list[int]:
        """Get seed for observation resampling at a given SNR.

        One per (run, SNR) pair; shared across ``n_samples`` values.

        Parameters
        ----------
        run_idx : int
            Index of the run.
        snr_idx : int
            Index into ``snr_values``.

        Returns
        -------
        list of int
            Seed sequence for ``np.random.default_rng()``.
        """
        return [run_idx, snr_idx, _STREAM_OBS, self.base_seed]

    def get_fit_seed(self, run_idx: int, snr_idx: int, n_samples_idx: int) -> list[int]:
        """Get seed for model fitting at a specific grid point.

        Parameters
        ----------
        run_idx : int
            Index of the run.
        snr_idx : int
            Index into ``snr_values``.
        n_samples_idx : int
            Index into ``n_samples_values``.

        Returns
        -------
        list of int
            Seed sequence for ``np.random.default_rng()``.
        """
        return [run_idx, snr_idx, n_samples_idx, _STREAM_FIT, self.base_seed]


DIMENSIONALITY_CONFIG = DimensionalityConfig()
"""Dimensionality recovery benchmark configuration (n_samples x SNR grid)."""
