"""GFA runtime benchmark orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.gfa.config import BenchmarkConfig
from benchmarks.gfa.data import (
    build_y_dims,
    generate_ground_truth,
    subset_by_samples,
    subset_by_y_dim,
)
from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.gfa.model import GFAModel

# Sweeps where data is generated once per run and subsetted per value
_SUBSETTING_SWEEPS = {"n_samples", "y_dim_per_group"}

# Sweeps where data structure differs per value (fresh generation)
_STRUCTURAL_SWEEPS = {"x_dim", "n_groups"}


def run_single_fit(
    Y: ObsStatic,
    x_dim_init: int,
    seed: list[int],
) -> dict:
    """Run one GFA fit and return timing results.

    Parameters
    ----------
    Y : ObsStatic
        Observation data to fit.
    x_dim_init : int
        Initial latent dimensionality.
    seed : list of int
        Random seed sequence for fitting.

    Returns
    -------
    dict
        Timing results with keys: runtime_per_iter, n_iters, total_runtime,
        converged, decreasing_lb, private_var_floor.
    """
    fit_config = GFAFitConfig(
        x_dim_init=x_dim_init,
        prune_x=False,  # Disable ARD pruning for runtime benchmarks
        fit_tol=1e-8,  # Tight tolerance ensures full convergence
        max_iter=50_000,  # High limit; most fits converge much sooner
        random_seed=seed,
        min_var_frac=1e-6,  # Small floor to avoid numerical issues
    )

    model = GFAModel(config=fit_config)
    model.fit(Y)  # No callbacks (quiet fit)

    # Extract timing from tracker
    n_iters = len(model.tracker.iter_time)
    total_runtime = float(np.sum(model.tracker.iter_time))
    runtime_per_iter = float(np.mean(model.tracker.iter_time))

    return {
        "runtime_per_iter": runtime_per_iter,
        "n_iters": n_iters,
        "total_runtime": total_runtime,
        "converged": model.flags.converged,
        "decreasing_lb": model.flags.decreasing_lb,
        "private_var_floor": model.flags.private_var_floor,
    }


def run_single_run(
    config: BenchmarkConfig,
    sweep_name: str,
    run_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Run all sweep values for one run.

    Handles both subsetting sweeps (generate once, subset per value) and
    structural sweeps (fresh generation per value).

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    sweep_name : str
        Name of the sweep (n_samples, y_dim_per_group, x_dim, n_groups).
    run_idx : int
        Index of the run (0 to n_runs - 1).

    Returns
    -------
    results : list of dict
        Per-fit timing results, each with sweep_value and run_idx added.
    warnings : list of dict
        Fits with issues (not converged, decreasing lb, variance floor).
    """
    sweep_config = config.get_sweep_config(sweep_name)
    if not sweep_config.is_swept:
        msg = f"Sweep '{sweep_name}' has no values defined"
        raise ValueError(msg)

    if sweep_name in _SUBSETTING_SWEEPS:
        return _run_subsetting_sweep_single_run(config, sweep_name, run_idx)
    if sweep_name in _STRUCTURAL_SWEEPS:
        return _run_structural_sweep_single_run(config, sweep_name, run_idx)

    msg = f"Unknown sweep type: {sweep_name}"
    raise ValueError(msg)


def _run_subsetting_sweep_single_run(
    config: BenchmarkConfig,
    sweep_name: str,
    run_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Run subsetting sweep for one run.

    Generates data once at max scale, subsets for each sweep value.
    """
    results = []
    warnings = []

    sweep_config = config.get_sweep_config(sweep_name)
    sweep_values = sweep_config.values

    # Generate data at max scale
    max_value = sweep_config.max_value
    y_dims = build_y_dims(config, sweep_name, max_value)

    if sweep_name == "n_samples":
        n_samples = int(max_value)
    else:
        n_samples = int(config.n_samples.default)

    x_dim = int(config.x_dim.default)
    snr = config.snr.default

    data_seed = config.get_data_seed(sweep_name, run_idx)
    sim_result = generate_ground_truth(n_samples, y_dims, x_dim, snr, data_seed)

    # Fit at each sweep value
    for value_idx, sweep_value in enumerate(sweep_values):
        # Subset data
        if sweep_name == "n_samples":
            Y_subset, _ = subset_by_samples(sim_result, int(sweep_value))
        else:  # y_dim_per_group
            Y_subset, _ = subset_by_y_dim(sim_result, int(sweep_value))

        # Fit
        fit_seed = config.get_fit_seed(sweep_name, run_idx, value_idx)
        fit_result = run_single_fit(Y_subset, x_dim, fit_seed)

        # Add metadata and collect
        fit_result["sweep_value"] = sweep_value
        fit_result["run_idx"] = run_idx
        results.append(fit_result)

        _collect_warnings(fit_result, sweep_name, sweep_value, run_idx, warnings)

    return results, warnings


def _run_structural_sweep_single_run(
    config: BenchmarkConfig,
    sweep_name: str,
    run_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Run structural sweep for one run.

    Generates fresh data for each sweep value.
    """
    results = []
    warnings = []

    sweep_config = config.get_sweep_config(sweep_name)
    sweep_values = sweep_config.values

    n_samples = int(config.n_samples.default)
    snr = config.snr.default

    for value_idx, sweep_value in enumerate(sweep_values):
        y_dims = build_y_dims(config, sweep_name, sweep_value)

        # x_dim comes from sweep value for x_dim sweep, otherwise from config
        x_dim = int(sweep_value) if sweep_name == "x_dim" else int(config.x_dim.default)

        # Generate data for this specific value
        data_seed = config.get_data_seed(sweep_name, run_idx, value_idx)
        sim_result = generate_ground_truth(n_samples, y_dims, x_dim, snr, data_seed)

        # Fit
        fit_seed = config.get_fit_seed(sweep_name, run_idx, value_idx)
        fit_result = run_single_fit(sim_result.observations, x_dim, fit_seed)

        # Add metadata and collect
        fit_result["sweep_value"] = sweep_value
        fit_result["run_idx"] = run_idx
        results.append(fit_result)

        _collect_warnings(fit_result, sweep_name, sweep_value, run_idx, warnings)

    return results, warnings


def _collect_warnings(
    fit_result: dict,
    sweep_name: str,
    sweep_value: int | float,
    run_idx: int,
    warnings: list[dict],
) -> None:
    """Check fit result for issues and append to warnings if found."""
    issues = []
    if not fit_result["converged"]:
        issues.append("not_converged")
    if fit_result["decreasing_lb"]:
        issues.append("decreasing_lb")
    if fit_result["private_var_floor"]:
        issues.append("private_var_floor")

    if issues:
        warnings.append(
            {
                "sweep_name": sweep_name,
                "sweep_value": sweep_value,
                "run_idx": run_idx,
                "issues": issues,
            }
        )


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-fit results to mean ± SEM per sweep value.

    Parameters
    ----------
    results : list of dict
        Per-fit timing results from run_single_run().

    Returns
    -------
    DataFrame
        Aggregated results with columns: sweep_value, mean_runtime_per_iter,
        sem_runtime_per_iter, mean_iters, sem_iters, mean_runtime, sem_runtime.
    """
    df = pd.DataFrame(results)

    agg = df.groupby("sweep_value").agg(
        mean_runtime_per_iter=("runtime_per_iter", "mean"),
        sem_runtime_per_iter=("runtime_per_iter", "sem"),
        mean_iters=("n_iters", "mean"),
        sem_iters=("n_iters", "sem"),
        mean_runtime=("total_runtime", "mean"),
        sem_runtime=("total_runtime", "sem"),
    )

    # Reset index to make sweep_value a column, sort for consistent output
    return agg.reset_index().sort_values("sweep_value").reset_index(drop=True)
