"""GFA parameter recovery benchmark orchestration."""

from __future__ import annotations

import pandas as pd

from benchmarks.gfa.config import BenchmarkConfig
from benchmarks.gfa.data import (
    build_y_dims,
    generate_ground_truth,
    resample_at_snr,
    subset_by_samples,
    subset_by_y_dim,
)
from benchmarks.metrics import (
    denoised_r2,
    latent_permutation,
    relative_l2_error,
    subspace_error,
)
from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.gfa.model import GFAModel
from latents.observation import ObsParamsRealization
from latents.state import LatentsRealization

# Sweep categories for recovery
_SUBSETTING_SWEEPS = {"n_samples", "y_dim_per_group"}
_STRUCTURAL_SWEEPS = {"x_dim", "n_groups"}

# Metric names determine CSV column naming: {name}_mean, {name}_sem
_METRIC_NAMES = (
    "C_subspace_error",
    "d_error",
    "noise_var_error",
    "ard_var_error",
    "denoised_r2",
)


def run_single_fit(
    Y: ObsStatic,
    x_dim_true: int,
    seed: list[int],
    obs_params_true: ObsParamsRealization,
    latents_true: LatentsRealization,
) -> dict:
    """Run one GFA fit and compute recovery metrics against ground truth.

    Parameters
    ----------
    Y : ObsStatic
        Observation data to fit.
    x_dim_true : int
        True latent dimensionality.
    seed : list of int
        Random seed sequence for fitting.
    obs_params_true : ObsParamsRealization
        Ground truth observation parameters for metric comparison.
    latents_true : LatentsRealization
        Ground truth latent factors for metric comparison.

    Returns
    -------
    dict
        Recovery metrics (C_subspace_error, d_error, noise_var_error,
        ard_var_error, denoised_r2) plus fit diagnostics (converged,
        decreasing_lb, private_var_floor). ARD variance error uses
        permutation-aligned latent dimensions.
    """
    fit_config = GFAFitConfig(
        x_dim_init=x_dim_true,
        prune_x=False,  # Fixed dimensionality; no ARD pruning
        save_x=True,  # Need latent posteriors for denoised R²
        fit_tol=1e-8,  # Tight tolerance ensures full convergence
        max_iter=50_000,  # High limit; most fits converge much sooner
        random_seed=seed,
        min_var_frac=1e-6,  # Small floor to avoid numerical issues
    )

    model = GFAModel(config=fit_config)
    model.fit(Y)

    # Extract posterior means
    obs_est = model.obs_posterior.posterior_mean
    latents_est = model.latents_posterior.posterior_mean

    # Align estimated latent dimensions to true ones via correlation matching
    perm = latent_permutation(latents_true.data, latents_est.data)

    # Compute recovery metrics (report variances, not precisions)
    return {
        "C_subspace_error": subspace_error(obs_params_true.C, obs_est.C),
        "d_error": relative_l2_error(obs_params_true.d, obs_est.d),
        "noise_var_error": relative_l2_error(
            1.0 / obs_params_true.phi, 1.0 / obs_est.phi
        ),
        "ard_var_error": relative_l2_error(
            1.0 / obs_params_true.alpha, 1.0 / obs_est.alpha[:, perm]
        ),
        "denoised_r2": denoised_r2(
            obs_params_true.C,
            latents_true.data,
            obs_params_true.d,
            obs_est.C,
            latents_est.data,
            obs_est.d,
        ),
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

    Handles subsetting sweeps (generate once, subset per value), structural
    sweeps (fresh generation per value), and SNR sweeps (adjust noise level).

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    sweep_name : str
        Name of the sweep (n_samples, y_dim_per_group, x_dim, n_groups, snr).
    run_idx : int
        Index of the run (0 to n_runs - 1).

    Returns
    -------
    results : list of dict
        Per-fit recovery metrics, each with sweep_value and run_idx added.
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
    if sweep_name == "snr":
        return _run_snr_sweep_single_run(config, run_idx)

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

    # Same signal structure for all sweep values (no sweep_value_idx)
    data_seed = config.get_data_seed(sweep_name, run_idx)
    sim_result = generate_ground_truth(n_samples, y_dims, x_dim, snr, data_seed)

    # Fit at each sweep value
    for value_idx, sweep_value in enumerate(sweep_values):
        # Subset data and determine ground truth
        if sweep_name == "n_samples":
            Y_subset, X_subset = subset_by_samples(sim_result, int(sweep_value))
            obs_params_true = sim_result.obs_params
            latents_true = X_subset
        else:  # y_dim_per_group
            Y_subset, obs_params_sub = subset_by_y_dim(sim_result, int(sweep_value))
            obs_params_true = obs_params_sub  # C, d, phi subsetted; alpha unchanged
            latents_true = sim_result.latents

        # Fit
        fit_seed = config.get_fit_seed(sweep_name, run_idx, value_idx)
        fit_result = run_single_fit(
            Y_subset, x_dim, fit_seed, obs_params_true, latents_true
        )

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
        fit_result = run_single_fit(
            sim_result.observations,
            x_dim,
            fit_seed,
            sim_result.obs_params,
            sim_result.latents,
        )

        # Add metadata and collect
        fit_result["sweep_value"] = sweep_value
        fit_result["run_idx"] = run_idx
        results.append(fit_result)

        _collect_warnings(fit_result, sweep_name, sweep_value, run_idx, warnings)

    return results, warnings


def _run_snr_sweep_single_run(
    config: BenchmarkConfig,
    run_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Run SNR sweep for one run.

    Generates base params once at default SNR, then per SNR value:
    adjusts phi via adjust_snr, re-samples observations, and fits.
    """
    results = []
    warnings = []

    sweep_values = config.snr.values
    x_dim = int(config.x_dim.default)
    n_samples = int(config.n_samples.default)
    y_dims = build_y_dims(config, "snr", config.snr.default)

    # Generate base params once (same signal structure for all SNR values)
    data_seed = config.get_data_seed("snr", run_idx)
    sim_result = generate_ground_truth(
        n_samples, y_dims, x_dim, config.snr.default, data_seed
    )

    for value_idx, snr_value in enumerate(sweep_values):
        # Resample observations at target SNR
        obs_seed = config.get_obs_seed("snr", run_idx, value_idx)
        Y_snr, obs_params_snr = resample_at_snr(sim_result, snr_value, obs_seed)

        # Fit (obs_params_snr has adjusted phi but original C/d/alpha)
        fit_seed = config.get_fit_seed("snr", run_idx, value_idx)
        fit_result = run_single_fit(
            Y_snr, x_dim, fit_seed, obs_params_snr, sim_result.latents
        )

        # Add metadata and collect
        fit_result["sweep_value"] = snr_value
        fit_result["run_idx"] = run_idx
        results.append(fit_result)

        _collect_warnings(fit_result, "snr", snr_value, run_idx, warnings)

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
    """Aggregate per-fit results to mean +/- SEM per sweep value.

    Parameters
    ----------
    results : list of dict
        Per-fit recovery metrics from run_single_run().

    Returns
    -------
    DataFrame
        Aggregated results with columns: sweep_value, plus _mean and _sem
        columns for each metric (C_subspace_error, d_error, noise_var_error,
        ard_var_error, denoised_r2).
    """
    df = pd.DataFrame(results)

    agg_dict = {}
    for metric in _METRIC_NAMES:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_sem"] = (metric, "sem")

    agg = df.groupby("sweep_value").agg(**agg_dict)

    # Reset index to make sweep_value a column, sort for consistent output
    return agg.reset_index().sort_values("sweep_value").reset_index(drop=True)
