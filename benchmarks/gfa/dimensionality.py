"""GFA dimensionality recovery benchmark orchestration.

Evaluates post-hoc dimensionality selection via
:meth:`~latents.observation.ObsParamsPosterior.compute_dimensionalities`
across a 2D grid of sample sizes and signal-to-noise ratios.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.gfa.config import DimensionalityConfig
from benchmarks.gfa.data import generate_ground_truth, resample_at_snr
from latents.data import ObsStatic
from latents.gfa.config import GFAFitConfig
from latents.gfa.model import GFAModel


def run_single_fit(
    Y: ObsStatic,
    config: DimensionalityConfig,
    seed: list[int],
) -> dict:
    """Fit GFA and compute dimensionality error.

    Uses :meth:`~latents.observation.ObsParamsPosterior.compute_dimensionalities`
    with default cutoffs to determine effective dimensionality.

    Parameters
    ----------
    Y : ObsStatic
        Observation data.
    config : DimensionalityConfig
        Benchmark configuration (provides ``x_dim_init`` and ``x_dim_true``).
    seed : list of int
        Random seed for fitting.

    Returns
    -------
    dict
        Keys: ``x_dim_error`` (signed; positive = overestimate),
        ``converged``, ``decreasing_lb``, ``private_var_floor``.
    """
    fit_config = GFAFitConfig(
        x_dim_init=config.x_dim_init,
        prune_x=False,  # No pruning; dimensionality via post-hoc selection
        save_x=False,  # Latent posteriors not needed
        fit_tol=1e-8,
        max_iter=50_000,
        random_seed=seed,
        min_var_frac=1e-6,
    )

    model = GFAModel(config=fit_config)
    model.fit(Y)

    # Compute effective dimensionality via default cutoffs.
    # sig_dims: (n_groups, x_dim) boolean — True for significant dimensions.
    _, sig_dims, _, _ = model.obs_posterior.compute_dimensionalities()
    effective_x_dim = int(sig_dims.sum())

    return {
        "x_dim_error": effective_x_dim - config.x_dim_true,
        "converged": model.flags.converged,
        "decreasing_lb": model.flags.decreasing_lb,
        "private_var_floor": model.flags.private_var_floor,
    }


def run_single_run(
    config: DimensionalityConfig,
    run_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Run all grid points for one independent run.

    Generates ground truth once at ``max_n_samples``, then iterates over
    SNR values (resample observations) and sample sizes (subset).

    Parameters
    ----------
    config : DimensionalityConfig
        Benchmark configuration.
    run_idx : int
        Run index (0 to ``n_runs - 1``).

    Returns
    -------
    results : list of dict
        Per-fit results with ``n_samples``, ``snr``, ``x_dim_error``,
        ``run_idx``.
    warnings : list of dict
        Fits with convergence issues.
    """
    results = []
    warnings = []

    # Generate ground truth once at max n_samples
    data_seed = config.get_data_seed(run_idx)
    sim_result = generate_ground_truth(
        n_samples=config.max_n_samples,
        y_dims=config.y_dims,
        x_dim=config.x_dim_true,
        snr=1.0,  # Base SNR; resample_at_snr adjusts per target
        seed=data_seed,
    )

    # Outer loop: SNR (resample observations once per SNR)
    for snr_idx, snr_value in enumerate(config.snr_values):
        obs_seed = config.get_obs_seed(run_idx, snr_idx)
        Y_snr, _ = resample_at_snr(sim_result, snr_value, obs_seed)

        # Inner loop: n_samples (subset from SNR-resampled observations)
        for ns_idx, n_samples in enumerate(config.n_samples_values):
            # Subset observations: take first n_samples columns.
            # Can't reuse data.subset_by_samples() which expects
            # GFASimulationResult; resample_at_snr returns ObsStatic.
            Y_subset = ObsStatic(
                data=Y_snr.data[:, :n_samples].copy(),
                dims=Y_snr.dims.copy(),
            )

            fit_seed = config.get_fit_seed(run_idx, snr_idx, ns_idx)
            fit_result = run_single_fit(Y_subset, config, fit_seed)

            fit_result["n_samples"] = n_samples
            fit_result["snr"] = snr_value
            fit_result["run_idx"] = run_idx
            results.append(fit_result)

            _collect_warnings(fit_result, n_samples, snr_value, run_idx, warnings)

    return results, warnings


def _collect_warnings(
    fit_result: dict,
    n_samples: int,
    snr: float,
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
                "n_samples": n_samples,
                "snr": snr,
                "run_idx": run_idx,
                "issues": issues,
            }
        )


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-fit results to mean +/- SEM per grid point.

    Parameters
    ----------
    results : list of dict
        Per-fit results from :func:`~benchmarks.gfa.dimensionality.run_single_run`.

    Returns
    -------
    DataFrame
        Columns: ``n_samples``, ``snr``, ``x_dim_error_mean``,
        ``x_dim_error_sem``.
    """
    df = pd.DataFrame(results)

    agg = df.groupby(["n_samples", "snr"]).agg(
        x_dim_error_mean=("x_dim_error", "mean"),
        x_dim_error_sem=("x_dim_error", "sem"),
    )

    return agg.reset_index().sort_values(["snr", "n_samples"]).reset_index(drop=True)
