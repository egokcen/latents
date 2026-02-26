"""GFA benchmark CLI subcommands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from benchmarks.gfa import dimensionality, recovery, runtime
from benchmarks.gfa.config import DIMENSIONALITY_CONFIG, RECOVERY_CONFIG, RUNTIME_CONFIG

_RUNTIME_SWEEPS = ["n_samples", "y_dim_per_group", "x_dim", "n_groups"]
_RECOVERY_SWEEPS = ["n_samples", "y_dim_per_group", "x_dim", "n_groups", "snr"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface (called by main CLI)
# ---------------------------------------------------------------------------


def register(method_parsers: argparse._SubParsersAction) -> None:
    """Register GFA subcommands with the main CLI.

    Parameters
    ----------
    method_parsers : argparse._SubParsersAction
        Subparser action from the main CLI's argument parser.
    """
    gfa_parser = method_parsers.add_parser("gfa", help="GFA benchmarks")
    subs = gfa_parser.add_subparsers(dest="command", required=True)

    # "all" - run all GFA benchmarks
    all_parser = subs.add_parser("all", help="Run all GFA benchmarks")
    all_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    all_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Base output directory (default: benchmarks/results/)",
    )

    # "runtime" - run runtime benchmarks
    runtime_parser = subs.add_parser("runtime", help="Runtime scaling benchmarks")
    sweep_group = runtime_parser.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument(
        "--sweep",
        choices=_RUNTIME_SWEEPS,
        help="Run a single sweep",
    )
    sweep_group.add_argument(
        "--all",
        action="store_true",
        help="Run all runtime sweeps",
    )
    runtime_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    runtime_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Base output directory (default: benchmarks/results/)",
    )

    # "recovery" - run parameter recovery benchmarks
    recovery_parser = subs.add_parser("recovery", help="Parameter recovery benchmarks")
    recovery_sweep_group = recovery_parser.add_mutually_exclusive_group(required=True)
    recovery_sweep_group.add_argument(
        "--sweep",
        choices=_RECOVERY_SWEEPS,
        help="Run a single sweep",
    )
    recovery_sweep_group.add_argument(
        "--all",
        action="store_true",
        help="Run all recovery sweeps",
    )
    recovery_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    recovery_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Base output directory (default: benchmarks/results/)",
    )

    # "dimensionality" - run dimensionality recovery benchmarks
    dim_parser = subs.add_parser(
        "dimensionality", help="Dimensionality recovery benchmarks"
    )
    dim_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    dim_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Base output directory (default: benchmarks/results/)",
    )


def run(args: argparse.Namespace, output_dir: Path) -> None:
    """Execute GFA benchmark command based on parsed args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with `command` and related options.
    output_dir : Path
        Directory for benchmark output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "all":
        run_all_benchmarks(args.workers, output_dir)
    elif args.command == "runtime":
        sweeps = _RUNTIME_SWEEPS if args.all else [args.sweep]
        run_runtime(sweeps, args.workers, output_dir)
    elif args.command == "recovery":
        sweeps = _RECOVERY_SWEEPS if args.all else [args.sweep]
        run_recovery(sweeps, args.workers, output_dir)
    elif args.command == "dimensionality":
        run_dimensionality(args.workers, output_dir)


def run_all_benchmarks(workers: int, output_dir: Path) -> None:
    """Run all GFA benchmarks (runtime + recovery + dimensionality).

    Parameters
    ----------
    workers : int
        Number of parallel workers for joblib.
    output_dir : Path
        Directory for benchmark output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    run_runtime(_RUNTIME_SWEEPS, workers, output_dir)
    run_recovery(_RECOVERY_SWEEPS, workers, output_dir)
    run_dimensionality(workers, output_dir)


def run_runtime(sweeps: list[str], workers: int, output_dir: Path) -> None:
    """Run runtime benchmarks for the specified sweeps.

    Parameters
    ----------
    sweeps : list of str
        Sweep names to run (e.g., ``["n_samples", "x_dim"]``).
    workers : int
        Number of parallel workers for joblib.
    output_dir : Path
        Directory for benchmark output files.
    """
    for sweep_name in sweeps:
        _run_runtime_sweep(sweep_name, workers, output_dir)


def run_recovery(sweeps: list[str], workers: int, output_dir: Path) -> None:
    """Run parameter recovery benchmarks for the specified sweeps.

    Parameters
    ----------
    sweeps : list of str
        Sweep names to run (e.g., ``["n_samples", "snr"]``).
    workers : int
        Number of parallel workers for joblib.
    output_dir : Path
        Directory for benchmark output files.
    """
    for sweep_name in sweeps:
        _run_recovery_sweep(sweep_name, workers, output_dir)


def run_dimensionality(workers: int, output_dir: Path) -> None:
    """Run dimensionality recovery benchmark (n_samples x SNR grid).

    Parameters
    ----------
    workers : int
        Number of parallel workers for joblib.
    output_dir : Path
        Directory for benchmark output files.
    """
    config = DIMENSIONALITY_CONFIG

    logger.info(f"Running dimensionality benchmark with {workers} workers")
    logger.info(
        f"Config: {config.n_runs} runs, "
        f"n_samples={config.n_samples_values}, snr={config.snr_values}"
    )

    # Parallelize over runs
    results_and_warnings = list(
        tqdm(
            Parallel(n_jobs=workers, return_as="generator")(
                delayed(dimensionality.run_single_run)(config, run_idx)
                for run_idx in range(config.n_runs)
            ),
            total=config.n_runs,
            desc="dimensionality",
        )
    )

    all_results = []
    all_warnings = []
    for results, warnings in results_and_warnings:
        all_results.extend(results)
        all_warnings.extend(warnings)

    if all_warnings:
        logger.warning(f"{len(all_warnings)} fits had issues:")
        for w in all_warnings:
            logger.warning(
                f"  n_samples={w['n_samples']}, snr={w['snr']}, "
                f"run={w['run_idx']}: {w['issues']}"
            )

    df = dimensionality.aggregate_results(all_results)
    dim_dir = output_dir / "dimensionality"
    dim_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dim_dir / "n_samples_snr.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _run_runtime_sweep(sweep_name: str, n_workers: int, output_dir: Path) -> None:
    """Run a single runtime benchmark sweep."""
    config = RUNTIME_CONFIG
    sweep_values = config.get_sweep_config(sweep_name).values

    logger.info(f"Running runtime/{sweep_name} sweep with {n_workers} workers")
    logger.info(f"Config: {config.n_runs} runs, values={sweep_values}")

    # Parallelize over runs with progress bar
    results_and_warnings = list(
        tqdm(
            Parallel(n_jobs=n_workers, return_as="generator")(
                delayed(runtime.run_single_run)(config, sweep_name, run_idx)
                for run_idx in range(config.n_runs)
            ),
            total=config.n_runs,
            desc=f"runtime/{sweep_name}",
        )
    )

    # Separate results and warnings
    all_results = []
    all_warnings = []
    for results, warnings in results_and_warnings:
        all_results.extend(results)
        all_warnings.extend(warnings)

    # Log warnings if any
    if all_warnings:
        logger.warning(f"{len(all_warnings)} fits had issues:")
        for w in all_warnings:
            sweep_val = f"{w['sweep_name']}={w['sweep_value']}"
            logger.warning(f"  {sweep_val}, run={w['run_idx']}: {w['issues']}")

    # Aggregate and save to runtime/ subdirectory
    df = runtime.aggregate_results(all_results)
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runtime_dir / f"{sweep_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")


def _run_recovery_sweep(sweep_name: str, n_workers: int, output_dir: Path) -> None:
    """Run a single parameter recovery benchmark sweep."""
    config = RECOVERY_CONFIG
    sweep_values = config.get_sweep_config(sweep_name).values

    logger.info(f"Running recovery/{sweep_name} sweep with {n_workers} workers")
    logger.info(f"Config: {config.n_runs} runs, values={sweep_values}")

    # Parallelize over runs with progress bar
    results_and_warnings = list(
        tqdm(
            Parallel(n_jobs=n_workers, return_as="generator")(
                delayed(recovery.run_single_run)(config, sweep_name, run_idx)
                for run_idx in range(config.n_runs)
            ),
            total=config.n_runs,
            desc=f"recovery/{sweep_name}",
        )
    )

    # Separate results and warnings
    all_results = []
    all_warnings = []
    for results, warnings in results_and_warnings:
        all_results.extend(results)
        all_warnings.extend(warnings)

    # Log warnings if any
    if all_warnings:
        logger.warning(f"{len(all_warnings)} fits had issues:")
        for w in all_warnings:
            sweep_val = f"{w['sweep_name']}={w['sweep_value']}"
            logger.warning(f"  {sweep_val}, run={w['run_idx']}: {w['issues']}")

    # Aggregate and save to recovery/ subdirectory
    df = recovery.aggregate_results(all_results)
    recovery_dir = output_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    csv_path = recovery_dir / f"{sweep_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
