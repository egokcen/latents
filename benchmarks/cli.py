"""Command-line interface for running benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.gfa import cli as gfa_cli
from benchmarks.system_info import save_system_info

_LOG_DIR = Path(__file__).parent / "logs"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

# Configure parent logger so all benchmarks.* loggers inherit handlers
logger = logging.getLogger("benchmarks")


def _setup_logging() -> None:
    """Configure logging to console and file."""
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    # File handler (in benchmarks/logs/)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOG_DIR / f"benchmark_{timestamp}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_path}")


def main() -> int:
    """Run the benchmark CLI."""
    parser = argparse.ArgumentParser(description="Run latents benchmarks")
    subparsers = parser.add_subparsers(dest="method", required=True)

    # Top-level "all" command - run all methods
    all_parser = subparsers.add_parser("all", help="Run all benchmarks (all methods)")
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

    # Register method CLIs
    gfa_cli.register(subparsers)

    args = parser.parse_args()

    _setup_logging()

    # Resolve output directory
    output_dir = args.output or _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save system info once
    system_info_path = output_dir / "system_info.json"
    save_system_info(system_info_path)
    logger.info(f"Saved system info to {system_info_path}")

    # Dispatch
    if args.method == "all":
        gfa_cli.run_all_benchmarks(args.workers, output_dir / "gfa")
    elif args.method == "gfa":
        gfa_cli.run(args, output_dir / "gfa")

    return 0


if __name__ == "__main__":
    sys.exit(main())
