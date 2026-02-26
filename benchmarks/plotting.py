"""Shared plotting utilities for benchmark gallery scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_runtime_sweep(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ref_slope: float | None = None,
    ref_label: str | None = None,
) -> plt.Figure:
    """Plot runtime sweep results as 1x3 subplots.

    Creates a figure with three panels showing:

    1. Time per iteration vs sweep parameter
    2. Iterations to convergence vs sweep parameter
    3. Total runtime vs sweep parameter

    All plots use log-log scaling with shaded SEM bands.

    Parameters
    ----------
    df : DataFrame
        Sweep results with columns: sweep_value, mean_runtime_per_iter,
        sem_runtime_per_iter, mean_iters, sem_iters, mean_runtime, sem_runtime.
    title : str
        Overall figure title.
    xlabel : str
        Label for x-axis (the swept parameter).
    ref_slope : float or None, default None
        Slope for reference line on the time-per-iteration plot
        (e.g., 1 for O(N), 3 for O(K³)).
    ref_label : str or None, default None
        Label for reference line (e.g., "O(N)", "O(K³)").

    Returns
    -------
    Figure
        Matplotlib figure with 3 subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    x = df["sweep_value"].values

    # Column definitions: (mean_col, sem_col, ylabel)
    columns = [
        ("mean_runtime_per_iter", "sem_runtime_per_iter", "Time per iteration (s)"),
        ("mean_iters", "sem_iters", "Iterations to convergence"),
        ("mean_runtime", "sem_runtime", "Total runtime (s)"),
    ]

    for ax, (mean_col, sem_col, ylabel) in zip(axes, columns, strict=True):
        y = df[mean_col].values
        sem = df[sem_col].values

        # Main line with markers
        ax.plot(x, y, "o-", color="C0", linewidth=2, markersize=6, zorder=3)

        # Shaded SEM band
        ax.fill_between(x, y - sem, y + sem, color="C0", alpha=0.2, zorder=2)

        # Reference line (for time per iteration only, where theory applies directly)
        if ref_slope is not None and mean_col == "mean_runtime_per_iter":
            # Anchor reference line at first data point
            x_ref = np.array([x[0], x[-1]])
            y_ref = y[0] * (x_ref / x[0]) ** ref_slope
            ax.plot(
                x_ref,
                y_ref,
                "--",
                color="gray",
                linewidth=1.5,
                label=ref_label,
                zorder=1,
            )
            ax.legend(loc="upper left", frameon=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Clean style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_recovery_sweep(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    log_x: bool = True,
) -> plt.Figure:
    """Plot recovery sweep results on a single panel.

    All error metrics are overlaid with distinct colors and markers.
    Lower is better for all metrics (R² is inverted to 1 - R²).

    Parameters
    ----------
    df : DataFrame
        Recovery results with columns: sweep_value, plus _mean/_sem
        columns for each metric.
    title : str
        Overall figure title.
    xlabel : str
        Label for x-axis (the swept parameter).
    log_x : bool, default True
        Whether to use log scale for the x-axis.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    x = df["sweep_value"].values

    # Metric definitions: (metric_name, label, invert, marker)
    # invert=True plots 1 - value (for R² -> error)
    metrics = [
        ("C_subspace_error", "Subspace error (C)", False, "o"),
        ("d_error", "Rel. L2 error (d)", False, "s"),
        ("noise_var_error", "Rel. L2 error (noise var.)", False, "^"),
        ("ard_var_error", "Rel. L2 error (ARD var.)", False, "D"),
        ("denoised_r2", "Signal error (X)", True, "v"),
    ]

    for i, (metric, label, invert, marker) in enumerate(metrics):
        y = df[f"{metric}_mean"].values
        sem = df[f"{metric}_sem"].values

        if invert:
            y = 1.0 - y
            # SEM stays the same magnitude (linear transformation)

        color = f"C{i}"
        ax.plot(
            x,
            y,
            marker=marker,
            linestyle="-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label,
            zorder=3,
        )
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15, zorder=2)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    return fig


def plot_dimensionality(
    df: pd.DataFrame,
    title: str = "Dimensionality recovery",
) -> plt.Figure:
    """Plot dimensionality recovery across sample sizes and SNR levels.

    Single panel with n_samples on x-axis, separate curves per SNR value,
    and a horizontal reference line at 0 (perfect recovery).

    Parameters
    ----------
    df : DataFrame
        Aggregated results with columns: n_samples, snr,
        x_dim_error_mean, x_dim_error_sem.
    title : str, default "Dimensionality recovery"
        Figure title.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Reference line at 0 (perfect recovery)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)

    snr_values = sorted(df["snr"].unique())

    for i, snr in enumerate(snr_values):
        mask = df["snr"] == snr
        sub = df.loc[mask].sort_values("n_samples")
        x = sub["n_samples"].values
        y = sub["x_dim_error_mean"].values
        sem = sub["x_dim_error_sem"].values

        color = f"C{i}"
        ax.plot(
            x,
            y,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"SNR = {snr}",
            zorder=3,
        )
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Dimensionality error (est. - true)")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    return fig


def display_system_info(system_info_path: Path) -> None:
    """Display system information for reproducibility.

    Prints hardware, software, and OS details in a formatted block.

    Parameters
    ----------
    system_info_path : Path
        Path to system_info.json file.
    """
    with open(system_info_path) as f:
        info = json.load(f)

    print("Hardware")
    print("--------")
    print(f"CPU: {info['hardware']['cpu']}")
    print(
        f"Cores: {info['hardware']['cpu_cores_physical']} physical, "
        f"{info['hardware']['cpu_cores_logical']} logical"
    )
    print(f"RAM: {info['hardware']['ram_gb']:.1f} GB")
    print()
    print("Software")
    print("--------")
    print(f"Python: {info['software']['python']}")
    print(f"NumPy: {info['software']['numpy']}")
    print(f"SciPy: {info['software']['scipy']}")
    print(f"Latents: {info['software']['latents']}")
    print()
    print("Operating System")
    print("----------------")
    print(f"System: {info['os']['system']}")
    print(f"Distribution: {info['os']['distro']}")
    print(f"Kernel: {info['os']['release']}")
