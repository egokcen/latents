"""Base classes for fit tracking infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from latents.base import ArrayContainer


class FitTracker(ArrayContainer):
    """Base class for quantities tracked during a model fit.

    Parameters
    ----------
    lb : ndarray of float, shape (num_iter,) or None, default None
        Variational lower bound at each iteration.
    iter_time : ndarray of float, shape (num_iter,) or None, default None
        Runtime on each iteration.
    lb_base : float or None, default None
        Baseline lower bound for convergence checking. Set during initial
        iterations of a fresh fit and preserved during resume.

    Attributes
    ----------
    lb : ndarray of float, shape (num_iter,) or None
        Variational lower bound at each iteration.
    iter_time : ndarray of float, shape (num_iter,) or None
        Runtime on each iteration.
    lb_base : float or None
        Baseline lower bound for convergence checking.
    """

    def __init__(
        self,
        lb: np.ndarray | None = None,
        iter_time: np.ndarray | None = None,
        lb_base: float | None = None,
    ):
        self.lb = lb
        self.iter_time = iter_time
        self.lb_base = lb_base

    def plot_lb(self) -> None:
        """Plot the variational lower bound each iteration."""
        if self.lb is not None:
            # create figure
            fig, ax_lb = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3))

            # Plot the lower bound. It should be monotonically increasing.
            ax_lb.plot(self.lb, color="black", linestyle="solid", linewidth=1.0)
            ax_lb.set_xlabel("Iteration")
            ax_lb.set_ylabel("Lower bound")
            ax_lb.spines["top"].set_visible(False)
            ax_lb.spines["right"].set_visible(False)

            fig.tight_layout()
            plt.show()
        else:
            print("No lower bound to plot.")

    def plot_runtime(self) -> None:
        """Plot the runtime at each iteration."""
        if self.iter_time is not None:
            # create figure
            fig, ax_rt = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3))

            # Plot cumulative runtime.
            ax_rt.plot(
                np.cumsum(self.iter_time),
                color="black",
                linestyle="solid",
                linewidth=1.0,
            )
            ax_rt.set_xlabel("Iteration")
            ax_rt.set_ylabel("Cumulative runtime (s)")
            ax_rt.spines["top"].set_visible(False)
            ax_rt.spines["right"].set_visible(False)

            fig.tight_layout()
            plt.show()
        else:
            print("No runtime to plot.")


@dataclass
class FitFlags:
    """Status flags from a model fit.

    Parameters
    ----------
    converged : bool, default False
        True if the lower bound converged before reaching max_iter.
    decreasing_lb : bool, default False
        True if lower bound decreased during fitting.
    private_var_floor : bool, default False
        True if the private variance floor was used on any values of phi.
    """

    converged: bool = False
    decreasing_lb: bool = False
    private_var_floor: bool = False

    def display(self) -> None:
        """Print the fit flags."""
        print(f"Converged: {self.converged}")
        print(f"Decreasing lower bound: {self.decreasing_lb}")
        print(f"Private variance floor: {self.private_var_floor}")
