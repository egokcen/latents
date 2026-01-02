"""Base classes for model fitting infrastructure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from latents._core.base import ArrayContainer


class FitTracker(ArrayContainer):
    """
    A class for quantities tracked during a model fit.

    Parameters
    ----------
    lb
        `ndarray` of `float`, shape ``(num_iter,)``.
        Variational lower bound at each iteration.
    iter_time
        `ndarray` of `float`, shape ``(num_iter,)``.
        Runtime on each iteration.

    Attributes
    ----------
    lb
        Same as **lb**, above.
    iter_time
        Same as **iter_time**, above.
    """

    def __init__(
        self,
        lb: np.ndarray | None = None,
        iter_time: np.ndarray | None = None,
    ):
        self.lb = lb
        self.iter_time = iter_time

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


class FitFlags:
    """
    A class for status messages during a model fit.

    Parameters
    ----------
    converged
        ``True`` if the lower bound converged before reaching ``max_iter``
        iterations.
    decreasing_lb
        ``True`` if lower bound decreased during fitting.
    private_var_floor
        ``True`` if the private variance floor was used on any values of
        ``phi``.

    Attributes
    ----------
    converged
        Same as **converged**, above.
    decreasing_lb
        Same as **decreasing_lb**, above.
    private_var_floor
        Same as **private_var_floor**, above.
    """

    def __init__(
        self,
        converged: bool = False,
        decreasing_lb: bool = False,
        private_var_floor: bool = False,
    ):
        self.converged = converged
        self.decreasing_lb = decreasing_lb
        self.private_var_floor = private_var_floor

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"converged={self.converged}, "
            f"decreasing_lb={self.decreasing_lb}, "
            f"private_var_floor={self.private_var_floor})"
        )

    def display(self) -> None:
        """Print out the fit flags."""
        print(f"Converged: {self.converged}")
        print(f"Decreasing lower bound: {self.decreasing_lb}")
        print(f"Private variance floor: {self.private_var_floor}")
