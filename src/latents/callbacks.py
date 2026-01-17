"""Callbacks for model fitting.

Callbacks allow custom actions at specific points during model fitting.
Implement any subset of the callback methods you need (duck typing).

Callback Methods
----------------
on_fit_start(ctx)
    Called when fit() begins, after initialization.

on_fit_end(ctx, reason)
    Called when fit() completes. reason is one of:
    "converged", "max_iter", "no_latents".

on_iteration_end(ctx, iteration, lb, lb_prev)
    Called at the end of each EM iteration.

on_flag_changed(ctx, flag, value, iteration)
    Called when a boolean flag changes (converged, decreasing_lb,
    private_var_floor).

on_x_dim_pruned(ctx, n_removed, x_dim_remaining, iteration)
    Called when latent dimensions are pruned.

Examples
--------
>>> from latents.callbacks import ProgressCallback, CheckpointCallback
>>>
>>> model.fit(Y, callbacks=[
...     ProgressCallback(),
...     CheckpointCallback(save_dir="./checkpoints"),
... ])
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from latents._internal.logging import FitEvent, log_event

# -----------------------------------------------------------------------------
# Callback Invocation Helper
# -----------------------------------------------------------------------------


def invoke_callbacks(callbacks: list, method: str, **kwargs: Any) -> None:
    """Call a method on all callbacks that implement it.

    Parameters
    ----------
    callbacks : list
        List of callback objects.
    method : str
        Method name to call.
    **kwargs : Any
        Arguments passed to the method.
    """
    for cb in callbacks:
        if hasattr(cb, method):
            getattr(cb, method)(**kwargs)


# -----------------------------------------------------------------------------
# Logging Callback
# -----------------------------------------------------------------------------


@dataclass
class LoggingCallback:
    """Emit log events to the 'latents' logger during fitting.

    Configure Python logging to see output::

        import logging
        logging.basicConfig(level=logging.INFO)

    Or to log to a file::

        import logging
        logging.basicConfig(level=logging.INFO, filename="fit.log")
    """

    def on_fit_start(self, ctx: Any) -> None:
        """Log fit start event with data dimensions.

        Parameters
        ----------
        ctx : Any
            Fitting context providing obs_posterior, latents_posterior, tracker.
        """
        # Log data shape info: n_samples from latents, y_dims per group, x_dim
        n_samples = ctx.latents_posterior.mean.shape[1]
        y_dims = ctx.obs_posterior.y_dims
        x_dim = ctx.obs_posterior.x_dim
        log_event(FitEvent.STARTED, n_samples=n_samples, y_dims=y_dims, x_dim=x_dim)

    def on_fit_end(self, ctx: Any, reason: str) -> None:
        """Log fit end event with termination reason.

        Parameters
        ----------
        ctx : Any
            Fitting context providing tracker.
        reason : str
            Termination reason ("converged", "max_iter", or "no_latents").
        """
        event_map = {
            "converged": FitEvent.CONVERGED,
            "max_iter": FitEvent.MAX_ITER,
            "no_latents": FitEvent.NO_LATENTS,
        }
        event = event_map.get(reason, FitEvent.CONVERGED)

        # no_latents suggests data/config issue
        level = logging.WARNING if reason == "no_latents" else logging.INFO

        iteration = len(ctx.tracker.lb) if ctx.tracker.lb is not None else 0
        log_event(event, level=level, iteration=iteration)

    def on_flag_changed(self, ctx: Any, flag: str, value: Any, iteration: int) -> None:
        """Log flag change events (warnings for decreasing_lb, private_var_floor).

        Parameters
        ----------
        ctx : Any
            Fitting context (unused by this callback).
        flag : str
            Name of the flag that changed.
        value : Any
            New value of the flag.
        iteration : int
            Current iteration number.
        """
        # converged is redundant with on_fit_end
        if flag == "converged":
            return

        # decreasing_lb and private_var_floor are warnings
        log_event(
            FitEvent.FLAG_CHANGED,
            level=logging.WARNING,
            flag=flag,
            value=value,
            iteration=iteration,
        )

    def on_x_dim_pruned(
        self, ctx: Any, n_removed: int, x_dim_remaining: int, iteration: int
    ) -> None:
        """Log latent dimension pruning event.

        Parameters
        ----------
        ctx : Any
            Fitting context (unused by this callback).
        n_removed : int
            Number of dimensions removed.
        x_dim_remaining : int
            Number of dimensions remaining.
        iteration : int
            Current iteration number.
        """
        log_event(
            FitEvent.X_DIM_PRUNED,
            n_removed=n_removed,
            x_dim_remaining=x_dim_remaining,
            iteration=iteration,
        )


# -----------------------------------------------------------------------------
# Progress Callback
# -----------------------------------------------------------------------------


@dataclass
class ProgressCallback:
    """Display tqdm progress bar during fitting.

    Parameters
    ----------
    desc : str, default "Fitting"
        Description shown next to the progress bar.
    """

    desc: str = "Fitting"

    # Internal state (not set by user)
    _pbar: tqdm | None = field(default=None, init=False, repr=False)
    _x_dim: int = field(default=0, init=False, repr=False)
    _lb_base: float | None = field(default=None, init=False, repr=False)

    def on_fit_start(self, ctx: Any) -> None:
        """Initialize progress bar.

        Parameters
        ----------
        ctx : Any
            Fitting context providing config, obs_posterior, tracker.
        """
        max_iter = ctx.config.max_iter
        self._pbar = tqdm(total=max_iter, desc=self.desc)
        self._x_dim = ctx.obs_posterior.x_dim
        # Initialize from tracker (handles resume case)
        self._lb_base = ctx.tracker.lb_base

    def on_iteration_end(
        self, ctx: Any, iteration: int, lb: float, lb_prev: float
    ) -> None:
        """Update progress bar with current lower bound and relative change.

        Parameters
        ----------
        ctx : Any
            Fitting context providing config, tracker.
        iteration : int
            Current iteration number.
        lb : float
            Current lower bound value.
        lb_prev : float
            Previous iteration's lower bound value.
        """
        if self._pbar is None:
            return

        self._pbar.update(1)

        # Build postfix matching original implementation
        postfix: dict[str, Any] = {"lb": f"{lb:.2e}"}

        # Update lb_base during burn-in (matches main fit loop: overwrite on 0 and 1)
        # Only do this for fresh fits (when _lb_base was None at start)
        if iteration <= 1 and ctx.tracker.lb_base is None:
            # Fresh fit, still in burn-in - tracker hasn't set lb_base yet
            pass
        elif iteration <= 1:
            # Fresh fit, tracker just set lb_base - sync our copy
            self._lb_base = ctx.tracker.lb_base

        # Relative change (after burn-in, when denominator is non-zero)
        if self._lb_base is not None and iteration > 1:
            denom = lb_prev - self._lb_base
            if denom != 0.0:
                rel_change = (lb - lb_prev) / denom
                postfix["Δ"] = f"{rel_change:.1e}"

        if ctx.config.prune_x:
            postfix["x_dim"] = self._x_dim

        self._pbar.set_postfix(postfix)

    def on_x_dim_pruned(
        self, ctx: Any, n_removed: int, x_dim_remaining: int, iteration: int
    ) -> None:
        """Update tracked x_dim for progress bar display.

        Parameters
        ----------
        ctx : Any
            Fitting context (unused by this callback).
        n_removed : int
            Number of dimensions removed.
        x_dim_remaining : int
            Number of dimensions remaining.
        iteration : int
            Current iteration number.
        """
        self._x_dim = x_dim_remaining

    def on_fit_end(self, ctx: Any, reason: str) -> None:
        """Close progress bar.

        Parameters
        ----------
        ctx : Any
            Fitting context (unused by this callback).
        reason : str
            Termination reason.
        """
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# -----------------------------------------------------------------------------
# Checkpoint Callback
# -----------------------------------------------------------------------------


@dataclass
class CheckpointCallback:
    """Save model checkpoints during fitting.

    Checkpoints are saved in safetensors format and can be loaded via
    :meth:`GFAModel.load`.

    Parameters
    ----------
    save_dir : str or Path
        Directory for checkpoint files. Created if it doesn't exist.
    every_n_iter : int, default 5000
        Save every N iterations. Set to 0 to disable periodic saves.
    save_initial : bool, default True
        If True, save immediately after initialization (before iteration 0).
    save_final : bool, default True
        If True, save after fit completes.
    save_on_interrupt : bool, default True
        If True, save checkpoint when Ctrl+C is pressed. Only works in the
        main process; in parallel workers, rely on periodic checkpoints.
    max_checkpoints : int, default 3
        Maximum periodic checkpoints to keep. Older ones are deleted.
        Set to 0 to keep all. Does not affect initial, final, or interrupt
        checkpoints.
    prefix : str, default ""
        Optional prefix for checkpoint filenames.

    Examples
    --------
    >>> callback = CheckpointCallback(
    ...     save_dir="./checkpoints",
    ...     every_n_iter=5000,
    ...     prefix="experiment1",
    ... )
    >>> model.fit(Y, callbacks=[callback])

    # Produces files like:
    # ./checkpoints/experiment1_init.safetensors
    # ./checkpoints/experiment1_iter_005000.safetensors
    # ./checkpoints/experiment1_final.safetensors
    """

    save_dir: str | Path
    every_n_iter: int = 5000
    save_initial: bool = True
    save_final: bool = True
    save_on_interrupt: bool = True
    max_checkpoints: int = 3
    prefix: str = ""

    # Internal state
    _ctx: Any = field(default=None, init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)
    _periodic_paths: list[Path] = field(default_factory=list, init=False, repr=False)
    _original_sigint: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _is_main_process(self) -> bool:
        """Check if running in the main process."""
        return multiprocessing.current_process().name == "MainProcess"

    def on_fit_start(self, ctx: Any) -> None:
        """Initialize checkpointing, register interrupt handler, save initial.

        Parameters
        ----------
        ctx : Any
            Fitting context providing save() method.
        """
        self._ctx = ctx
        self._iteration = 0
        self._periodic_paths = []

        # Register interrupt handler (main process only)
        if self.save_on_interrupt:
            if self._is_main_process():
                self._original_sigint = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handle_interrupt)
            else:
                # Warn user that interrupt handling won't work in workers
                warnings.warn(
                    "save_on_interrupt has no effect in worker processes; "
                    "rely on periodic checkpoints (every_n_iter) for parallel fits",
                    UserWarning,
                    stacklevel=2,
                )

        if self.save_initial:
            self._save("init")

    def on_iteration_end(
        self, ctx: Any, iteration: int, lb: float, lb_prev: float
    ) -> None:
        """Save periodic checkpoint if iteration matches every_n_iter.

        Parameters
        ----------
        ctx : Any
            Fitting context (unused, uses stored context).
        iteration : int
            Current iteration number.
        lb : float
            Current lower bound value (unused).
        lb_prev : float
            Previous lower bound value (unused).
        """
        self._iteration = iteration

        # Periodic checkpoint (iteration is 0-indexed, so add 1 for display)
        iter_num = iteration + 1
        if self.every_n_iter > 0 and iter_num % self.every_n_iter == 0:
            path = self._save(f"iter_{iter_num:06d}")
            self._periodic_paths.append(path)
            self._prune_old_checkpoints()

    def on_fit_end(self, ctx: Any, reason: str) -> None:
        """Save final checkpoint and restore signal handler.

        Parameters
        ----------
        ctx : Any
            Fitting context (unused, uses stored context).
        reason : str
            Termination reason (unused).
        """
        if self.save_final:
            self._save("final")
        self._restore_signal_handler()

    def _save(self, suffix: str) -> Path:
        """Save checkpoint and log the event."""
        if self.prefix:
            filename = f"{self.prefix}_{suffix}.safetensors"
        else:
            filename = f"checkpoint_{suffix}.safetensors"

        path = self.save_dir / filename
        self._ctx.save(path)

        log_event(
            FitEvent.CHECKPOINT_SAVED,
            path=str(path),
            iteration=self._iteration,
        )
        return path

    def _prune_old_checkpoints(self) -> None:
        """Delete oldest periodic checkpoints if we exceed max_checkpoints."""
        if self.max_checkpoints <= 0:
            return

        while len(self._periodic_paths) > self.max_checkpoints:
            oldest = self._periodic_paths.pop(0)
            if oldest.exists():
                oldest.unlink()

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle Ctrl+C: save checkpoint then exit."""
        log_event(
            FitEvent.INTERRUPTED,
            level=logging.WARNING,
            iteration=self._iteration,
        )
        self._save(f"interrupted_{self._iteration + 1:06d}")
        self._restore_signal_handler()
        sys.exit(130)  # Standard exit code for SIGINT

    def _restore_signal_handler(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
