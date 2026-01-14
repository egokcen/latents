"""Logging infrastructure for latents package.

This module provides structured event logging. Users configure the logger
to see output:

    import logging
    logging.basicConfig(level=logging.INFO)

Or for more control:

    logger = logging.getLogger("latents")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler("fit.log"))
"""

from __future__ import annotations

import logging
from typing import Any

# Package-level logger - users configure handlers to see output
logger = logging.getLogger("latents")


class FitEvent:
    """Event name constants for structured logging."""

    STARTED = "fit.started"
    CONVERGED = "fit.converged"
    MAX_ITER = "fit.max_iter"
    NO_LATENTS = "fit.no_latents"
    X_DIM_PRUNED = "fit.x_dim_pruned"
    FLAG_CHANGED = "fit.flag_changed"
    CHECKPOINT_SAVED = "fit.checkpoint_saved"
    INTERRUPTED = "fit.interrupted"


def log_event(event: str, level: int = logging.INFO, **context: Any) -> None:
    """Log a structured event with context.

    Parameters
    ----------
    event
        Event name (use FitEvent constants).
    level
        Logging level (DEBUG, INFO, WARNING, ERROR).
    **context
        Key-value pairs included in the log message. None values are omitted.

    Examples
    --------
    >>> log_event(FitEvent.STARTED, n_samples=1000, x_dim=10)
    >>> log_event(FitEvent.FLAG_CHANGED, level=logging.WARNING,
    ...           flag="decreasing_lb", value=True, iteration=150)
    """
    if context:
        # Filter out None values
        ctx_str = "; ".join(f"{k}={v}" for k, v in context.items() if v is not None)
        msg = f"{event} [{ctx_str}]" if ctx_str else event
    else:
        msg = event
    logger.log(level, msg)
