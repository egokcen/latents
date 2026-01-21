"""
Production workflows
====================

This example demonstrates callbacks, checkpointing, and serialization
features for production use with long-running analyses.
"""

# %%
# Imports
# -------

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

from latents.callbacks import CheckpointCallback, LoggingCallback, ProgressCallback
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.gfa.simulation import load_simulation, save_simulation, simulate
from latents.observation import ObsParamsHyperPrior

# Configure logging to stdout (sphinx-gallery captures stdout, not stderr)
# force=True ensures fresh config even if another example ran first
logging.basicConfig(
    level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
)

# %%
# Setup: simulate data
# --------------------
#
# We use a simple simulation for demonstration. See
# :ref:`sphx_glr_auto_examples_gfa_plot_1_simulation.py` for details on the
# generative model.

sim_config = GFASimConfig(
    n_samples=50,
    y_dims=np.array([8, 8]),
    x_dim=3,
    snr=1.0,
    random_seed=0,
)
hyperprior = ObsParamsHyperPrior(
    a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
)
sim_result = simulate(sim_config, hyperprior)
Y = sim_result.observations

# %%
# Callback overview
# -----------------
#
# Callbacks allow custom actions at specific points during fitting:
#
# - ``on_fit_start(ctx)`` — called after initialization
# - ``on_fit_end(ctx, reason)`` — called when fitting completes
# - ``on_iteration_end(ctx, iteration, lb, lb_prev)`` — called each iteration
# - ``on_flag_changed(ctx, flag, value, iteration)`` — called on status changes
# - ``on_x_dim_pruned(ctx, n_removed, x_dim_remaining, iteration)`` — called when ARD
#   prunes
#
# Implement any subset of these methods (duck typing). Three callbacks are
# provided: :class:`~latents.callbacks.ProgressCallback`,
# :class:`~latents.callbacks.LoggingCallback`, and
# :class:`~latents.callbacks.CheckpointCallback`.

# %%
# ProgressCallback
# ----------------
#
# :class:`~latents.callbacks.ProgressCallback` displays a tqdm progress bar
# during fitting. Best for interactive terminal sessions.

progress_cb = ProgressCallback(desc="Fitting GFA")
print(f"ProgressCallback: {progress_cb}")

# %%
# .. note::
#
#    ProgressCallback works best in terminals. In notebooks and documentation
#    builds, tqdm output can be verbose. We don't run it in this example.

# %%
# LoggingCallback
# ---------------
#
# :class:`~latents.callbacks.LoggingCallback` emits structured log events
# to Python's logging system. Configure logging to see output (done in imports).

config = GFAFitConfig(
    x_dim_init=5,
    fit_tol=1e-6,
    max_iter=500,
    random_seed=0,
    prune_x=True,
)

model = GFAModel(config=config)
model.fit(Y, callbacks=[LoggingCallback()])

# %%
# Log output shows fit start, dimension pruning events, and convergence.
# For debugging, set ``level=logging.DEBUG`` to see more detail.

# %%
# CheckpointCallback
# ------------------
#
# :class:`~latents.callbacks.CheckpointCallback` saves model state periodically
# during fitting. Useful for long runs where you want recovery from interruption.
#
# We use a temporary directory for this example.

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_dir = Path(tmpdir)

    checkpoint_cb = CheckpointCallback(
        save_dir=checkpoint_dir,
        every_n_iter=100,  # Save every 100 iterations
        save_initial=True,  # Save before iteration 0
        save_final=True,  # Save after convergence
        max_checkpoints=2,  # Keep only 2 periodic checkpoints
    )

    model2 = GFAModel(config=config)
    model2.fit(Y, callbacks=[LoggingCallback(), checkpoint_cb])

    # List saved checkpoints
    checkpoints = sorted(checkpoint_dir.glob("*.safetensors"))
    print(f"\nCheckpoints saved to {checkpoint_dir}:")
    for cp in checkpoints:
        print(f"  {cp.name}")

# %%
# The ``max_checkpoints`` parameter limits disk usage by deleting older
# periodic checkpoints. Initial, final, and interrupt checkpoints are
# always kept.
#
# For interrupt handling, ``CheckpointCallback`` installs a SIGINT handler
# that saves state before exiting when you press Ctrl+C.

# %%
# Combining callbacks
# -------------------
#
# Multiple callbacks can be used together. They execute in list order.

with tempfile.TemporaryDirectory() as tmpdir:
    callbacks = [
        LoggingCallback(),
        CheckpointCallback(save_dir=tmpdir, every_n_iter=200),
    ]

    model3 = GFAModel(config=config)
    model3.fit(Y, callbacks=callbacks)

# %%
# Model serialization
# -------------------
#
# Save fitted models with :meth:`~latents.gfa.GFAModel.save` and reload with
# :meth:`~latents.gfa.GFAModel.load`. Uses safetensors format for security
# (no arbitrary code execution on load).

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "model.safetensors"

    # Save the fitted model
    model.save(model_path)
    print(f"Model saved to {model_path.name}")

    # Load into a new instance
    loaded_model = GFAModel.load(model_path)

    # Verify posteriors match
    C_original = model.obs_posterior.C.mean
    C_loaded = loaded_model.obs_posterior.C.mean
    print(f"Posteriors match: {np.allclose(C_original, C_loaded)}")

    # Loaded model is ready for inference
    X_inferred = loaded_model.infer_latents(Y)
    print(f"Inferred latents shape: {X_inferred.mean.shape}")

# %%
# Resume interrupted fits
# -----------------------
#
# If a fit is interrupted (e.g., timeout, Ctrl+C with checkpoint), you can
# resume from a saved checkpoint with :meth:`~latents.gfa.GFAModel.resume_fit`.

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "partial.safetensors"

    # Fit with early stopping (simulate interruption)
    # save_x=True is required for resume_fit to work without manual recomputation
    short_config = GFAFitConfig(
        x_dim_init=5,
        fit_tol=1e-10,  # Very tight tolerance
        max_iter=50,  # Stop early
        random_seed=0,
        prune_x=True,
        save_x=True,
    )

    model_partial = GFAModel(config=short_config)
    model_partial.fit(Y, callbacks=[LoggingCallback()])
    model_partial.save(model_path)

    iterations_before = len(model_partial.tracker.lb)
    print(f"\nIterations before resume: {iterations_before}")
    print(f"Converged: {model_partial.flags.converged}")

    # Resume from checkpoint
    resumed = GFAModel.load(model_path)
    resumed.resume_fit(Y, max_iter=5000, callbacks=[LoggingCallback()])

    iterations_after = len(resumed.tracker.lb)
    print(f"Iterations after resume: {iterations_after}")
    print(f"Converged: {resumed.flags.converged}")

# %%
# The tracker appends iterations from the resumed fit, preserving the full
# optimization history.

# %%
# Simulation serialization
# ------------------------
#
# For reproducibility, save simulation results with
# :func:`~latents.gfa.simulation.save_simulation` and reload with
# :func:`~latents.gfa.simulation.load_simulation`.

with tempfile.TemporaryDirectory() as tmpdir:
    sim_path = Path(tmpdir) / "simulation.safetensors"

    # Save complete simulation (observations, latents, parameters)
    save_simulation(sim_path, sim_result)
    print(f"Simulation saved to {sim_path.name}")

    # Reload
    loaded_sim = load_simulation(sim_path)

    # Verify
    obs_match = np.allclose(sim_result.observations.data, loaded_sim.observations.data)
    print(f"Observations match: {obs_match}")
    latents_match = np.allclose(sim_result.latents.data, loaded_sim.latents.data)
    print(f"Latents match: {latents_match}")

# %%
# This saves the complete snapshot: config, hyperprior, sampled parameters,
# latents, and observations. For smaller files when you only need
# reproducibility, use :func:`~latents.gfa.simulation.save_simulation_recipe`
# which saves just the config and hyperprior (requires ``random_seed``).
