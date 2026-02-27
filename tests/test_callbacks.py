"""Tests for the callback system."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from latents.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
    invoke_callbacks,
)
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.gfa.tracking import GFAFitContext, GFAFitFlags, GFAFitTracker
from latents.observation import (
    ARDPosterior,
    LoadingPosterior,
    ObsMeanPosterior,
    ObsParamsHyperPrior,
    ObsParamsPosterior,
    ObsPrecPosterior,
)
from latents.state import LatentsPosteriorStatic


# -----------------------------------------------------------------------------
# Synthetic fixture (no fitting required)
# -----------------------------------------------------------------------------


@pytest.fixture
def fit_context():
    """Synthetic GFAFitContext for callback unit tests.

    Constructs a minimal valid context without running model fitting.
    Callbacks only need shape info, config values, and a working save().
    """
    y_dims = np.array([3, 2])
    y_dim = int(y_dims.sum())
    x_dim = 2
    n_samples = 10

    config = GFAFitConfig(x_dim_init=x_dim, max_iter=100, prune_x=True)
    obs_hyperprior = ObsParamsHyperPrior()

    obs_posterior = ObsParamsPosterior(
        x_dim=x_dim,
        y_dims=y_dims,
        C=LoadingPosterior(
            mean=np.zeros((y_dim, x_dim)),
            cov=np.zeros((y_dim, x_dim, x_dim)),
            moment=np.zeros((y_dim, x_dim, x_dim)),
        ),
        alpha=ARDPosterior(
            a=np.ones(len(y_dims)),
            b=np.ones((len(y_dims), x_dim)),
            mean=np.ones((len(y_dims), x_dim)),
        ),
        d=ObsMeanPosterior(
            mean=np.zeros(y_dim),
            cov=np.ones(y_dim),
        ),
        phi=ObsPrecPosterior(
            a=1.0,
            b=np.ones(y_dim),
            mean=np.ones(y_dim),
        ),
    )

    latents_posterior = LatentsPosteriorStatic(
        mean=np.zeros((x_dim, n_samples)),
        cov=np.eye(x_dim),
        moment=n_samples * np.eye(x_dim),
    )

    tracker = GFAFitTracker(
        lb=np.arange(10, dtype=float),
        iter_time=np.ones(10),
        lb_base=0.0,
    )

    flags = GFAFitFlags()

    return GFAFitContext(
        config=config,
        obs_hyperprior=obs_hyperprior,
        obs_posterior=obs_posterior,
        latents_posterior=latents_posterior,
        tracker=tracker,
        flags=flags,
    )


# -----------------------------------------------------------------------------
# invoke_callbacks helper
# -----------------------------------------------------------------------------


class TestInvokeCallbacks:
    """Tests for the invoke_callbacks helper."""

    def test_calls_implemented_methods(self):
        """Callbacks with the method get called."""

        class MyCallback:
            def __init__(self):
                self.called = False

            def on_fit_start(self, ctx):
                self.called = True

        cb = MyCallback()
        invoke_callbacks([cb], "on_fit_start", ctx="test")
        assert cb.called

    def test_skips_unimplemented_methods(self):
        """Callbacks without the method are skipped (no error)."""

        class EmptyCallback:
            pass

        cb = EmptyCallback()
        # Should not raise
        invoke_callbacks([cb], "on_fit_start", ctx="test")

    def test_empty_list(self):
        """Empty callback list works."""
        invoke_callbacks([], "on_fit_start", ctx="test")


# -----------------------------------------------------------------------------
# LoggingCallback
# -----------------------------------------------------------------------------


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_logs_fit_start(self, fit_context, caplog):
        """on_fit_start logs the fit.started event with data shape info."""
        callback = LoggingCallback()

        with caplog.at_level(logging.INFO, logger="latents"):
            callback.on_fit_start(fit_context)

        assert "fit.started" in caplog.text
        assert "n_samples=" in caplog.text
        assert "y_dims=" in caplog.text
        assert "x_dim=" in caplog.text

    def test_logs_fit_end(self, fit_context, caplog):
        """on_fit_end logs the appropriate event."""
        callback = LoggingCallback()

        with caplog.at_level(logging.INFO, logger="latents"):
            callback.on_fit_end(fit_context, "converged")

        assert "fit.converged" in caplog.text

    def test_logs_x_dim_pruned(self, fit_context, caplog):
        """on_x_dim_pruned logs the pruning event with details."""
        callback = LoggingCallback()

        with caplog.at_level(logging.INFO, logger="latents"):
            callback.on_x_dim_pruned(
                fit_context, n_removed=2, x_dim_remaining=5, iteration=10
            )

        assert "fit.x_dim_pruned" in caplog.text
        assert "n_removed=2" in caplog.text
        assert "x_dim_remaining=5" in caplog.text
        # User-facing output is 1-indexed (input 10 -> output 11)
        assert "iteration=11" in caplog.text

    def test_skips_converged_flag(self, fit_context, caplog):
        """on_flag_changed skips converged flag (redundant with on_fit_end)."""
        callback = LoggingCallback()

        with caplog.at_level(logging.INFO, logger="latents"):
            callback.on_flag_changed(fit_context, "converged", True, iteration=100)

        # Should not log anything for converged flag
        assert "fit.flag_changed" not in caplog.text


# -----------------------------------------------------------------------------
# ProgressCallback
# -----------------------------------------------------------------------------


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_creates_and_closes_pbar(self, fit_context):
        """Progress bar is created on start and closed on end."""
        callback = ProgressCallback(desc="Test")

        callback.on_fit_start(fit_context)
        assert callback._pbar is not None

        callback.on_fit_end(fit_context, "converged")
        assert callback._pbar is None

    def test_tracks_x_dim_on_prune(self, fit_context):
        """x_dim is tracked when pruning occurs."""
        callback = ProgressCallback()

        callback.on_fit_start(fit_context)
        initial_x_dim = callback._x_dim

        callback.on_x_dim_pruned(
            fit_context, n_removed=2, x_dim_remaining=initial_x_dim - 2, iteration=5
        )
        assert callback._x_dim == initial_x_dim - 2

        callback.on_fit_end(fit_context, "converged")


# -----------------------------------------------------------------------------
# CheckpointCallback
# -----------------------------------------------------------------------------


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_creates_save_dir(self, tmp_path):
        """save_dir is created if it doesn't exist."""
        save_dir = tmp_path / "checkpoints"
        assert not save_dir.exists()

        CheckpointCallback(save_dir=save_dir)
        assert save_dir.exists()

    def test_saves_initial(self, fit_context, tmp_path):
        """Initial checkpoint is saved when save_initial=True."""
        callback = CheckpointCallback(
            save_dir=tmp_path,
            save_initial=True,
            save_final=False,
            save_on_interrupt=False,
        )

        callback.on_fit_start(fit_context)

        files = list(tmp_path.glob("*.safetensors"))
        assert len(files) == 1
        assert "init" in files[0].name

    def test_saves_final(self, fit_context, tmp_path):
        """Final checkpoint is saved when save_final=True."""
        callback = CheckpointCallback(
            save_dir=tmp_path,
            save_initial=False,
            save_final=True,
            save_on_interrupt=False,
        )

        callback.on_fit_start(fit_context)
        callback.on_fit_end(fit_context, "converged")

        files = list(tmp_path.glob("*.safetensors"))
        assert len(files) == 1
        assert "final" in files[0].name

    def test_saves_periodic(self, fit_context, tmp_path):
        """Periodic checkpoints are saved at correct intervals."""
        callback = CheckpointCallback(
            save_dir=tmp_path,
            every_n_iter=5,
            save_initial=False,
            save_final=False,
            save_on_interrupt=False,
            max_checkpoints=0,  # Keep all
        )

        callback.on_fit_start(fit_context)

        # Simulate 12 iterations (0-11)
        # Should save at iterations 4 (iter 5) and 9 (iter 10)
        for i in range(12):
            callback.on_iteration_end(fit_context, iteration=i, lb=0.0, lb_prev=0.0)

        files = list(tmp_path.glob("*.safetensors"))
        assert len(files) == 2

    def test_prunes_old_checkpoints(self, fit_context, tmp_path):
        """Old periodic checkpoints are deleted when max_checkpoints exceeded."""
        callback = CheckpointCallback(
            save_dir=tmp_path,
            every_n_iter=1,
            save_initial=False,
            save_final=False,
            save_on_interrupt=False,
            max_checkpoints=2,
        )

        callback.on_fit_start(fit_context)

        # Simulate 5 iterations - saves 5 checkpoints but keeps only 2
        for i in range(5):
            callback.on_iteration_end(fit_context, iteration=i, lb=0.0, lb_prev=0.0)

        files = list(tmp_path.glob("*.safetensors"))
        assert len(files) == 2

    def test_prefix_in_filename(self, fit_context, tmp_path):
        """Prefix is included in checkpoint filenames."""
        callback = CheckpointCallback(
            save_dir=tmp_path,
            save_initial=True,
            save_final=False,
            save_on_interrupt=False,
            prefix="myexperiment",
        )

        callback.on_fit_start(fit_context)

        files = list(tmp_path.glob("*.safetensors"))
        assert len(files) == 1
        assert "myexperiment" in files[0].name


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


@pytest.mark.fit
class TestCallbackIntegration:
    """Integration tests for callbacks with real model fitting."""

    @staticmethod
    def _simulate_small():
        """Generate small simulated data for integration tests."""
        import latents.gfa.simulation as gfa_sim

        hyperprior = ObsParamsHyperPrior(
            a_alpha=1.0, b_alpha=1.0, a_phi=1.0, b_phi=1.0, beta_d=1.0
        )
        sim_config = GFASimConfig(
            n_samples=50,
            y_dims=np.array([8, 8]),
            x_dim=3,
            snr=1.0,
            random_seed=42,
        )
        return gfa_sim.simulate(sim_config, hyperprior)

    def test_fit_with_callbacks(self):
        """Callbacks are invoked during model.fit()."""
        sim_result = self._simulate_small()

        # Custom callback that records events
        class RecordingCallback:
            def __init__(self):
                self.events = []

            def on_fit_start(self, ctx):
                self.events.append("fit_start")

            def on_fit_end(self, ctx, reason):
                self.events.append(f"fit_end:{reason}")

            def on_iteration_end(self, ctx, iteration, lb, lb_prev):
                self.events.append(f"iter:{iteration}")

        recorder = RecordingCallback()

        config = GFAFitConfig(
            x_dim_init=5,
            max_iter=10,
            random_seed=0,
        )
        model = GFAModel(config=config)
        model.fit(sim_result.observations, callbacks=[recorder])

        # Verify events were recorded
        assert "fit_start" in recorder.events
        assert any(e.startswith("fit_end:") for e in recorder.events)
        assert any(e.startswith("iter:") for e in recorder.events)

    def test_checkpoint_roundtrip(self, tmp_path):
        """Checkpoint files can be loaded as GFAModel."""
        sim_result = self._simulate_small()

        config = GFAFitConfig(
            x_dim_init=5,
            max_iter=50,
            random_seed=0,
            save_x=True,
            save_fit_progress=True,
        )

        checkpoint_cb = CheckpointCallback(
            save_dir=tmp_path,
            save_initial=False,
            save_final=True,
            save_on_interrupt=False,
        )

        model = GFAModel(config=config)
        model.fit(sim_result.observations, callbacks=[checkpoint_cb])

        # Find the checkpoint file
        files = list(tmp_path.glob("*final*.safetensors"))
        assert len(files) == 1

        # Load and verify
        loaded = GFAModel.load(files[0])
        assert loaded.obs_posterior is not None
        assert loaded.latents_posterior is not None

        # Loaded model can infer latents
        new_latents = loaded.infer_latents(sim_result.observations)
        n_samples = sim_result.observations.data.shape[1]
        assert new_latents.mean.shape[1] == n_samples
