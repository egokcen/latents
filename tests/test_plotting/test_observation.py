"""Test observation model plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from latents.plotting.observation import (
    plot_dims_pairs,
    plot_dimensionalities,
    plot_var_exp,
    plot_var_exp_pairs,
)

# Minimal data: 2 groups, 4 dim types (2^2)
N_GROUPS = 2
N_DIM_TYPES = 4
DIM_TYPES = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # (n_groups, n_dim_types)
NUM_DIM = np.array([0, 1, 1, 2])  # (n_dim_types,)
VAR_EXP = np.array([[0.0, 0.2, 0.3, 0.5], [0.0, 0.4, 0.1, 0.5]])

# Pairwise data: 1 pair for 2 groups
PAIR_DIMS = np.array([[4, 2, 3]])  # (1, 3)
PAIR_VAR_EXP = np.array([[0.5, 0.5]])  # (1, 2)
PAIRS = np.array([[0, 1]])  # (1, 2)


class TestPlotDimensionalities:
    """Smoke tests for plot_dimensionalities()."""

    def test_basic(self):
        """Runs without error with minimal data."""
        _fig, ax = plt.subplots()
        plot_dimensionalities(NUM_DIM, DIM_TYPES, ax=ax)

    def test_with_group_names(self):
        """Custom group names are accepted."""
        _fig, ax = plt.subplots()
        plot_dimensionalities(NUM_DIM, DIM_TYPES, group_names=["A", "B"], ax=ax)

    def test_with_sem(self):
        """Error bars via sem_dim are accepted."""
        _fig, ax = plt.subplots()
        sem = np.ones(N_DIM_TYPES) * 0.1
        plot_dimensionalities(NUM_DIM, DIM_TYPES, sem_dim=sem, ax=ax)

    def test_plot_zero_dim(self):
        """Including zero-cardinality types doesn't error."""
        _fig, ax = plt.subplots()
        plot_dimensionalities(NUM_DIM, DIM_TYPES, plot_zero_dim=True, ax=ax)


class TestPlotVarExp:
    """Smoke tests for plot_var_exp()."""

    def test_basic(self):
        """Runs without error with minimal data."""
        fig = plt.figure()
        plot_var_exp(VAR_EXP, DIM_TYPES, fig=fig)

    def test_with_group_names(self):
        """Custom group names are accepted."""
        fig = plt.figure()
        plot_var_exp(VAR_EXP, DIM_TYPES, group_names=["A", "B"], fig=fig)


class TestPlotDimsPairs:
    """Smoke tests for plot_dims_pairs()."""

    def test_basic(self):
        """Runs without error with minimal data."""
        fig = plt.figure()
        plot_dims_pairs(PAIR_DIMS, PAIRS, n_groups=N_GROUPS, fig=fig)

    def test_with_group_names(self):
        """Custom group names are accepted."""
        fig = plt.figure()
        plot_dims_pairs(
            PAIR_DIMS, PAIRS, n_groups=N_GROUPS, group_names=["A", "B"], fig=fig
        )


class TestPlotVarExpPairs:
    """Smoke tests for plot_var_exp_pairs()."""

    def test_basic(self):
        """Runs without error with minimal data."""
        fig = plt.figure()
        plot_var_exp_pairs(PAIR_VAR_EXP, PAIRS, n_groups=N_GROUPS, fig=fig)

    def test_with_group_names(self):
        """Custom group names are accepted."""
        fig = plt.figure()
        plot_var_exp_pairs(
            PAIR_VAR_EXP, PAIRS, n_groups=N_GROUPS, group_names=["A", "B"], fig=fig
        )
