"""Test Hinton diagram plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from latents.plotting import hinton_diagram


class TestHintonDiagram:
    """Smoke tests for hinton_diagram()."""

    def test_2d_matrix(self):
        """2D matrix produces a figure without error."""
        _fig, ax = plt.subplots()
        hinton_diagram(np.random.default_rng(0).standard_normal((5, 3)), ax=ax)
        assert len(ax.patches) == 15

    def test_1d_array(self):
        """1D array is treated as a column vector."""
        _fig, ax = plt.subplots()
        hinton_diagram(np.array([1.0, -2.0, 0.5]), ax=ax)
        assert len(ax.patches) == 3

    def test_no_ax(self):
        """Calling without ax uses the current axes."""
        plt.figure()
        hinton_diagram(np.eye(2))
        ax = plt.gca()
        assert len(ax.patches) == 4

    def test_custom_max_weight(self):
        """Custom max_weight is accepted without error."""
        _fig, ax = plt.subplots()
        hinton_diagram(np.ones((2, 2)), max_weight=5.0, ax=ax)
        assert len(ax.patches) == 4
