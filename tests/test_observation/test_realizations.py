"""Test observation model realization classes."""

from __future__ import annotations

import numpy as np

from latents.observation import ObsParamsPoint, ObsParamsRealization, adjust_snr

# --------------------------------------------------------------------------- #
# Shared test dimensions: 2 groups, y_dims=[3,2] (y_dim=5), x_dim=2
# --------------------------------------------------------------------------- #
Y_DIMS = np.array([3, 2])
Y_DIM = int(Y_DIMS.sum())  # 5
X_DIM = 2
N_GROUPS = len(Y_DIMS)


def _make_realization() -> ObsParamsRealization:
    """Small deterministic ObsParamsRealization for testing."""
    rng = np.random.default_rng(0)
    return ObsParamsRealization(
        C=rng.standard_normal((Y_DIM, X_DIM)),
        d=rng.standard_normal(Y_DIM),
        phi=np.abs(rng.standard_normal(Y_DIM)) + 0.1,
        alpha=np.abs(rng.standard_normal((N_GROUPS, X_DIM))) + 0.1,
        y_dims=Y_DIMS.copy(),
        x_dim=X_DIM,
    )


def _compute_snr(r: ObsParamsRealization) -> np.ndarray:
    """Compute SNR from a realization, mirroring ObsParamsPosterior.compute_snr.

    For point values, C moment per row is C_i C_i^T (no posterior covariance).
    SNR_g = trace(sum_i C_i C_i^T) / sum_i(1/phi_i)  for rows i in group g.
    """
    C_groups = np.split(r.C, np.cumsum(r.y_dims)[:-1])
    phi_groups = np.split(r.phi, np.cumsum(r.y_dims)[:-1])
    return np.array(
        [
            # trace(C_g^T C_g) = sum of squared Frobenius norms
            np.trace(C_groups[g].T @ C_groups[g]) / np.sum(1 / phi_groups[g])
            for g in range(r.n_groups)
        ]
    )


class TestObsParamsRealization:
    """Tests for ObsParamsRealization dataclass."""

    def test_properties(self):
        """Derived properties n_groups and y_dim are correct."""
        r = _make_realization()
        assert r.n_groups == N_GROUPS
        assert r.y_dim == Y_DIM

    def test_shapes(self):
        """All arrays have correct shapes."""
        r = _make_realization()
        assert r.C.shape == (Y_DIM, X_DIM)
        assert r.d.shape == (Y_DIM,)
        assert r.phi.shape == (Y_DIM,)
        assert r.alpha.shape == (N_GROUPS, X_DIM)


class TestObsParamsPoint:
    """Tests for ObsParamsPoint."""

    def test_properties(self):
        """Derived properties n_groups and y_dim are correct."""
        p = ObsParamsPoint(
            C=np.zeros((Y_DIM, X_DIM)),
            d=np.zeros(Y_DIM),
            phi=np.ones(Y_DIM),
            y_dims=Y_DIMS.copy(),
            x_dim=X_DIM,
        )
        assert p.n_groups == N_GROUPS
        assert p.y_dim == Y_DIM


class TestAdjustSnr:
    """Tests for the adjust_snr utility function."""

    def test_scalar_snr(self):
        """Scalar SNR target is applied to all groups."""
        r = _make_realization()
        target_snr = 5.0
        adjusted = adjust_snr(r, snr=target_snr)

        achieved = _compute_snr(adjusted)
        np.testing.assert_allclose(achieved, target_snr, rtol=1e-10)

    def test_per_group_snr(self):
        """Per-group SNR targets."""
        r = _make_realization()
        targets = np.array([2.0, 10.0])
        adjusted = adjust_snr(r, snr=targets)

        achieved = _compute_snr(adjusted)
        np.testing.assert_allclose(achieved, targets, rtol=1e-10)

    def test_original_unchanged(self):
        """The adjust_snr function returns a new realization; original is unmodified."""
        r = _make_realization()
        phi_before = r.phi.copy()
        _ = adjust_snr(r, snr=5.0)
        np.testing.assert_array_equal(r.phi, phi_before)

    def test_non_phi_params_preserved(self):
        """C, d, alpha are copied but not modified."""
        r = _make_realization()
        adjusted = adjust_snr(r, snr=5.0)
        np.testing.assert_array_equal(adjusted.C, r.C)
        np.testing.assert_array_equal(adjusted.d, r.d)
        np.testing.assert_array_equal(adjusted.alpha, r.alpha)
