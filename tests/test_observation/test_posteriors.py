"""Test observation model posterior classes."""

from __future__ import annotations

import numpy as np
import pytest

from latents.observation import (
    ARDPosterior,
    LoadingPosterior,
    ObsMeanPosterior,
    ObsParamsPosterior,
    ObsPrecPosterior,
)
from latents.observation.realizations import ObsParamsRealization

# --------------------------------------------------------------------------- #
# Shared test dimensions: 2 groups, y_dims=[3,2] (y_dim=5), x_dim=2
# --------------------------------------------------------------------------- #
Y_DIMS = np.array([3, 2])
Y_DIM = int(Y_DIMS.sum())  # 5
X_DIM = 2
N_GROUPS = len(Y_DIMS)


# -- Helpers ---------------------------------------------------------------- #


def _make_loading_posterior() -> LoadingPosterior:
    """Small deterministic LoadingPosterior for testing."""
    rng = np.random.default_rng(0)
    mean = rng.standard_normal((Y_DIM, X_DIM))
    cov = np.zeros((Y_DIM, X_DIM, X_DIM))
    for i in range(Y_DIM):
        cov[i] = np.eye(X_DIM) * 0.1
    # moment = outer(mean) + cov, computed lazily
    moment = np.einsum("ij,ik->ijk", mean, mean) + cov
    return LoadingPosterior(mean=mean, cov=cov, moment=moment)


def _make_ard_posterior() -> ARDPosterior:
    """Small deterministic ARDPosterior for testing."""
    a = np.array([10.0, 20.0])  # (n_groups,)
    b = np.array([[2.0, 4.0], [3.0, 6.0]])  # (n_groups, x_dim)
    mean = a[:, np.newaxis] / b  # expected mean
    return ARDPosterior(a=a, b=b, mean=mean)


def _make_obs_mean_posterior() -> ObsMeanPosterior:
    return ObsMeanPosterior(
        mean=np.arange(Y_DIM, dtype=float),
        cov=np.ones(Y_DIM) * 0.5,
    )


def _make_obs_prec_posterior() -> ObsPrecPosterior:
    return ObsPrecPosterior(
        a=5.0,
        b=np.arange(1, Y_DIM + 1, dtype=float),
        mean=5.0 / np.arange(1, Y_DIM + 1, dtype=float),
    )


def _make_obs_params_posterior() -> ObsParamsPosterior:
    """Full ObsParamsPosterior with all sub-posteriors initialized."""
    return ObsParamsPosterior(
        x_dim=X_DIM,
        y_dims=Y_DIMS.copy(),
        C=_make_loading_posterior(),
        alpha=_make_ard_posterior(),
        d=_make_obs_mean_posterior(),
        phi=_make_obs_prec_posterior(),
    )


# =========================================================================== #
# LoadingPosterior
# =========================================================================== #


class TestLoadingPosterior:
    """Tests for LoadingPosterior dataclass."""

    def test_init_defaults(self):
        """All fields default to None."""
        lp = LoadingPosterior()
        assert lp.mean is None
        assert lp.cov is None
        assert lp.moment is None

    def test_init_with_arrays(self):
        """Arrays are stored with correct shapes."""
        lp = _make_loading_posterior()
        assert lp.mean.shape == (Y_DIM, X_DIM)
        assert lp.cov.shape == (Y_DIM, X_DIM, X_DIM)
        assert lp.moment.shape == (Y_DIM, X_DIM, X_DIM)

    def test_type_validation(self):
        """Non-ndarray inputs are rejected."""
        with pytest.raises(TypeError, match=r"mean must be a numpy\.ndarray"):
            LoadingPosterior(mean=[[1, 2]])

    def test_compute_moment_in_place(self):
        """In-place moment computation stores result on the object."""
        lp = _make_loading_posterior()
        expected = np.einsum("ij,ik->ijk", lp.mean, lp.mean) + lp.cov
        result = lp.compute_moment(in_place=True)
        np.testing.assert_allclose(result, expected)
        # Verify it's stored in place
        assert result is lp.moment

    def test_compute_moment_copy(self):
        """Copy mode returns a new array without modifying the object."""
        lp = _make_loading_posterior()
        result = lp.compute_moment(in_place=False)
        expected = np.einsum("ij,ik->ijk", lp.mean, lp.mean) + lp.cov
        np.testing.assert_allclose(result, expected)
        # Verify result is not stored in place
        assert result is not lp.moment

    def test_get_groups(self):
        """Splitting along y_dim produces correctly shaped per-group arrays."""
        lp = _make_loading_posterior()
        means, covs, moments = lp.get_groups(Y_DIMS)
        assert len(means) == N_GROUPS
        assert means[0].shape == (3, X_DIM)
        assert means[1].shape == (2, X_DIM)
        assert covs[0].shape == (3, X_DIM, X_DIM)
        assert moments[1].shape == (2, X_DIM, X_DIM)

    def test_get_groups_dim_mismatch(self):
        """Dimension mismatch raises ValueError."""
        lp = _make_loading_posterior()
        with pytest.raises(ValueError, match="sum of y_dims"):
            lp.get_groups(np.array([10, 10]))

    def test_compute_squared_norms(self):
        """Squared norms are non-negative with correct shape."""
        lp = _make_loading_posterior()
        norms = lp.compute_squared_norms(Y_DIMS)
        assert norms.shape == (N_GROUPS, X_DIM)
        # Squared norms are traces of moment matrices per group/dim, so >= 0
        assert np.all(norms >= 0)

    def test_get_subset_dims_in_place(self):
        """In-place subsetting reduces x_dim on the object."""
        lp = _make_loading_posterior()
        keep = np.array([0])
        lp.get_subset_dims(keep, in_place=True)
        assert lp.mean.shape == (Y_DIM, 1)
        assert lp.cov.shape == (Y_DIM, 1, 1)
        assert lp.moment.shape == (Y_DIM, 1, 1)

    def test_get_subset_dims_copy(self):
        """Copy mode returns a subset without modifying the original."""
        lp = _make_loading_posterior()
        original_shape = lp.mean.shape
        new_lp = lp.get_subset_dims(np.array([1]), in_place=False)
        # Original unchanged
        assert lp.mean.shape == original_shape
        # New has subset
        assert new_lp.mean.shape == (Y_DIM, 1)

    def test_copy_independence(self):
        """Copied posterior is independent of the original."""
        lp = _make_loading_posterior()
        lp2 = lp.copy()
        lp2.mean[:] = 999.0
        assert not np.any(lp.mean == 999.0)

    def test_clear(self):
        """Clear resets all fields to None."""
        lp = _make_loading_posterior()
        lp.clear()
        assert lp.mean is None
        assert lp.cov is None
        assert lp.moment is None

    def test_repr_shows_shapes(self):
        """Repr includes array shape information."""
        lp = _make_loading_posterior()
        r = repr(lp)
        assert "mean.shape=" in r
        assert "cov.shape=" in r


# =========================================================================== #
# ARDPosterior
# =========================================================================== #


class TestARDPosterior:
    """Tests for ARDPosterior dataclass."""

    def test_init_defaults(self):
        """All fields default to None."""
        ard = ARDPosterior()
        assert ard.a is None
        assert ard.b is None
        assert ard.mean is None

    def test_compute_mean_in_place(self):
        """In-place mean computation stores result on the object."""
        ard = _make_ard_posterior()
        expected = ard.a[:, np.newaxis] / ard.b  # (n_groups, x_dim)
        result = ard.compute_mean(in_place=True)
        np.testing.assert_allclose(result, expected)
        assert result is ard.mean

    def test_compute_mean_copy(self):
        """Copy mode returns a new array without modifying the object."""
        ard = _make_ard_posterior()
        result = ard.compute_mean(in_place=False)
        expected = ard.a[:, np.newaxis] / ard.b
        np.testing.assert_allclose(result, expected)
        assert result is not ard.mean

    def test_get_subset_dims_in_place(self):
        """In-place subsetting reduces x_dim on b and mean."""
        ard = _make_ard_posterior()
        ard.get_subset_dims(np.array([0]), in_place=True)
        # a is unchanged (shared across dims)
        assert ard.a.shape == (N_GROUPS,)
        assert ard.b.shape == (N_GROUPS, 1)
        assert ard.mean.shape == (N_GROUPS, 1)

    def test_get_subset_dims_copy(self):
        """Copy mode returns a subset without modifying the original."""
        ard = _make_ard_posterior()
        new_ard = ard.get_subset_dims(np.array([1]), in_place=False)
        assert ard.b.shape == (N_GROUPS, X_DIM)  # original unchanged
        assert new_ard.b.shape == (N_GROUPS, 1)

    def test_type_validation(self):
        """Non-ndarray inputs are rejected."""
        with pytest.raises(TypeError, match=r"a must be a numpy\.ndarray"):
            ARDPosterior(a=[1.0, 2.0])


# =========================================================================== #
# ObsMeanPosterior
# =========================================================================== #


class TestObsMeanPosterior:
    """Tests for ObsMeanPosterior dataclass."""

    def test_init_defaults(self):
        """All fields default to None."""
        omp = ObsMeanPosterior()
        assert omp.mean is None
        assert omp.cov is None

    def test_get_groups(self):
        """Splitting along y_dim produces correctly shaped per-group arrays."""
        omp = _make_obs_mean_posterior()
        means, covs = omp.get_groups(Y_DIMS)
        assert len(means) == N_GROUPS
        assert means[0].shape == (3,)
        assert means[1].shape == (2,)
        assert covs[0].shape == (3,)

    def test_get_groups_dim_mismatch(self):
        """Dimension mismatch raises ValueError."""
        omp = _make_obs_mean_posterior()
        with pytest.raises(ValueError, match="sum of y_dims"):
            omp.get_groups(np.array([10]))


# =========================================================================== #
# ObsPrecPosterior
# =========================================================================== #


class TestObsPrecPosterior:
    """Tests for ObsPrecPosterior dataclass."""

    def test_init_defaults(self):
        """All fields default to None."""
        opp = ObsPrecPosterior()
        assert opp.a is None
        assert opp.b is None
        assert opp.mean is None

    def test_compute_mean_in_place(self):
        """In-place mean computation stores result on the object."""
        opp = _make_obs_prec_posterior()
        result = opp.compute_mean(in_place=True)
        expected = opp.a / opp.b
        np.testing.assert_allclose(result, expected)
        assert result is opp.mean

    def test_compute_mean_copy(self):
        """Copy mode returns a new array without modifying the object."""
        opp = _make_obs_prec_posterior()
        result = opp.compute_mean(in_place=False)
        expected = opp.a / opp.b
        np.testing.assert_allclose(result, expected)
        assert result is not opp.mean

    def test_get_groups(self):
        """Splitting along y_dim produces correctly shaped per-group arrays."""
        opp = _make_obs_prec_posterior()
        means, bs = opp.get_groups(Y_DIMS)
        assert len(means) == N_GROUPS
        assert means[0].shape == (3,)
        assert bs[1].shape == (2,)

    def test_type_validation(self):
        """Non-float input for scalar parameter is rejected."""
        with pytest.raises(TypeError, match="a must be a float"):
            ObsPrecPosterior(a=1)  # int, not float


# =========================================================================== #
# ObsParamsPosterior
# =========================================================================== #


class TestObsParamsPosterior:
    """Tests for ObsParamsPosterior composite class."""

    def test_init_defaults(self):
        """Uninitialized posterior has empty sub-posteriors."""
        opp = ObsParamsPosterior()
        assert opp.x_dim is None
        assert opp.y_dims is None
        assert isinstance(opp.C, LoadingPosterior)
        assert opp.C.mean is None

    def test_is_initialized_true(self):
        """Initialized posterior reports True."""
        opp = _make_obs_params_posterior()
        assert opp.is_initialized()

    def test_is_initialized_false(self):
        """Default posterior reports False."""
        opp = ObsParamsPosterior()
        assert not opp.is_initialized()

    def test_posterior_mean(self):
        """Posterior mean returns an ObsParamsRealization with matching values."""
        opp = _make_obs_params_posterior()
        pm = opp.posterior_mean
        assert isinstance(pm, ObsParamsRealization)
        np.testing.assert_array_equal(pm.C, opp.C.mean)
        np.testing.assert_array_equal(pm.d, opp.d.mean)
        np.testing.assert_array_equal(pm.phi, opp.phi.mean)
        assert pm.x_dim == X_DIM

    def test_sample_shapes(self):
        """Sampled realization has correct array shapes."""
        opp = _make_obs_params_posterior()
        rng = np.random.default_rng(42)
        sample = opp.sample(rng)
        assert isinstance(sample, ObsParamsRealization)
        assert sample.C.shape == (Y_DIM, X_DIM)
        assert sample.d.shape == (Y_DIM,)
        assert sample.phi.shape == (Y_DIM,)
        assert sample.alpha.shape == (N_GROUPS, X_DIM)
        assert sample.x_dim == X_DIM

    def test_sample_reproducibility(self):
        """Same seed produces identical samples."""
        opp = _make_obs_params_posterior()
        s1 = opp.sample(np.random.default_rng(0))
        s2 = opp.sample(np.random.default_rng(0))
        np.testing.assert_array_equal(s1.C, s2.C)
        np.testing.assert_array_equal(s1.phi, s2.phi)

    def test_get_subset_dims_in_place(self):
        """In-place subsetting reduces x_dim across all sub-posteriors."""
        opp = _make_obs_params_posterior()
        opp.get_subset_dims(np.array([0]), in_place=True)
        assert opp.x_dim == 1
        assert opp.C.mean.shape == (Y_DIM, 1)
        assert opp.alpha.b.shape == (N_GROUPS, 1)

    def test_get_subset_dims_copy(self):
        """Copy mode returns a subset without modifying the original."""
        opp = _make_obs_params_posterior()
        new_opp = opp.get_subset_dims(np.array([1]), in_place=False)
        assert opp.x_dim == X_DIM  # original unchanged
        assert new_opp.x_dim == 1
        assert new_opp.C.mean.shape == (Y_DIM, 1)

    def test_copy_independence(self):
        """Copied posterior is independent of the original."""
        opp = _make_obs_params_posterior()
        opp2 = opp.copy()
        opp2.C.mean[:] = 999.0
        assert not np.any(opp.C.mean == 999.0)

    def test_compute_snr(self):
        """SNR is positive with correct shape."""
        opp = _make_obs_params_posterior()
        snr = opp.compute_snr()
        assert snr.shape == (N_GROUPS,)
        assert np.all(snr > 0)

    def test_get_dim_types(self):
        """Binary encoding: 2 groups -> 4 types (00, 01, 10, 11)."""
        dim_types = ObsParamsPosterior.get_dim_types(N_GROUPS)
        assert dim_types.shape == (N_GROUPS, 2**N_GROUPS)
        # First column is all-zero (null type), last is all-one (shared)
        assert np.all(dim_types[:, 0] == 0)
        assert np.all(dim_types[:, -1] == 1)

    def test_compute_dimensionalities_shapes(self):
        """Dimensionality outputs have correct shapes."""
        opp = _make_obs_params_posterior()
        num_dim, sig_dims, var_exp, dim_types = opp.compute_dimensionalities()
        n_dim_types = 2**N_GROUPS
        assert num_dim.shape == (n_dim_types,)
        assert sig_dims.shape == (N_GROUPS, X_DIM)
        assert var_exp.shape == (N_GROUPS, n_dim_types)
        assert dim_types.shape == (N_GROUPS, n_dim_types)

    def test_compute_dims_pairs(self):
        """Pairwise dimensionality computation for 2 groups yields 1 pair."""
        opp = _make_obs_params_posterior()
        num_dim, _, var_exp, dim_types = opp.compute_dimensionalities()
        pair_dims, pair_var_exp, pairs = ObsParamsPosterior.compute_dims_pairs(
            num_dim, dim_types, var_exp
        )
        # 2 groups -> 1 pair
        assert pairs.shape == (1, 2)
        assert pair_dims.shape == (1, 3)
        assert pair_var_exp.shape == (1, 2)

    def test_repr(self):
        """Repr includes class name and x_dim."""
        opp = _make_obs_params_posterior()
        r = repr(opp)
        assert "ObsParamsPosterior" in r
        assert "x_dim=" in r

    def test_type_validation(self):
        """Non-integer x_dim is rejected."""
        with pytest.raises(TypeError, match="x_dim must be an integer"):
            ObsParamsPosterior(x_dim=2.0)
