"""Posterior distributions for observation model parameters."""

from __future__ import annotations

import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from latents._core.base import ArrayContainer
from latents._core.numerics import stability_floor
from latents.observation.realizations import ObsParamsRealization


class LoadingPosterior(ArrayContainer):
    """Posterior distribution over loading matrices.

    Parameters
    ----------
    mean
        Posterior mean, shape (y_dim, x_dim).
    cov
        Posterior covariances, shape (y_dim, x_dim, x_dim).
    moment
        Posterior second moments, shape (y_dim, x_dim, x_dim).
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        moment: np.ndarray | None = None,
    ):
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

        if cov is not None and not isinstance(cov, np.ndarray):
            msg = "cov must be a numpy.ndarray."
            raise TypeError(msg)
        self.cov = cov

        if moment is not None and not isinstance(moment, np.ndarray):
            msg = "moment must be a numpy.ndarray."
            raise TypeError(msg)
        self.moment = moment

    def get_groups(
        self,
        y_dims: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Get list of views into posterior parameters, one per group.

        Parameters
        ----------
        y_dims
            Dimensionalities of each observed group, shape (n_groups,).

        Returns
        -------
        group_means
            List of views into mean, one per group. None if mean is None.
        group_covs
            List of views into cov, one per group. None if cov is None.
        group_moments
            List of views into moment, one per group. None if moment is None.
        """
        group_means = None
        if self.mean is not None:
            if np.sum(y_dims) != self.mean.shape[0]:
                msg = "The sum of y_dims must equal the number of rows in mean."
                raise ValueError(msg)
            group_means = np.split(self.mean, np.cumsum(y_dims)[:-1], axis=0)

        group_covs = None
        if self.cov is not None:
            if np.sum(y_dims) != self.cov.shape[0]:
                msg = "The sum of y_dims must equal the size of the first axis of cov."
                raise ValueError(msg)
            group_covs = np.split(self.cov, np.cumsum(y_dims)[:-1], axis=0)

        group_moments = None
        if self.moment is not None:
            if np.sum(y_dims) != self.moment.shape[0]:
                msg = (
                    "The sum of y_dims must equal the size of the first axis of moment."
                )
                raise ValueError(msg)
            group_moments = np.split(self.moment, np.cumsum(y_dims)[:-1], axis=0)

        return group_means, group_covs, group_moments

    def compute_moment(self, in_place: bool = True) -> np.ndarray:
        """Compute posterior second moments E[C_i C_i^T] for each row i.

        Parameters
        ----------
        in_place
            If True, store in self.moment and return reference.
            If False, return new array.

        Returns
        -------
        ndarray
            Second moments, shape (y_dim, x_dim, x_dim).
        """
        if in_place:
            if self.moment is None:
                self.moment = np.zeros_like(self.cov)
            self.moment[:] = np.einsum("ij,ik->ijk", self.mean, self.mean) + self.cov
            return self.moment

        return np.einsum("ij,ik->ijk", self.mean, self.mean) + self.cov

    def compute_squared_norms(self, y_dims: np.ndarray) -> np.ndarray:
        """Compute expected squared norm of each column per group.

        Parameters
        ----------
        y_dims
            Dimensionalities of each observed group, shape (n_groups,).

        Returns
        -------
        ndarray
            Squared norms, shape (n_groups, x_dim).
        """
        n_groups = len(y_dims)
        x_dim = self.moment.shape[1]

        _, _, moments = self.get_groups(y_dims)
        squared_norms = np.zeros((n_groups, x_dim))
        for group_idx in range(n_groups):
            squared_norms[group_idx, :] = np.sum(
                moments[group_idx].diagonal(offset=0, axis1=1, axis2=2), axis=0
            )

        return squared_norms

    def get_subset_dims(
        self,
        x_indices: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """Keep only specified latent dimensions.

        Parameters
        ----------
        x_indices
            Indices of latent dimensions to keep.
        in_place
            If True, modify self. If False, return new instance.

        Returns
        -------
        Self
            Modified or new instance.
        """
        new_mean = self.mean[:, x_indices] if self.mean is not None else None
        new_cov = (
            self.cov[np.ix_(np.arange(self.cov.shape[0]), x_indices, x_indices)]
            if self.cov is not None
            else None
        )
        new_moment = (
            self.moment[np.ix_(np.arange(self.moment.shape[0]), x_indices, x_indices)]
            if self.moment is not None
            else None
        )

        if in_place:
            self.mean = new_mean
            self.cov = new_cov
            self.moment = new_moment
            return self

        return self.__class__(mean=new_mean, cov=new_cov, moment=new_moment)


class ARDPosterior(ArrayContainer):
    """Posterior distribution over ARD parameters.

    Parameters
    ----------
    a
        Shape parameters, shape (n_groups,).
    b
        Rate parameters, shape (n_groups, x_dim).
    mean
        Posterior mean a/b, shape (n_groups, x_dim).
    """

    def __init__(
        self,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
        mean: np.ndarray | None = None,
    ):
        if a is not None and not isinstance(a, np.ndarray):
            msg = "a must be a numpy.ndarray."
            raise TypeError(msg)
        self.a = a

        if b is not None and not isinstance(b, np.ndarray):
            msg = "b must be a numpy.ndarray."
            raise TypeError(msg)
        self.b = b

        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

    def compute_mean(self, in_place: bool = True) -> np.ndarray:
        """Compute posterior mean a/b.

        Parameters
        ----------
        in_place
            If True, store in self.mean and return reference.
            If False, return new array.

        Returns
        -------
        ndarray
            Posterior mean, shape (n_groups, x_dim).
        """
        floor = stability_floor(self.b.dtype)
        if in_place:
            if self.mean is None:
                self.mean = np.zeros_like(self.b)
            self.mean[:] = self.a[:, np.newaxis] / np.maximum(self.b, floor)
            return self.mean

        return self.a[:, np.newaxis] / np.maximum(self.b, floor)

    def get_subset_dims(
        self,
        x_indices: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """Keep only specified latent dimensions.

        Parameters
        ----------
        x_indices
            Indices of latent dimensions to keep.
        in_place
            If True, modify self. If False, return new instance.

        Returns
        -------
        Self
            Modified or new instance.
        """
        new_b = self.b[:, x_indices] if self.b is not None else None
        new_mean = self.mean[:, x_indices] if self.mean is not None else None

        if in_place:
            self.b = new_b
            self.mean = new_mean
            return self

        new_a = self.a.copy() if self.a is not None else None
        return self.__class__(a=new_a, b=new_b, mean=new_mean)


class ObsMeanPosterior(ArrayContainer):
    """Posterior distribution over observation mean parameters.

    Parameters
    ----------
    mean
        Posterior mean, shape (y_dim,).
    cov
        Posterior variance (diagonal), shape (y_dim,).
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        cov: np.ndarray | None = None,
    ):
        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

        if cov is not None and not isinstance(cov, np.ndarray):
            msg = "cov must be a numpy.ndarray."
            raise TypeError(msg)
        self.cov = cov

    def get_groups(
        self,
        y_dims: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get list of views into posterior parameters, one per group.

        Parameters
        ----------
        y_dims
            Dimensionalities of each observed group, shape (n_groups,).

        Returns
        -------
        group_means
            List of views into mean, one per group. None if mean is None.
        group_covs
            List of views into cov, one per group. None if cov is None.
        """
        group_means = None
        if self.mean is not None:
            if np.sum(y_dims) != len(self.mean):
                msg = "The sum of y_dims must equal the length of mean."
                raise ValueError(msg)
            group_means = np.split(self.mean, np.cumsum(y_dims)[:-1], axis=0)

        group_covs = None
        if self.cov is not None:
            if np.sum(y_dims) != len(self.cov):
                msg = "The sum of y_dims must equal the length of cov."
                raise ValueError(msg)
            group_covs = np.split(self.cov, np.cumsum(y_dims)[:-1], axis=0)

        return group_means, group_covs


class ObsPrecPosterior(ArrayContainer):
    """Posterior distribution over observation precision parameters.

    Parameters
    ----------
    a
        Shape parameter (scalar, shared across dimensions).
    b
        Rate parameters, shape (y_dim,).
    mean
        Posterior mean a/b, shape (y_dim,).
    """

    def __init__(
        self,
        a: float | None = None,
        b: np.ndarray | None = None,
        mean: np.ndarray | None = None,
    ):
        if a is not None and not isinstance(a, float):
            msg = "a must be a float."
            raise TypeError(msg)
        self.a = a

        if b is not None and not isinstance(b, np.ndarray):
            msg = "b must be a numpy.ndarray."
            raise TypeError(msg)
        self.b = b

        if mean is not None and not isinstance(mean, np.ndarray):
            msg = "mean must be a numpy.ndarray."
            raise TypeError(msg)
        self.mean = mean

    def get_groups(
        self,
        y_dims: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get list of views into posterior parameters, one per group.

        Parameters
        ----------
        y_dims
            Dimensionalities of each observed group, shape (n_groups,).

        Returns
        -------
        group_means
            List of views into mean, one per group. None if mean is None.
        group_bs
            List of views into b, one per group. None if b is None.
        """
        group_means = None
        if self.mean is not None:
            if np.sum(y_dims) != len(self.mean):
                msg = "The sum of y_dims must equal the length of mean."
                raise ValueError(msg)
            group_means = np.split(self.mean, np.cumsum(y_dims)[:-1], axis=0)

        group_bs = None
        if self.b is not None:
            if np.sum(y_dims) != len(self.b):
                msg = "The sum of y_dims must equal the length of b."
                raise ValueError(msg)
            group_bs = np.split(self.b, np.cumsum(y_dims)[:-1], axis=0)

        return group_means, group_bs

    def compute_mean(self, in_place: bool = True) -> np.ndarray:
        """Compute posterior mean a/b.

        Parameters
        ----------
        in_place
            If True, store in self.mean and return reference.
            If False, return new array.

        Returns
        -------
        ndarray
            Posterior mean, shape (y_dim,).
        """
        floor = stability_floor(self.b.dtype)
        if in_place:
            if self.mean is None:
                self.mean = np.zeros_like(self.b)
            self.mean[:] = self.a / np.maximum(self.b, floor)
            return self.mean

        return self.a / np.maximum(self.b, floor)


class ObsParamsPosterior:
    """Posterior distributions over all observation model parameters.

    Bundle of posteriors q(C), q(alpha), q(d), q(phi) with methods for
    sampling, computing point estimates, and analysis.

    Parameters
    ----------
    x_dim
        Number of latent dimensions.
    y_dims
        Dimensionalities of each observed group, shape (n_groups,).
    C
        Posterior over loading matrices.
    alpha
        Posterior over ARD parameters.
    d
        Posterior over observation means.
    phi
        Posterior over observation precisions.
    """

    def __init__(
        self,
        x_dim: int | None = None,
        y_dims: np.ndarray | None = None,
        C: LoadingPosterior | None = None,
        alpha: ARDPosterior | None = None,
        d: ObsMeanPosterior | None = None,
        phi: ObsPrecPosterior | None = None,
    ):
        if x_dim is not None and not isinstance(x_dim, int):
            msg = "x_dim must be an integer."
            raise TypeError(msg)
        self.x_dim = x_dim

        if y_dims is not None and not isinstance(y_dims, np.ndarray):
            msg = "y_dims must be a numpy.ndarray of integers."
            raise TypeError(msg)
        self.y_dims = y_dims

        if C is None:
            self.C = LoadingPosterior()
        elif not isinstance(C, LoadingPosterior):
            msg = "C must be a LoadingPosterior object."
            raise TypeError(msg)
        else:
            self.C = C

        if alpha is None:
            self.alpha = ARDPosterior()
        elif not isinstance(alpha, ARDPosterior):
            msg = "alpha must be an ARDPosterior object."
            raise TypeError(msg)
        else:
            self.alpha = alpha

        if d is None:
            self.d = ObsMeanPosterior()
        elif not isinstance(d, ObsMeanPosterior):
            msg = "d must be an ObsMeanPosterior object."
            raise TypeError(msg)
        else:
            self.d = d

        if phi is None:
            self.phi = ObsPrecPosterior()
        elif not isinstance(phi, ObsPrecPosterior):
            msg = "phi must be an ObsPrecPosterior object."
            raise TypeError(msg)
        else:
            self.phi = phi

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(x_dim={self.x_dim}, "
            f"y_dims={self.y_dims}, "
            f"C={self.C}, "
            f"alpha={self.alpha}, "
            f"d={self.d}, "
            f"phi={self.phi})"
        )

    @property
    def posterior_mean(self) -> ObsParamsRealization:
        """Return posterior means as a realization.

        Returns
        -------
        ObsParamsRealization
            Realization with posterior mean values.
        """
        return ObsParamsRealization(
            C=self.C.mean.copy(),
            d=self.d.mean.copy(),
            phi=self.phi.mean.copy(),
            alpha=self.alpha.mean.copy(),
            y_dims=self.y_dims.copy(),
            x_dim=self.x_dim,
        )

    def sample(self, rng: np.random.Generator) -> ObsParamsRealization:
        """Draw a sample from the posterior distributions.

        Samples from q(C), q(alpha), q(d), q(phi) independently.

        Parameters
        ----------
        rng
            Random number generator.

        Returns
        -------
        ObsParamsRealization
            Sampled parameter values.
        """
        y_dim = int(self.y_dims.sum())
        n_groups = len(self.y_dims)

        # Sample C: each row independently from N(mean_i, cov_i)
        C_sample = np.zeros_like(self.C.mean)
        for i in range(y_dim):
            C_sample[i, :] = rng.multivariate_normal(self.C.mean[i, :], self.C.cov[i])

        # Sample alpha: Gamma(a, b) for each element
        # a has shape (n_groups,), b has shape (n_groups, x_dim)
        alpha_sample = np.zeros_like(self.alpha.mean)
        for g in range(n_groups):
            alpha_sample[g, :] = rng.gamma(
                shape=self.alpha.a[g], scale=1 / self.alpha.b[g, :]
            )

        # Sample d: N(mean, cov) element-wise (diagonal covariance)
        d_sample = rng.normal(self.d.mean, np.sqrt(self.d.cov))

        # Sample phi: Gamma(a, b) for each element
        phi_sample = rng.gamma(shape=self.phi.a, scale=1 / self.phi.b)

        return ObsParamsRealization(
            C=C_sample,
            d=d_sample,
            phi=phi_sample,
            alpha=alpha_sample,
            y_dims=self.y_dims.copy(),
            x_dim=self.x_dim,
        )

    def is_initialized(self) -> bool:
        """Check if all posterior parameters have been initialized."""
        return (
            self.x_dim is not None
            and self.y_dims is not None
            and self.C.mean is not None
            and self.C.moment is not None
            and self.alpha.a is not None
            and self.alpha.b is not None
            and self.alpha.mean is not None
            and self.d.mean is not None
            and self.d.cov is not None
            and self.phi.a is not None
            and self.phi.b is not None
            and self.phi.mean is not None
        )

    def get_subset_dims(
        self,
        x_indices: np.ndarray,
        in_place: bool = True,
    ) -> Self:
        """Keep only specified latent dimensions.

        Parameters
        ----------
        x_indices
            Indices of latent dimensions to keep.
        in_place
            If True, modify self. If False, return new instance.

        Returns
        -------
        Self
            Modified or new instance.
        """
        if in_place:
            self.x_dim = len(x_indices)
            self.C.get_subset_dims(x_indices, in_place=True)
            self.alpha.get_subset_dims(x_indices, in_place=True)
            return self

        return self.__class__(
            x_dim=len(x_indices),
            y_dims=self.y_dims.copy(),
            C=self.C.get_subset_dims(x_indices, in_place=False),
            alpha=self.alpha.get_subset_dims(x_indices, in_place=False),
            d=self.d.copy(),
            phi=self.phi.copy(),
        )

    def copy(self) -> Self:
        """Return a deep copy."""
        return self.__class__(
            x_dim=self.x_dim,
            y_dims=self.y_dims.copy(),
            C=self.C.copy(),
            alpha=self.alpha.copy(),
            d=self.d.copy(),
            phi=self.phi.copy(),
        )

    def compute_snr(self, y_dims: np.ndarray | None = None) -> np.ndarray:
        """Compute signal-to-noise ratio for each group.

        Parameters
        ----------
        y_dims
            Dimensionalities of observed groups. Defaults to self.y_dims.

        Returns
        -------
        ndarray
            SNR for each group, shape (n_groups,).
        """
        if y_dims is None:
            y_dims = self.y_dims

        _, _, C_moments = self.C.get_groups(y_dims)
        phi_means, _ = self.phi.get_groups(y_dims)

        return np.array(
            [
                np.trace(np.sum(C_moments[group_idx], axis=0))
                / np.sum(1 / phi_means[group_idx])
                for group_idx in range(len(y_dims))
            ]
        )

    @staticmethod
    def get_dim_types(n_groups: int) -> np.ndarray:
        """Generate all dimension types for n_groups.

        Parameters
        ----------
        n_groups
            Number of observed groups.

        Returns
        -------
        ndarray
            Boolean array, shape (n_groups, 2^n_groups). Column j indicates
            which groups are involved in dimension type j.
        """
        n_dim_types = 2**n_groups
        dim_types = np.empty((n_groups, n_dim_types))

        for dim_idx in range(n_dim_types):
            dim_str = format(dim_idx, f"0{n_groups}b")
            dim_types[:, dim_idx] = np.array([int(b) for b in dim_str], dtype=bool)

        return dim_types

    def compute_dimensionalities(
        self,
        cutoff_shared_var: float = 0.02,
        cutoff_snr: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute dimensionalities and variance explained by dimension type.

        Parameters
        ----------
        cutoff_shared_var
            Minimum fraction of shared variance for significance.
        cutoff_snr
            Minimum SNR for any latents to be significant.

        Returns
        -------
        num_dim
            Number of each dimension type, shape (n_dim_types,).
        sig_dims
            Significant dimensions, shape (n_groups, x_dim).
        var_exp
            Variance explained by type, shape (n_groups, n_dim_types).
        dim_types
            Dimension type indicators, shape (n_groups, n_dim_types).
        """
        n_groups = len(self.y_dims)
        dim_types = self.get_dim_types(n_groups)
        n_dim_types = dim_types.shape[1]

        snr = self.compute_snr()

        alpha_inv = 1 / self.alpha.mean
        alpha_inv_rel = alpha_inv / np.sum(alpha_inv, axis=1, keepdims=True)

        sig_dims = (alpha_inv_rel > cutoff_shared_var) & (snr > cutoff_snr)[
            :, np.newaxis
        ]
        num_dim = np.zeros(n_dim_types)
        var_exp = np.zeros((n_groups, n_dim_types))
        for dim_idx in range(n_dim_types):
            dims = np.all(sig_dims == dim_types[:, dim_idx, np.newaxis], axis=0)
            num_dim[dim_idx] = np.sum(dims)
            var_exp[:, dim_idx] = np.sum(alpha_inv_rel[:, dims], axis=1)

        return num_dim, sig_dims, var_exp, dim_types

    @staticmethod
    def compute_dims_pairs(
        num_dim: np.ndarray,
        dim_types: np.ndarray,
        var_exp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze shared dimensionalities between pairs of groups.

        Parameters
        ----------
        num_dim
            Number of each dimension type, shape (n_dim_types,).
        dim_types
            Dimension type indicators, shape (n_groups, n_dim_types).
        var_exp
            Variance explained by type, shape (n_groups, n_dim_types).

        Returns
        -------
        pair_dims
            Dimensionalities per pair, shape (n_pairs, 3).
        pair_var_exp
            Variance explained per pair, shape (n_pairs, 2).
        pairs
            Pair indices, shape (n_pairs, 2).
        """
        n_groups = dim_types.shape[0]
        group_idxs = [np.nonzero(dim_types[g, :])[0] for g in range(n_groups)]
        pairs = list(combinations(range(n_groups), 2))
        num_pairs = len(pairs)

        pair_dims = np.zeros((num_pairs, 3), dtype=int)
        pair_var_exp = np.zeros((num_pairs, 2))
        for pair_idx, pair in enumerate(pairs):
            pair_dims[pair_idx, 0] = num_dim[group_idxs[pair[0]]].sum()
            pair_dims[pair_idx, 2] = num_dim[group_idxs[pair[1]]].sum()

            shared_idxs = np.intersect1d(group_idxs[pair[0]], group_idxs[pair[1]])
            pair_dims[pair_idx, 1] = num_dim[shared_idxs].sum()

            pair_var_exp[pair_idx, 0] = var_exp[pair[0], shared_idxs].sum()
            pair_var_exp[pair_idx, 1] = var_exp[pair[1], shared_idxs].sum()

        return pair_dims, pair_var_exp, np.array(pairs)

    # --- Plotting methods (to be extracted to plotting/ in Phase 5) ---

    @staticmethod
    def plot_dimensionalities(
        num_dim: np.ndarray,
        dim_types: np.ndarray,
        sem_dim: np.ndarray | None = None,
        group_names: list[str] | None = None,
        plot_zero_dim: bool = False,
        ax: Axes | None = None,
    ) -> None:
        """Plot number of each dimension type."""
        ax = ax if ax is not None else plt.gca()

        n_groups, n_dim_types = dim_types.shape
        dim_cardinality = dim_types.sum(axis=0)

        if group_names is None:
            group_names = [f"{i + 1}" for i in range(n_groups)]
        xticklbls = ["" for i in range(n_dim_types)]
        for dim_idx in range(n_dim_types):
            if dim_cardinality[dim_idx] == 0:
                xticklbls[dim_idx] = "n.s."
            else:
                involved_groups = np.where(dim_types[:, dim_idx])[0]
                xticklbls[dim_idx] = "-".join([group_names[i] for i in involved_groups])

        sort_idxs = np.argsort(dim_cardinality)
        if not plot_zero_dim:
            sort_idxs = sort_idxs[dim_cardinality[sort_idxs] > 0]
            n_dim_types = len(sort_idxs)

        if sem_dim is None:
            ax.bar(np.arange(1, n_dim_types + 1), num_dim[sort_idxs])
        else:
            ax.bar(
                np.arange(1, n_dim_types + 1),
                num_dim[sort_idxs],
                yerr=sem_dim[sort_idxs],
            )
        ax.set_xlabel("Dimension type")
        ax.set_ylabel("Dimensionality")
        ax.set_xticks(np.arange(1, n_dim_types + 1))
        ax.set_xticklabels([xticklbls[i] for i in sort_idxs])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @staticmethod
    def plot_var_exp(
        var_exp: np.ndarray,
        dim_types: np.ndarray,
        sem_var_exp: np.ndarray | None = None,
        group_names: list[str] | None = None,
        plot_zero_dim: bool = False,
        fig: Figure | None = None,
    ) -> None:
        """Plot shared variance explained by each dimension type."""
        fig = fig if fig is not None else plt.gcf()

        n_groups, n_dim_types = dim_types.shape
        dim_cardinality = dim_types.sum(axis=0)

        if group_names is None:
            group_names = [f"{i + 1}" for i in range(n_groups)]
        xticklbls = ["" for i in range(n_dim_types)]
        for dim_idx in range(n_dim_types):
            if dim_cardinality[dim_idx] == 0:
                xticklbls[dim_idx] = "n.s."
            else:
                involved_groups = np.where(dim_types[:, dim_idx])[0]
                xticklbls[dim_idx] = "-".join([group_names[i] for i in involved_groups])

        sort_idxs = np.argsort(dim_cardinality)
        if not plot_zero_dim:
            sort_idxs = sort_idxs[dim_cardinality[sort_idxs] > 0]
            n_dim_types = len(sort_idxs)

        for group_idx in range(n_groups):
            plt.subplot(n_groups, 1, group_idx + 1)
            if sem_var_exp is None:
                plt.bar(np.arange(1, n_dim_types + 1), var_exp[group_idx, sort_idxs])
            else:
                plt.bar(
                    np.arange(1, n_dim_types + 1),
                    var_exp[group_idx, sort_idxs],
                    yerr=sem_var_exp[group_idx, sort_idxs],
                )
            plt.ylim([0, 1])
            plt.xlabel("Dimension type")
            plt.ylabel("Frac. shared var. exp.")
            plt.xticks(np.arange(1, n_dim_types + 1), [xticklbls[i] for i in sort_idxs])
            plt.title(f"Group {group_names[group_idx]}")
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

        fig.tight_layout()

    @staticmethod
    def plot_dims_pairs(
        pair_dims: np.ndarray,
        pairs: np.ndarray,
        n_groups: int,
        sem_pair_dims: np.ndarray | None = None,
        group_names: list[str] | None = None,
        fig: Figure | None = None,
    ) -> None:
        """Visualize pairwise dimensionality analysis."""
        fig = fig if fig is not None else plt.gcf()
        num_pairs = pairs.shape[0]

        if group_names is None:
            group_names = [f"{i + 1}" for i in range(n_groups)]
        xticklbls = np.full((num_pairs, 3), "", dtype=object)
        for pair_idx in range(num_pairs):
            xticklbls[pair_idx, 0] = f"Total, {group_names[pairs[pair_idx, 0]]}"
            xticklbls[pair_idx, 1] = (
                f"{group_names[pairs[pair_idx, 0]]}-{group_names[pairs[pair_idx, 1]]}"
            )
            xticklbls[pair_idx, 2] = f"Total, {group_names[pairs[pair_idx, 1]]}"

        for pair_idx in range(num_pairs):
            plt.subplot(1, num_pairs, pair_idx + 1)
            if sem_pair_dims is None:
                plt.bar(np.arange(1, pair_dims.shape[1] + 1), pair_dims[pair_idx, :])
            else:
                plt.bar(
                    np.arange(1, pair_dims.shape[1] + 1),
                    pair_dims[pair_idx, :],
                    yerr=sem_pair_dims[pair_idx, :],
                )
            plt.xlabel("Dimension type")
            plt.ylabel("Dimensionality")
            plt.xticks(np.arange(1, pair_dims.shape[1] + 1), xticklbls[pair_idx, :])
            plt.title(xticklbls[pair_idx, 1])
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

        fig.tight_layout()

    @staticmethod
    def plot_var_exp_pairs(
        pair_var_exp: np.ndarray,
        pairs: np.ndarray,
        n_groups: int,
        sem_pair_var_exp: np.ndarray | None = None,
        group_names: list[str] | None = None,
        fig: Figure | None = None,
    ) -> None:
        """Visualize pairwise shared variance analysis."""
        fig = fig if fig is not None else plt.gcf()
        num_pairs = pairs.shape[0]

        if group_names is None:
            group_names = [f"{i + 1}" for i in range(n_groups)]

        pairlbls = np.array(
            [
                f"{group_names[pairs[i, 0]]}" + f"-{group_names[pairs[i, 1]]}"
                for i in range(num_pairs)
            ]
        )

        for pair_idx in range(num_pairs):
            plt.subplot(1, num_pairs, pair_idx + 1)
            if sem_pair_var_exp is None:
                plt.bar(
                    np.arange(1, pair_var_exp.shape[1] + 1), pair_var_exp[pair_idx, :]
                )
            else:
                plt.bar(
                    np.arange(1, pair_var_exp.shape[1] + 1),
                    pair_var_exp[pair_idx, :],
                    yerr=sem_pair_var_exp[pair_idx, :],
                )
            plt.ylim([0, 1])
            plt.xlabel("Group")
            plt.ylabel("Frac. shared var. exp.")
            plt.xticks(
                np.arange(1, pair_var_exp.shape[1] + 1),
                np.array(group_names)[pairs[pair_idx, :]],
            )
            plt.title(pairlbls[pair_idx])
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

        fig.tight_layout()
