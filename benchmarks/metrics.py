"""Benchmark metrics for parameter recovery evaluation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def latent_permutation(X_true: np.ndarray, X_est: np.ndarray) -> np.ndarray:
    """Find permutation aligning estimated latents to true latents.

    Computes absolute Pearson correlations between all pairs of true and
    estimated latent dimensions, then solves the assignment problem to
    maximize total absolute correlation. Handles sign ambiguity (a latent
    factor recovered with flipped sign still matches correctly).

    Parameters
    ----------
    X_true : ndarray of shape (x_dim, n_samples)
        Ground truth latent factors.
    X_est : ndarray of shape (x_dim, n_samples)
        Estimated latent factors (same x_dim as X_true).

    Returns
    -------
    ndarray of shape (x_dim,)
        Permutation indices: estimated dimension ``perm[k]`` best matches
        true dimension ``k``.
    """
    x_dim = X_true.shape[0]
    # Cross-correlation block: upper-right of full corrcoef matrix.
    # Zero-variance dimensions produce 0/0 = NaN in the normalization;
    # suppress the warning and replace with 0 (genuinely uncorrelated).
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X_true, X_est)[:x_dim, x_dim:]  # (x_dim, x_dim)
    np.nan_to_num(corr, copy=False, nan=0.0)
    # Minimize negative absolute correlation = maximize absolute correlation
    _, col_ind = linear_sum_assignment(-np.abs(corr))
    return col_ind


def subspace_error(C_true: np.ndarray, C_est: np.ndarray) -> float:
    r"""Compute normalized subspace error between loading matrices.

    Measures how well the estimated loading matrix captures the column space
    of the ground truth. Invariant to column ordering and handles different
    column counts.

    Parameters
    ----------
    C_true : ndarray of shape (y_dim, x_dim_true)
        Ground truth loading matrix.
    C_est : ndarray of shape (y_dim, x_dim_est)
        Estimated loading matrix.

    Returns
    -------
    float
        Subspace error in [0, 1]. 0 means the estimate captures the full
        column space of ground truth; 1 means orthogonal subspaces.

    Notes
    -----
    The metric is computed as:

    .. math::

        e_{\\text{sub}} = \\frac{\\|(I - \\hat{C}\\hat{C}^+)C\\|_F}{\\|C\\|_F}

    where :math:`\\hat{C}^+` is the Moore-Penrose pseudoinverse.
    """
    # Project C_true onto the null space of C_est
    C_est_pinv = np.linalg.pinv(C_est)  # (x_dim_est, y_dim)
    P_null = np.eye(C_true.shape[0]) - C_est @ C_est_pinv  # (y_dim, y_dim)
    residual = P_null @ C_true  # (y_dim, x_dim_true)

    return float(np.linalg.norm(residual, "fro") / np.linalg.norm(C_true, "fro"))


def relative_l2_error(v_true: np.ndarray, v_est: np.ndarray) -> float:
    """Compute relative L2 error between vectors.

    Parameters
    ----------
    v_true : ndarray
        Ground truth vector (any shape, will be flattened).
    v_est : ndarray
        Estimated vector (same shape as v_true).

    Returns
    -------
    float
        Relative L2 error: ||v_true - v_est||_2 / ||v_true||_2.
    """
    v_true_flat = v_true.ravel()
    v_est_flat = v_est.ravel()
    return float(np.linalg.norm(v_true_flat - v_est_flat) / np.linalg.norm(v_true_flat))


def denoised_r2(
    C_true: np.ndarray,
    X_true: np.ndarray,
    d_true: np.ndarray,
    C_est: np.ndarray,
    X_est: np.ndarray,
    d_est: np.ndarray,
) -> float:
    r"""Compute R-squared between true and estimated noise-free signals.

    Compares reconstructed observation-space signals without noise,
    measuring how well the estimated latents and loadings recover
    the true underlying signal.

    Parameters
    ----------
    C_true : ndarray of shape (y_dim, x_dim_true)
        Ground truth loading matrix.
    X_true : ndarray of shape (x_dim_true, n_samples)
        Ground truth latent factors.
    d_true : ndarray of shape (y_dim,)
        Ground truth observation means.
    C_est : ndarray of shape (y_dim, x_dim_est)
        Estimated loading matrix.
    X_est : ndarray of shape (x_dim_est, n_samples)
        Estimated latent factors.
    d_est : ndarray of shape (y_dim,)
        Estimated observation means.

    Returns
    -------
    float
        R-squared value. 1.0 means perfect recovery of noise-free signal.

    Notes
    -----
    The metric is computed as:

    .. math::

        R^2 = 1 - \\frac{\\|CX + d - (\\hat{C}\\hat{X} + \\hat{d})\\|_F^2}
                       {\\|CX + d - \\bar{Y}\\|_F^2}

    where the denominator uses the mean of the true noise-free signal.
    """
    # Noise-free signals: (y_dim, n_samples)
    signal_true = C_true @ X_true + d_true[:, np.newaxis]
    signal_est = C_est @ X_est + d_est[:, np.newaxis]

    # Residual sum of squares
    ss_res = np.sum((signal_true - signal_est) ** 2)

    # Total sum of squares (variance of true signal)
    signal_mean = signal_true.mean(axis=1, keepdims=True)  # (y_dim, 1)
    ss_tot = np.sum((signal_true - signal_mean) ** 2)

    return float(1.0 - ss_res / ss_tot)
