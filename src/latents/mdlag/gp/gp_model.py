"""Gaussian Process model for mDLAG with multi-group support."""

from __future__ import annotations

import numpy as np

from .fit_config import GPFitConfig
from .kernels.multigroup_kernel import MultiGroupGPKernel
from .kernels.rbf.rbf_kernel import RBFKernel
from .multigroup_params import MultiGroupGPHyperParams, MultiGroupGPParams
from .optimize import run_gp_optimizer


class mDLAGGP:
    """A class that encapsulates the GP model components.

    Attributes
    ----------
    params : MultiGroupGPParams
        The parameters of the GP (gamma, delays, eps).
    kernel : MultiGroupGPKernel
        The kernel instance (e.g., RBFKernel) that defines the covariance.
    hyper_params : MultiGroupGPHyperParams
        Hyperparameters for parameter constraints (e.g., min_gamma, max_delay).
    """

    def __init__(
        self,
        gamma: np.ndarray,
        delays: np.ndarray,
        eps: np.ndarray,
        kernel: MultiGroupGPKernel | None = None,
        hyper_params: MultiGroupGPHyperParams | None = None,
    ):
        """Initialize the mDLAGGP model with direct parameters.

        Parameters
        ----------
        gamma : np.ndarray
            Length scale parameters, shape (x_dim,).
        delays : np.ndarray
            Delay parameters, shape (num_groups, x_dim).
        eps : np.ndarray
            Noise parameters, shape (x_dim,).
        kernel : MultiGroupGPKernel, optional
            Kernel instance. Defaults to RBFKernel().
        hyper_params : MultiGroupGPHyperParams, optional
            Hyperparameters for parameter constraints. Defaults to default values.
        """
        # Create MultiGroupGPParams from direct parameters
        # (convert to JAX for internal use)
        import jax.numpy as jnp

        self.params = MultiGroupGPParams(
            gamma=jnp.array(gamma, dtype=jnp.float64),
            delays=jnp.array(delays, dtype=jnp.float64),
            eps=jnp.array(eps, dtype=jnp.float64),
        )

        # Set default kernel if none provided
        self.kernel = kernel or RBFKernel()

        # Set default hyper_params if none provided
        self.hyper_params = hyper_params or MultiGroupGPHyperParams()

    @classmethod
    def generate(
        cls,
        x_dim: int,
        num_groups: int,
        delay_lim: tuple[float, float] = (-5.0, 5.0),
        eps_lim: tuple[float, float] = (0.001, 0.001),
        gamma_lim: tuple[float, float] = (0.01, 0.5),
        rng: np.random.Generator | None = None,
        hyper_params: MultiGroupGPHyperParams | None = None,
        kernel: MultiGroupGPKernel | None = None,
    ):
        """Generate random mDLAGGP with random parameters.

        Parameters
        ----------
        x_dim : int
            Number of latent dimensions.
        num_groups : int
            Number of groups.
        delay_lim : tuple[float, float], optional
            Limits for delay parameters. Defaults to (-5.0, 5.0).
        eps_lim : tuple[float, float], optional
            Limits for noise parameters. Defaults to (0.001, 0.001).
        gamma_lim : tuple[float, float], optional
            Limits for length scale parameters. Defaults to (0.01, 0.5).
        rng : np.random.Generator, optional
            Random number generator. Defaults to None.
        hyper_params : MultiGroupGPHyperParams, optional
            Hyperparameters for parameter constraints. Defaults to None.
        kernel : MultiGroupGPKernel, optional
            Kernel instance. Defaults to None (uses RBFKernel).

        Returns
        -------
        mDLAGGP
            New mDLAGGP instance with random parameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Use hyper_params limits if provided
        if hyper_params is not None:
            gamma_lim = (
                hyper_params.min_gamma,
                gamma_lim[1],
            )  # Use min_gamma from hyper_params
            delay_lim = (
                -hyper_params.max_delay,
                hyper_params.max_delay,
            )  # Use max_delay from hyper_params

        gamma = rng.uniform(gamma_lim[0], gamma_lim[1], size=x_dim).astype(np.float64)
        eps = rng.uniform(eps_lim[0], eps_lim[1], size=x_dim).astype(np.float64)
        delays = rng.uniform(
            delay_lim[0], delay_lim[1], size=(num_groups, x_dim)
        ).astype(np.float64)
        delays[0, :] = 0  # Set first group delays to 0

        return cls(
            gamma=gamma,
            delays=delays,
            eps=eps,
            kernel=kernel,
            hyper_params=hyper_params,
        )

    def build_kernel_matrix(
        self, T: int, return_tensor: bool = False, order: str = "F"
    ) -> np.ndarray:
        """Build full kernel matrix across all dimensions.

        Parameters
        ----------
        T : int
            Number of time points.
        return_tensor : bool, optional
            If True, return 6D tensor. If False, return flattened matrix.
            Defaults to False.
        order : str, optional
            Order for reshaping ('F' for Fortran, 'C' for C). Defaults to 'F'.

        Returns
        -------
        np.ndarray
            Kernel matrix with shape depending on return_tensor.
        """
        kernel_matrix = self.kernel.build_full_kernel_matrix(
            self.params, T, return_tensor, order
        )
        return np.array(kernel_matrix)

    def compute_loss(self, X_moment: np.ndarray, N: int, T: int) -> float:
        """Compute total loss across all latent dimensions.

        Parameters
        ----------
        X_moment : np.ndarray
            Moment data for all latent dimensions, shape (x_dim, num_groups*T,
            num_groups*T).
        N : int
            Number of samples.
        T : int
            Number of time points.

        Returns
        -------
        float
            Total loss value.
        """
        import jax.numpy as jnp

        X_moment_jax = jnp.array(X_moment, dtype=jnp.float64)

        total_loss = 0.0
        for i in range(self.params.x_dim):
            # Convert JAX arrays to numpy arrays to avoid JIT hashability issues
            gamma_i = np.array(self.params.gamma[i])
            delays_i = np.array(self.params.delays[:, i])
            eps_i = float(
                self.params.eps[i]
            )  # Convert to Python float for static argument

            Ki = self.kernel.build_single_latent_kernel(
                gamma_i, delays_i, eps_i, T, return_tensor=False
            )
            X_moment_i = X_moment_jax[i, :, :]
            total_loss += self.kernel.compute_elbo_from_kernel_matrix(Ki, X_moment_i, N)
        return float(total_loss)

    def fit(
        self,
        X_moment: np.ndarray,
        N: int,
        T: int,
        config: GPFitConfig = GPFitConfig(),
    ) -> tuple[mDLAGGP, float]:
        """Fit the GP parameters using the provided moment data.

        This method delegates to the existing `run_gp_optimizer` function
        and updates the model's internal parameters with the optimized values.

        Parameters
        ----------
        X_moment : np.ndarray
            Moment data for all latent dimensions, shape (x_dim, num_groups*T,
            num_groups*T).
        N : int
            Number of samples.
        T : int
            Number of time points.
        config : GPFitConfig, optional
            Configuration for the optimization. Defaults to default GPFitConfig.

        Returns
        -------
        tuple[mDLAGGP, float]
            Updated mDLAGGP instance and total loss value.

        Notes
        -----
        The model's internal parameters (`self.params`) are updated with the
        result of the optimization.
        """
        import jax.numpy as jnp

        X_moment_jax = jnp.array(X_moment, dtype=jnp.float64)

        # Delegate the actual optimization to the existing function
        updated_params, total_loss = run_gp_optimizer(
            self.params,
            self.kernel,
            X_moment_jax,
            N,
            T,
            config,
            self.hyper_params,
        )

        # Update the model's parameters
        self.params = updated_params

        return self, total_loss

    def copy(self) -> mDLAGGP:
        """Create a copy of the mDLAGGP instance.

        Returns
        -------
        mDLAGGP
            A new instance with the same parameters.
        """
        return mDLAGGP(
            gamma=np.array(self.params.gamma),
            delays=np.array(self.params.delays),
            eps=np.array(self.params.eps),
            kernel=self.kernel,
            hyper_params=self.hyper_params,
        )

    def __repr__(self) -> str:
        """Return string representation of mDLAGGP."""
        return (
            f"mDLAGGP("
            f"gamma={self.params.gamma}, "
            f"delays={self.params.delays}, "
            f"eps={self.params.eps}, "
            f"kernel={self.kernel}, "
            f"hyper_params={self.hyper_params})"
        )
