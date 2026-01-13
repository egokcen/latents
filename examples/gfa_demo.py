"""
Group Factor Analysis (GFA) Demo
================================

This example demonstrates how to use Group Factor Analysis (GFA) to
discover shared and group-specific latent structure in multi-group data.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np

import latents.gfa.analysis as gfa_stats
import latents.gfa.simulation as gfa_sim
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.gfa.inference import compute_lower_bound
from latents.observation import ObsParamsHyperPriorStructured, ObsParamsPosterior
from latents.plotting import (
    hinton_diagram,
    plot_dimensionalities,
    plot_dims_pairs,
    plot_var_exp,
    plot_var_exp_pairs,
)

# %%
# Generate Data from the GFA Model
# ---------------------------------
#
# First, we set up the simulation parameters and generate synthetic data
# with known ground truth.

# Dataset characteristics
n_samples = 100  # Total number of samples
y_dims = np.array([10, 10, 10])  # Dimensionality of each observed group
n_groups = len(y_dims)
x_dim = 7  # Latent dimensionality

# Build the desired sparsity pattern of the loading matrices.
# A (n_groups x x_dim) array where row i corresponds to group i,
# column j corresponds to latent j. A value of np.inf indicates that
# a latent is NOT present in a group.
sparsity_pattern = np.array(
    [
        [1, 1, 1, np.inf, 1, np.inf, np.inf],
        [1, 1, np.inf, 1, np.inf, 1, np.inf],
        [1, np.inf, 1, 1, np.inf, np.inf, 1],
    ],
)

# Set up simulation hyperprior
MAG = 100  # Control the variance of alpha parameters (larger = less variance)
hyperprior = ObsParamsHyperPriorStructured(
    a_alpha=MAG * sparsity_pattern,
    b_alpha=MAG * np.ones_like(sparsity_pattern),
    a_phi=1.0,
    b_phi=1.0,
    beta_d=1.0,
)

# Configure simulation
sim_config = GFASimConfig(
    n_samples=n_samples,
    y_dims=y_dims,
    x_dim=x_dim,
    snr=1.0,
    random_seed=0,
)

# Simulate data
sim_result = gfa_sim.simulate(sim_config, hyperprior)
Y = sim_result.observations
obs_params_true = sim_result.obs_params

# %%
# Fit a GFA Model
# ---------------
#
# Now we instantiate and fit a GFA model to the simulated data.

# Configure fitting
config = GFAFitConfig(
    x_dim_init=10,  # Set larger than the hypothesized latent dimensionality
    fit_tol=1e-8,
    max_iter=20000,
    verbose=True,
    random_seed=0,
    min_var_frac=0.001,
    prune_x=True,
    prune_tol=1e-7,
    save_x=True,  # Required for compute_lower_bound
    save_c_cov=True,  # Required for compute_lower_bound
    save_fit_progress=True,
)

# Instantiate a GFA model with config
model = GFAModel(config=config)

# Fit the model (automatically initializes if needed)
model.fit(Y)

# %%
# Check Fitting Results
# ---------------------
#
# Display fitting diagnostics and convergence information.

# Display flags indicating fitting procedure status
model.flags.display()

# Plot the lower bound and cumulative runtime at each iteration
model.tracker.plot_lb()
model.tracker.plot_runtime()

# %%
# Visualize Loading Matrix Recovery
# ---------------------------------
#
# Compare the estimated loading matrices with ground truth.
# Note: Columns may be reordered and sign-flipped.

# Define column reordering and sign flips for comparison
reorder = np.array([1, 4, 3, 6, 0, 2, 5])
rescale = np.array([-1, -1, 1, 1, 1, 1, 1])

# Ground truth
plt.figure(figsize=(3, 5))
plt.subplot(1, 2, 1)
plt.title("Ground truth C")
hinton_diagram(obs_params_true.C)

# Plot estimated C, reordered and rescaled to match ground truth
plt.subplot(1, 2, 2)
plt.title("Estimated C")
hinton_diagram(model.obs_posterior.C.mean[:, reorder] * rescale)

plt.tight_layout()
plt.show()

# %%
# Visualize ARD Parameter Recovery
# --------------------------------
#
# Compare the estimated ARD (automatic relevance determination) parameters
# with ground truth.

# Compute relative shared variance explained by each latent in each group
alpha_inv_true = 1 / obs_params_true.alpha
alpha_inv_rel_true = alpha_inv_true / np.sum(alpha_inv_true, axis=1, keepdims=True)

alpha_inv_est = 1 / model.obs_posterior.alpha.mean
alpha_inv_rel_est = alpha_inv_est / np.sum(alpha_inv_est, axis=1, keepdims=True)

# Ground truth
plt.figure(figsize=(5, 3))
plt.subplot(2, 1, 1)
plt.title("Ground truth ARD")
hinton_diagram(alpha_inv_rel_true)

# Plot estimated ARD, reordered to match ground truth
plt.subplot(2, 1, 2)
plt.title("Estimated ARD")
hinton_diagram(alpha_inv_rel_est[:, reorder])

plt.tight_layout()
plt.show()

# %%
# Performance Metrics
# -------------------
#
# Evaluate the model using the evidence lower bound and predictive performance.

# Evidence lower bound
lb = compute_lower_bound(
    Y, model.obs_posterior, model.latents_posterior, model.obs_hyperprior
)
print(f"Lower bound:         {lb:.4f}")

# Leave-group-out prediction
R2, MSE = gfa_stats.predictive_performance(Y, model.obs_posterior)
print(f"Leave-group-out R²:  {R2:.4f}")
print(f"                MSE: {MSE:.4f}")

# Leave-one-out prediction
R2, MSE = gfa_stats.predictive_performance(
    Y, model.obs_posterior, y_dims=np.ones(y_dims.sum(), dtype=int)
)
print(f"Leave-one-out R²:    {R2:.4f}")
print(f"              MSE:   {MSE:.4f}")

# %%
# Signal-to-Noise Ratios
# ----------------------
#
# Examine the estimated signal-to-noise ratios for each group.

snr_est = model.obs_posterior.compute_snr()

print("Estimated SNRs:")
for group_idx in range(n_groups):
    print(f"    Group {group_idx + 1}: {snr_est[group_idx]:.4f}")

# %%
# Dimensionality Analysis
# -----------------------
#
# Determine and visualize the dimensionalities of different latent types.

# Compute dimensionalities and shared variance explained
(
    num_dim,
    sig_dims,
    var_exp,
    dim_types,
) = model.obs_posterior.compute_dimensionalities(
    cutoff_shared_var=0.02, cutoff_snr=0.001
)

# Visualize the number of each type of dimension
plt.figure(figsize=(4, 2))
plot_dimensionalities(
    num_dim, dim_types, group_names=["A", "B", "C"], plot_zero_dim=False
)
plt.show()

# Visualize the shared variance explained by each dimension type
plt.figure(figsize=(4, 6))
plot_var_exp(var_exp, dim_types, group_names=["A", "B", "C"], plot_zero_dim=False)
plt.show()

# %%
# Pairwise Analysis
# -----------------
#
# Analyze interactions between pairs of groups.

# Compute pairwise dimensionalities and shared variances
pair_dims, pair_var_exp, pairs = ObsParamsPosterior.compute_dims_pairs(
    num_dim, dim_types, var_exp
)

# Visualize pairwise dimensionalities
plt.figure(figsize=(7, 2.5))
plot_dims_pairs(pair_dims, pairs, n_groups, group_names=["A", "B", "C"])
plt.show()

# Visualize pairwise shared variances
plt.figure(figsize=(5, 2.5))
plot_var_exp_pairs(
    pair_var_exp, pairs, n_groups, group_names=["A", "B", "C"], sem_pair_var_exp=None
)
plt.show()
