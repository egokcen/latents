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

import latents.gfa.descriptive_stats as gfa_stats
import latents.gfa.simulation as gfa_sim
from latents.gfa import GFAFitConfig, GFAModel
from latents.observation_model.probabilistic import ObsParamsARD, SimulationHyperPriors
from latents.plotting import hinton

# %%
# Generate Data from the GFA Model
# ---------------------------------
#
# First, we set up the simulation parameters and generate synthetic data
# with known ground truth.

# Set a random seed for reproducibility
random_seed = 1

# Dataset characteristics
N = 100  # Total number of samples
y_dims = np.array([10, 10, 10])  # Dimensionality of each observed group
num_groups = len(y_dims)
x_dim = 7  # Latent dimensionality
snr = 1.0 * np.ones(num_groups)  # Signal-to-noise ratio of each group

# Build the desired sparsity pattern of the loading matrices.
# A (num_groups x x_dim) array where row i corresponds to group i,
# column j corresponds to latent j. A value of np.inf indicates that
# a latent is NOT present in a group.
sparsity_pattern = np.array(
    [
        [1, 1, 1, np.inf, 1, np.inf, np.inf],
        [1, 1, np.inf, 1, np.inf, 1, np.inf],
        [1, np.inf, 1, 1, np.inf, np.inf, 1],
    ],
)

# Set up simulation hyperpriors
MAG = 100  # Control the variance of alpha parameters (larger = less variance)
sim_priors = SimulationHyperPriors(
    a_alpha=MAG * sparsity_pattern,
    b_alpha=MAG * np.ones_like(sparsity_pattern),
    a_phi=1.0,
    b_phi=1.0,
    d_beta=1.0,
)

# Simulate data
Y, X_true, obs_params_true = gfa_sim.simulate(
    N,
    y_dims,
    x_dim,
    sim_priors,
    snr,
    random_seed=random_seed,
)

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

# Initialize and fit the model
model.init(Y)
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
reorder = np.array([4, 2, 1, 5, 3, 6, 0])
rescale = np.array([-1, -1, 1, -1, 1, -1, 1])

# Ground truth
plt.figure(figsize=(3, 5))
plt.subplot(1, 2, 1)
plt.title("Ground truth C")
hinton(obs_params_true.C.mean)

# Plot estimated C, reordered and rescaled to match ground truth
plt.subplot(1, 2, 2)
plt.title("Estimated C")
hinton(model.params.obs_params.C.mean[:, reorder] * rescale)

plt.tight_layout()
plt.show()

# %%
# Visualize ARD Parameter Recovery
# --------------------------------
#
# Compare the estimated ARD (automatic relevance determination) parameters
# with ground truth.

# Compute relative shared variance explained by each latent in each group
alpha_inv_true = 1 / obs_params_true.alpha.mean
alpha_inv_rel_true = alpha_inv_true / np.sum(alpha_inv_true, axis=1, keepdims=True)

alpha_inv_est = 1 / model.params.obs_params.alpha.mean
alpha_inv_rel_est = alpha_inv_est / np.sum(alpha_inv_est, axis=1, keepdims=True)

# Ground truth
plt.figure(figsize=(5, 3))
plt.subplot(2, 1, 1)
plt.title("Ground truth ARD")
hinton(alpha_inv_rel_true)

# Plot estimated ARD, reordered to match ground truth
plt.subplot(2, 1, 2)
plt.title("Estimated ARD")
hinton(alpha_inv_rel_est[:, reorder])

plt.tight_layout()
plt.show()

# %%
# Performance Metrics
# -------------------
#
# Evaluate the model using the evidence lower bound and predictive performance.

# Evidence lower bound
lb = model.compute_lower_bound(Y)
print(f"Lower bound:         {lb:.4f}")

# Leave-group-out prediction
R2, MSE = gfa_stats.predictive_performance(Y, model.params)
print(f"Leave-group-out R²:  {R2:.4f}")
print(f"                MSE: {MSE:.4f}")

# Leave-one-out prediction
R2, MSE = gfa_stats.predictive_performance(
    Y, model.params, y_dims=np.ones(y_dims.sum(), dtype=int)
)
print(f"Leave-one-out R²:    {R2:.4f}")
print(f"              MSE:   {MSE:.4f}")

# %%
# Signal-to-Noise Ratios
# ----------------------
#
# Examine the estimated signal-to-noise ratios for each group.

snr_est = model.params.obs_params.compute_snr()

print("Estimated SNRs:")
for group_idx in range(num_groups):
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
) = model.params.obs_params.compute_dimensionalities(
    cutoff_shared_var=0.02, cutoff_snr=0.001
)

# Visualize the number of each type of dimension
plt.figure(figsize=(4, 2))
ObsParamsARD.plot_dimensionalities(
    num_dim, dim_types, group_names=["A", "B", "C"], plot_zero_dim=False
)
plt.show()

# Visualize the shared variance explained by each dimension type
plt.figure(figsize=(4, 6))
ObsParamsARD.plot_var_exp(
    var_exp, dim_types, group_names=["A", "B", "C"], plot_zero_dim=False
)
plt.show()

# %%
# Pairwise Analysis
# -----------------
#
# Analyze interactions between pairs of groups.

# Compute pairwise dimensionalities and shared variances
pair_dims, pair_var_exp, pairs = ObsParamsARD.compute_dims_pairs(
    num_dim, dim_types, var_exp
)

# Visualize pairwise dimensionalities
plt.figure(figsize=(7, 2.5))
ObsParamsARD.plot_dims_pairs(pair_dims, pairs, num_groups, group_names=["A", "B", "C"])
plt.show()

# Visualize pairwise shared variances
plt.figure(figsize=(5, 2.5))
ObsParamsARD.plot_var_exp_pairs(
    pair_var_exp, pairs, num_groups, group_names=["A", "B", "C"], sem_pair_var_exp=None
)
plt.show()
