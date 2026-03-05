(getting-started)=

# Getting started

## Installation

Latents requires Python 3.10 or higher.

```bash
pip install latents
```

For development setup, see the {doc}`Contributing guide </development/contributing>`.

## Quickstart

This example simulates multi-group data from a Group Factor Analysis (GFA) model and
fits it to recover the latent structure. For a detailed walkthrough of the generative
model, see {ref}`sphx_glr_auto_examples_gfa_plot_1_simulation.py`.

```python
import numpy as np

from latents.callbacks import ProgressCallback
from latents.gfa import GFAFitConfig, GFAModel
from latents.gfa.config import GFASimConfig
from latents.gfa.simulation import simulate
from latents.observation import ObsParamsHyperPriorStructured

# --- Simulate data ---

# Sparsity pattern: rows are groups, columns are latent factors.
# Finite values indicate a factor is present; np.inf indicates absence.
sparsity_pattern = np.array([
    [1, 1, np.inf],   # Group 0: factors 0, 1
    [1, np.inf, 1],   # Group 1: factors 0, 2
])

MAG = 100
hyperprior = ObsParamsHyperPriorStructured(
    a_alpha=MAG * sparsity_pattern,
    b_alpha=MAG * np.ones_like(sparsity_pattern),
    a_phi=1.0,
    b_phi=1.0,
    beta_d=1.0,
)

sim_config = GFASimConfig(
    y_dims=np.array([8, 8]),
    x_dim=3,
    n_samples=200,
    random_seed=42,
)

result = simulate(sim_config, hyperprior)
Y = result.observations  # ObsStatic with stacked (sum(y_dims), n_samples) array

# --- Fit model ---

config = GFAFitConfig(x_dim_init=6)  # Start with more dims than needed
model = GFAModel(config=config)
model.fit(Y, callbacks=[ProgressCallback()])

# --- Inspect results ---

model.flags.display()

# Discovered sparsity pattern: which factors are significant in each group
_, sig_dims, _, _ = model.obs_posterior.compute_dimensionalities()
print(sig_dims.astype(int))
```

GFA uses automatic relevance determination (ARD) to prune unnecessary
dimensions during fitting — starting with `x_dim_init=6`, the model
discovers that 3 latent dimensions explain the data. The recovered sparsity
pattern matches the input (up to column reordering, which is inherent to
factor models).

## Next steps

{ref}`Fitting GFA <sphx_glr_auto_examples_gfa_plot_2_fitting.py>`
: Detailed fitting example with parameter recovery against known ground truth.

{doc}`User guide </user_guide/index>`
: Conceptual foundations and method details — mathematics, design philosophy, and model specifications.

{doc}`Examples </auto_examples/index>`
: Full gallery covering simulation, fitting, posterior sampling, and production workflows.

{doc}`API reference </reference/index>`
: Complete documentation of all classes and functions.
