(gfa-method)=

# Group factor analysis (GFA)

Group factor analysis is a Bayesian dimensionality reduction method for
multi-group data. It automatically determines:

1. How many latent dimensions describe the data
2. Which groups each latent dimension involves

## Model specification

GFA uses a static state model with Bayesian observation parameters.

**State model.** Latent variables are independent across samples:

$$p(\mathbf{X}) = \prod_{n=1}^{N} \mathcal{N}(\mathbf{x}_n \mid \mathbf{0}, \mathbf{I})$$

**Observation model.** Each group $m$ has its own loading matrix and noise:

$$\mathbf{y}^{(m)} = \mathbf{C}^{(m)} \mathbf{x} + \mathbf{d}^{(m)} + \boldsymbol{\epsilon}^{(m)}$$

where $\boldsymbol{\epsilon}^{(m)} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\phi}^{(m)})^{-1})$.
The diagonal noise covariance (anisotropic noise) encourages latent variables
to explain shared variance among observed dimensions.

## Automatic relevance determination

GFA uses ARD priors to automatically select the number of latent dimensions.
Each loading column has a precision parameter $\alpha_j^{(m)}$ with a Gamma
prior. During inference:

- If latent $j$ is **needed** for group $m$: $\alpha_j^{(m)}$ stays small,
  allowing non-zero loadings
- If latent $j$ is **not needed**: $\alpha_j^{(m)}$ grows large, shrinking
  the loading column toward zero

This mechanism prunes unnecessary dimensions automatically. Initialize with
more latent dimensions than expected (`x_dim_init`), and GFA will prune to
the effective dimensionality.

## Inference

GFA uses mean-field variational inference, maximizing the evidence lower bound
(ELBO). Convergence is monitored via the ELBO, which is guaranteed to be
non-decreasing.

```python
from latents.gfa import GFAModel, GFAFitConfig
from latents.callbacks import ProgressCallback

model = GFAModel(config=GFAFitConfig(x_dim_init=10))
model.fit(Y, callbacks=[ProgressCallback()])

# Check convergence
model.tracker.plot_lb()
print(f"Effective dimensionality: {model.latents_posterior.x_dim}")
```

See {ref}`sphx_glr_auto_examples_gfa_plot_2_fitting.py` for a complete example.

## Interpreting results

### Variance explained

The fraction of shared variance explained by latent $j$ in group $m$ is:

$$\nu_j^{(m)} = \frac{\langle\alpha_j^{(m)}\rangle^{-1}}{\sum_k \langle\alpha_k^{(m)}\rangle^{-1}}$$

A latent is considered significant in a group if it explains at least
some threshold of shared variance (e.g., $\nu_j^{(m)} \geq 0.02$), though this threshold
depends on your use case.

```python
# Compute dimensionalities and variance explained
num_dim, sig_dims, var_exp, dim_types = model.obs_posterior.compute_dimensionalities(
    cutoff_shared_var=0.02,  # significance threshold
)

# Visualize
from latents.plotting import plot_dimensionalities, plot_var_exp
plot_dimensionalities(num_dim, dim_types, group_names=["A", "B", "C"])
plot_var_exp(var_exp, dim_types, group_names=["A", "B", "C"])
```

### Leave-group-out prediction

Leave-group-out prediction measures how well GFA captures cross-group
interactions. For each group, predict its activity using only the remaining
groups. The leave-group-out $R^2$ quantifies prediction accuracy:

- $R^2_{\text{lgo}} = 1$: perfect prediction
- $R^2_{\text{lgo}} < 0$: worse than predicting the mean

```python
from latents.gfa.analysis import predictive_performance

# Evaluate on held-out test data (Y_test is an ObsStatic instance)
R2, MSE = predictive_performance(Y_test, model.obs_posterior)
print(f"Leave-group-out R²: {R2:.3f}")
```

:::{seealso}
- {ref}`sphx_glr_auto_examples_gfa_plot_2_fitting.py`
- {ref}`sphx_glr_auto_examples_gfa_plot_3_posteriors.py`
:::

## References

Latents implements GFA with anisotropic observation noise:

> Gokcen, E., Jasper, A. I., Xu, A., Kohn, A., Machens, C. K. & Yu, B. M.
> Uncovering motifs of concurrent signaling across multiple neuronal
> populations. *Advances in Neural Information Processing Systems* **36**,
> 34711-34722 (2023).
> [Paper link](https://neurips.cc/virtual/2023/poster/70171)

The original GFA formulation with isotropic noise:

> Klami, A., Virtanen, S., Leppäaho, E. & Kaski, S. Group factor analysis.
> *IEEE Transactions on Neural Networks and Learning Systems* **26**,
> 2136-2147 (2015).
> [DOI: 10.1109/tnnls.2014.2376974](https://doi.org/10.1109/tnnls.2014.2376974)
