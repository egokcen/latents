(mathematical-background)=

# Mathematical background

All methods in Latents belong to the same family of probabilistic generative
models. This page describes the shared mathematical framework that unifies them.

## The generative model framework

Each method defines two components:

1. **State model** — the distribution over latent variables: $p(\mathbf{x})$
2. **Observation model** — how observations arise from latents: $p(\mathbf{y} \mid \mathbf{x})$

A specific method is a *composition* of choices for each component. The package
architecture mirrors this structure: shared building blocks are organized by
model component, and each method assembles the blocks it needs.

::::{div} mermaid-sm
```{mermaid}
flowchart LR
    subgraph Latents
        X(("x"))
    end
    subgraph Observations
        Y(("y"))
    end
    X --> Y
```
::::

## Axes of variation

Four orthogonal axes distinguish methods within the family:

| Axis | Options | Affects |
|------|---------|---------|
| Temporal structure | Static ↔ Time series | State model |
| Number of groups | 1 → 2 → N | Observation model |
| Noise structure | Isotropic ↔ Anisotropic | Observation model |
| Inference approach | Point estimates ↔ Bayesian | Parameter treatment |

### Temporal structure

**Static methods** (pPCA, FA, pCCA, GFA) assume latent variables are independent
across samples:

$$p(\mathbf{x}) = \prod_{n=1}^{N} \mathcal{N}(\mathbf{x}_n \mid \mathbf{0}, \mathbf{I})$$

**Time series methods** (GPFA, LDS, DLAG, mDLAG) model temporal dependencies.
For example, GPFA places a Gaussian process prior over each latent dimension,
capturing smooth temporal dynamics.

### Number of groups

Single-group methods model one set of observations. Multi-group methods model
multiple observation sets that share latent structure—for example, simultaneous
recordings from different brain regions.

Moving from one to multiple groups primarily affects the observation model: each
group $m$ has its own loading matrix $\mathbf{C}^{(m)}$ and noise parameters,
but all groups share the same latents $\mathbf{x}$.

### Noise structure

The observation model typically takes the form:

$$\mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{d} + \boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon}$ is Gaussian noise.

- **Isotropic** (pPCA): $\text{Cov}(\boldsymbol{\epsilon}) = \sigma^2 \mathbf{I}$
- **Anisotropic** (FA and beyond): $\text{Cov}(\boldsymbol{\epsilon}) = \text{diag}(\phi_1^{-1}, \ldots, \phi_D^{-1})$

Anisotropic noise allows a separate precision $\phi_i$ for each observed
dimension.

### Inference approach

**Non-Bayesian methods** (FA, pCCA, GPFA, DLAG) treat observation model
parameters as point estimates found via maximum likelihood or EM.

**Bayesian methods** (GFA, mDLAG) place priors over parameters and compute
posterior distributions. ARD (automatic relevance determination) priors on the
loading matrix columns enable automatic pruning of unnecessary latent dimensions
during inference.

## Further reading

- {doc}`../methods/index` — method family overview and how to choose
- {doc}`philosophy` — how this framework maps to code
