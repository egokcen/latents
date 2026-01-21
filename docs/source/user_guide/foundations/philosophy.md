(philosophy)=

# Design philosophy

Latents is designed around a consistent probabilistic hierarchy that reflects
the generative structure of latent variable models. This page explains the
conceptual framework and how it shapes the package design.

## The probabilistic hierarchy

Every method in Latents is a probabilistic generative model with a hierarchical
structure:

::::{div} mermaid-md
```{mermaid}
flowchart TB
    subgraph Hyperpriors
        OH["Observation hyperpriors<br/>(a_α, b_α, a_φ, b_φ, β)"]
        SH["State hyperpriors<br/>(GP kernel params)"]
    end

    subgraph Priors
        OP["Observation priors<br/>α ~ Gamma, φ ~ Gamma<br/>d ~ N, C|α ~ N"]
        SP["State prior<br/>Static: X ~ N(0, I)<br/>GP: X ~ GP(0, K)"]
    end

    subgraph Realizations
        OR["Observation parameters<br/>(C, d, φ, α values)"]
        SR["Latents<br/>(X values)"]
    end

    OH --> OP
    SH -.-> SP
    OP --> OR
    SP --> SR
    OR --> Y["Observations (Y)"]
    SR --> Y
```
::::

| Level | What it represents | Examples |
|-------|-------------------|----------|
| **Hyperpriors** | Parameters of prior distributions | $a_\alpha$, $b_\alpha$ (Gamma prior shape/rate) |
| **Priors** | The prior distributions themselves | $\alpha \sim \text{Gamma}(a_\alpha, b_\alpha)$ |
| **Realizations** | Concrete sampled values | Specific values of $\alpha$, $\mathbf{C}$, $\mathbf{X}$ |

This hierarchy flows in the generative (forward) direction. Inference flows in
reverse: given observations, we compute posterior distributions over latents
and parameters.

### Hyperpriors

Hyperpriors parametrize the prior distributions over model parameters.

**Observation hyperpriors** control priors on $\mathbf{C}$, $\mathbf{d}$,
$\phi$, and $\alpha$. Two levels of control are available:

- **Homogeneous** — scalar hyperpriors broadcast to all groups/latents
- **Structured** — per-group, per-latent control (enables sparsity constraints)

**State hyperpriors** depend on the state model. Static methods have no state
hyperpriors (the prior $\mathbf{X} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is
fixed). GP-based methods have kernel hyperpriors (timescale, variance).

### Priors

Prior classes encapsulate both the distribution specification and the ability
to sample from it. For example, `ObsParamsPrior`
handles the correct sampling order ($\alpha$ must be sampled before $\mathbf{C}$,
since $\mathbf{C} \mid \alpha \sim \mathcal{N}(\mathbf{0}, \alpha^{-1})$).

### Realizations

A realization is a single set of parameter values—concrete arrays rather than
distributions. Realizations arise from multiple sources:

- **Prior sampling** — ground truth for simulation
- **Posterior means** — point estimates from Bayesian inference
- **Posterior samples** — draws for uncertainty quantification
- **Optimization** — point estimates from non-Bayesian methods

### Posteriors

Posteriors are the result of inference—distributions over parameters and latents
conditioned on observed data. They provide:

- Statistics (mean, covariance, moments)
- Sampling via `.sample()` methods
- Point estimates via `.posterior_mean` properties

## Design principles

### Methods as compositions

Each method composes a state prior and observation prior. This determines the
method's behavior:

- **GFA** = static latents + Bayesian observation parameters with ARD
- **GPFA** = GP latents + point-estimate observation parameters
- **mDLAG** = GP latents with delays + Bayesian observation parameters

This composition pattern enables code reuse across methods.

### Explicit probabilistic levels

Hyperpriors, priors, and realizations are distinct classes with clear roles:

- **Hyperpriors** are configuration (frozen dataclasses)
- **Priors** encapsulate distribution specification and sampling
- **Realizations** are concrete values for computation

This explicitness prevents confusion between "the distribution" and "a sample
from it."

### Realizations vs distributions

A realization is a single set of values; a posterior is a distribution with
statistics. This distinction enables:

- **Posterior sampling** — `posterior.sample()` draws from the distribution
- **Clean data generation** — `sample_observations()` takes values, not distributions
- **Semantic clarity** — prior samples and posterior means are both realizations,
  but their provenance differs

### Universal data generation

A method's `sample_observations()` function accepts realizations from any source:

```python
# From prior sampling (simulation)
Y = sample_observations(latents, obs_params, rng)

# From posterior mean (point prediction)
Y = sample_observations(latents, model.obs_posterior.posterior_mean, rng)

# From posterior sample (uncertainty quantification)
Y = sample_observations(latents, model.obs_posterior.sample(rng), rng)
```

This decouples data generation from parameter provenance. See
{ref}`sphx_glr_auto_examples_gfa_plot_1_simulation.py` for a step-by-step example.

### Tiered API

Most users interact only with the method-level API:

```python
from latents.gfa import GFAModel, GFAFitConfig

model = GFAModel(config=GFAFitConfig(x_dim_init=10))
model.fit(Y)
```

The component-level API (`observation/`, `state/`) is available for advanced
use cases like custom simulation with structured hyperpriors.

## Further reading

- {doc}`/development/architecture` — package structure and how to extend Latents
- {doc}`../methods/gfa` — applying these concepts to GFA
