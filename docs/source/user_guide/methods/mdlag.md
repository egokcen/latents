(mdlag-method)=

# Delayed latents across groups (mDLAG)

Delayed latents across multiple groups (mDLAG) is a Bayesian dimensionality
reduction method for multi-group time series data. It addresses two challenges:

1. **Identifying network-level interactions**
  - How many latent dimensions describe the data
  - Which groups each latent dimension involves
2. **Disentangling concurrent signal flow**
  - The direction of signal flow across groups
  - How signals evolve over time and across trials (samples)

:::{note}
mDLAG is under active development. The current implementation provides stubs
for the planned API.
:::

## Model specification

**Observation model.** For group $m$ at time $t$ on trial $n$:

$$\mathbf{y}^{(m)}_{n,t} = \mathbf{C}^{(m)} \mathbf{x}^{(m)}_{n,t} + \mathbf{d}^{(m)} + \boldsymbol{\epsilon}^{(m)}$$

where $\boldsymbol{\epsilon}^{(m)} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\phi}^{(m)})^{-1})$.

This is the same linear-Gaussian structure as GFA, but now indexed by time $t$
and trial (sample) $n$. The latents $\mathbf{x}^{(m)}_{n,t}$ are coupled across groups
at each time point.

**State model.** Unlike GFA's i.i.d. prior, mDLAG places a Gaussian process
prior over latents to capture smooth temporal dynamics:

$$\mathbf{x}_{n,j,:} \sim \mathcal{GP}(0, K_j)$$

Each latent $j$ has:
- A **timescale** $\tau_j$ controlling smoothness
- **Time delays** $D_j^{(m)}$ describing the lead-lag relationship between groups

## Time delays and signal flow

Time delays are the key distinguishing feature of mDLAG. For each latent $j$, the
relative delay between groups indicates signal flow direction:

$$\Delta D_j = D_j^{(m_2)} - D_j^{(m_1)}$$

- $\Delta D_j > 0$: group $m_1$ **leads** group $m_2$
- $\Delta D_j < 0$: group $m_2$ **leads** group $m_1$
- $\Delta D_j = 0$: simultaneous activity

Time delays are continuous-valued—not restricted to discrete time bins. By
convention, group 1 is the reference ($D_j^{(1)} = 0$ for all latents).

## Automatic relevance determination

Like GFA, mDLAG uses ARD priors to automatically determine:
- The total number of latent dimensions
- Which groups each latent involves

See {doc}`gfa` for details on how ARD enables automatic dimensionality selection.

## References

> Gokcen, E., Jasper, A. I., Xu, A., Kohn, A., Machens, C. K. & Yu, B. M.
> Uncovering motifs of concurrent signaling across multiple neuronal
> populations. *Advances in Neural Information Processing Systems* **36**,
> 34711-34722 (2023).
> [Paper link](https://neurips.cc/virtual/2023/poster/70171)
